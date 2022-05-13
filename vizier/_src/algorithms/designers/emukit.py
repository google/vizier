"""EmukitDesigner wraps emukit into Vizier Designer."""

import enum
from typing import Optional, Sequence

from absl import logging
from emukit import core
from emukit import model_wrappers
from emukit.bayesian_optimization import acquisitions
from emukit.bayesian_optimization import loops
from emukit.core import initial_designs
from GPy import kern
from GPy import models
from GPy.core.parameterization import priors
from GPy.util import linalg
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.pyvizier import converters

RandomDesign = initial_designs.RandomDesign


def _create_constrained_gp(features: np.ndarray, labels: np.ndarray):
  """Creates a constraint gp."""

  # This logging is too chatty because paramz transformations do not implement
  # log jacobians. Silence it.
  logging.logging.getLogger('paramz.transformations').setLevel(
      logging.logging.CRITICAL)

  class LogGaussian:
    """Multi-variate version of Loggaussian.

    GPy surprisingly doesn't have this. The expected API of lnpdf and lnpdf_grad
    are not precisely defined, so this handwaves a lot of stuff based on how
    MultiVariateGaussian is implemented.
    """
    domain = 'positive'

    def __init__(self, mu, sigma):
      self.mu = (mu)
      self.sigma = (sigma)
      self.inv, _, self.hld, _ = linalg.pdinv(self.sigma)
      self.sigma2 = np.square(self.sigma)
      self.constant = -0.5 * (self.mu.size * np.log(2 * np.pi) + self.hld)

    def lnpdf(self, x):
      x = np.array(x).flatten()
      d = np.log(x) - self.mu
      # Constant is dropped. Exact value doesn't really matter. Hopefully.
      return -0.5 * np.dot(d.T, np.dot(self.inv, d))

    def lnpdf_grad(self, x):
      x = np.array(x).flatten()
      d = np.log(x) - self.mu
      return -np.dot(self.inv, d)

    def rvs(self, n):
      return np.exp(
          np.random.randn(int(n), self.sigma.shape[0]) * self.sigma + self.mu)

  # Use heavy tailed priors, but start with small values.
  kernel = kern.Matern52(features.shape[1], variance=.04, ARD=True)
  kernel.unconstrain()
  loggaussian = LogGaussian(
      np.zeros(features.shape[1:]),
      sigma=np.diag(np.ones(features.shape[1:]) * 4.6))
  kernel.lengthscale.set_prior(loggaussian)
  kernel.lengthscale.constrain_bounded(1e-2, 1e2)
  kernel.variance.set_prior(priors.LogGaussian(-3.2, 4.6))
  kernel.variance.constrain_bounded(1e-3, 1e1)

  gpy_model = models.GPRegression(features, labels, kernel, noise_var=0.0039)
  gpy_model.likelihood.unconstrain()
  gpy_model.likelihood.variance.set_prior(priors.LogGaussian(-5.5, sigma=4.6))
  gpy_model.likelihood.variance.constrain_bounded(1e-10, 1.)

  gpy_model.optimize_restarts(20, robust=True, optimizer='lbfgsb')
  logging.info('After train: %s, %s', gpy_model, gpy_model.kern.lengthscale)
  return gpy_model


def _to_emukit_parameter(spec: converters.NumpyArraySpec) -> core.Parameter:
  if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
    return core.CategoricalParameter(
        spec.name,
        core.OneHotEncoding(list(range(spec.num_dimensions - spec.num_oovs))))
  elif spec.type == converters.NumpyArraySpecType.CONTINUOUS:
    return core.ContinuousParameter(spec.name, spec.bounds[0], spec.bounds[1])
  else:
    raise ValueError(f'Unknown type: {spec.type.name} in {spec}')


def _to_emukit_parameters(search_space: vz.SearchSpace) -> core.ParameterSpace:
  parameters = [_to_emukit_parameter(pc) for pc in search_space.parameters]
  return core.ParameterSpace(parameters)


class Version(enum.Enum):
  DEFAULT_EI = 'emukit_default_ei'
  MATERN52_UCB = 'emukit_matern52_ucb_ard'


class EmukitDesigner(vza.Designer):
  """Wraps emukit library as a Vizier designer."""

  def __init__(self,
               study_config: vz.StudyConfig,
               *,
               version: Version = Version.DEFAULT_EI,
               num_random_samples: int = 10,
               metadata_ns: str = 'emukit'):
    """Init.

    Args:
      study_config: Must be a flat study with a single metric.
      version: Determines the behavior. See Version.
      num_random_samples: This designer suggests random points until this many
        trials have been observed.
      metadata_ns: Metadata namespace that this designer writes to.

    Raises:
      ValueError:
    """
    if study_config.search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional search.')
    if len(study_config.metric_information) != 1:
      raise ValueError(f'{type(self)} works with exactly one metric.')
    self._study_config = study_config
    self._version = Version(version)

    # Emukit pipeline's model and acquisition optimizer use the same
    # representation. We need to remove the oov dimensions.
    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config, pad_oovs=False)
    self._trials = tuple()
    self._emukit_space = core.ParameterSpace(
        [_to_emukit_parameter(spec) for spec in self._converter.output_specs])

    self._metadata_ns = metadata_ns
    self._num_random_samples = num_random_samples

  def update(self, trials: vza.CompletedTrials) -> None:
    self._trials += tuple(trials.completed)

  def _suggest_random(self, count: int) -> Sequence[vz.TrialSuggestion]:
    sampler = RandomDesign(self._emukit_space)  # Collect random points
    samples = sampler.get_samples(count)
    return self._to_suggestions(samples, 'random')

  def _to_suggestions(self, arr: np.ndarray,
                      source: str) -> Sequence[vz.TrialSuggestion]:
    """Convert arrays to suggestions and record the source."""
    suggestions = []
    for params in self._converter.to_parameters(arr):
      suggestion = vz.TrialSuggestion(params)
      suggestion.metadata.ns(self._metadata_ns)['source'] = source
      suggestions.append(suggestion)
    return suggestions

  def _suggest_bayesopt(self,
                        count: Optional[int] = None
                       ) -> Sequence[vz.TrialSuggestion]:
    features, labels = self._converter.to_xy(self._trials)
    # emukit minimizes. Flip the signs.
    labels = -labels

    if self._version == Version.DEFAULT_EI:
      gpy_model = models.GPRegression(features, labels)
      emukit_model = model_wrappers.GPyModelWrapper(gpy_model)
      acquisition = acquisitions.ExpectedImprovement(model=emukit_model)

    elif self._version == Version.MATERN52_UCB:
      gpy_model = _create_constrained_gp(features, labels)
      emukit_model = model_wrappers.GPyModelWrapper(gpy_model, n_restarts=20)
      acquisition = acquisitions.NegativeLowerConfidenceBound(
          model=emukit_model, beta=np.float64(1.8))

    logging.info(
        'Emukit model: model=%s',
        # Values associated with length-1 keys are useless.
        {k: v for k, v in emukit_model.model.to_dict().items() if len(k) > 1})

    logging.info('Gpy model: model=%s', gpy_model)

    bayesopt_loop = loops.BayesianOptimizationLoop(
        model=emukit_model,
        space=self._emukit_space,
        acquisition=acquisition,
        batch_size=count or 1)
    x1 = bayesopt_loop.get_next_points([])

    return self._to_suggestions(x1, 'bayesopt')

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    if len(self._trials) < self._num_random_samples:
      return self._suggest_random(
          count or (self._num_random_samples - len(self._trials)))

    try:
      return self._suggest_bayesopt(count)
    except np.linalg.LinAlgError as e:
      logging.exception('Training failed: %s', e)
      return self._suggest_random(count or 1)
