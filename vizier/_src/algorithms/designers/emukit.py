"""EmukitDesigner wraps emukit into Vizier Designer."""

from typing import Optional, Sequence

from emukit import core
from emukit import model_wrappers
from emukit.bayesian_optimization import acquisitions
from emukit.bayesian_optimization import loops
from emukit.core import initial_designs
from emukit.core import loop
from GPy import models
import numpy as np
from vizier import algorithms as vza
from vizier.pyvizier import converters
from vizier.pyvizier import pythia as vz

RandomDesign = initial_designs.RandomDesign


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


class EmukitDesigner(vza.Designer):
  """Wraps emukit library as a Vizier designer."""

  def __init__(self,
               study_config: vz.StudyConfig,
               *,
               num_random_samples: int = 10,
               metadata_ns: str = 'emukit'):
    """Init.

    Args:
      study_config: Must be a flat study with a single metric.
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

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    features, labels = self._converter.to_xy(self._trials)
    # emukit minimizes. Flip the signs.
    labels = -labels

    if len(self._trials) < self._num_random_samples:
      return self._suggest_random(
          count or (self._num_random_samples - len(self._trials)))

    gpy_model = models.GPRegression(features, labels)
    emukit_model = model_wrappers.GPyModelWrapper(gpy_model)
    expected_improvement = acquisitions.ExpectedImprovement(model=emukit_model)

    bayesopt_loop = loops.BayesianOptimizationLoop(
        model=emukit_model,
        space=self._emukit_space,
        acquisition=expected_improvement,
        batch_size=count or 1)
    x1 = bayesopt_loop.get_next_points(
        [loop.UserFunctionResult(x, y) for x, y in zip(features, labels)])

    return self._to_suggestions(x1, 'bayesopt')
