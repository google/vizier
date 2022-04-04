"""NSGA-II algorithm: https://ieeexplore.ieee.org/document/996017."""

import dataclasses
import pickle
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from vizier import algorithms
from vizier import pyvizier as vz
from vizier.pyvizier import converters


def _pareto_rank(ys: np.ndarray) -> np.ndarray:
  """Pareto rank, which is the number of points dominating it.

  Args:
    ys: (number of population) x (number of metrics) array.

  Returns:
    (number of population) integer array.
  """
  n = ys.shape[0]
  dominated = np.asarray(
      [[np.all(ys[i] <= ys[j]) & np.any(ys[j] > ys[i])
        for i in range(n)]
       for j in range(n)])
  return np.sum(dominated, axis=0)


def _crowding_distance(ys: np.ndarray) -> np.ndarray:
  """Crowding distance.

  Args:
    ys: (number of population) x (number of metrics) array.

  Returns:
    (number of population) float32 array. Higher numbers mean less crowding
    and more desirable.
  """
  scores = np.zeros([ys.shape[0]], dtype=np.float32)
  for m in range(ys.shape[1]):
    # Sort by the m-th metric.
    yy = ys[:, m]  # Shape: (num_population,)
    sid = np.argsort(yy)

    # Boundary are assigned infinity.
    scores[sid[0]] += np.inf
    scores[sid[-1]] += np.inf

    # Compute the crowding distance.
    yrange = yy[sid[-1]] - yy[sid[0]] + np.finfo(np.float32).eps
    scores[sid[1:-1]] += (yy[sid[2:]] - yy[sid[:-2]]) / yrange
  return scores


def _constraint_violation(ys: np.ndarray) -> np.ndarray:
  """Counts the constraints violated.

  Args:
    ys: (number of population) x (number of metrics) array.

  Returns:
    (number of population) array of integers.
  """
  return np.sum(ys < 0, axis=1)


@dataclasses.dataclass
class Population:
  """Population.

  No validations are done.

  Attributes:
    trials: Trials
    xs: Maps a parameter name to a [len(trials), D] array where D depends on the
      conversion logic.
    ys: [len(trials), M] array. Optimization metrics to be MAXIMIZED.
    cs: [len(trials), M] array. Soft constraints. Goal is to get them greater
      than zero.
    ages:
  """
  trials: List[vz.Trial]
  xs: converters.DictOf2DArrays
  ys: np.ndarray
  cs: np.ndarray
  ages: np.ndarray

  def __len__(self):
    return len(self.trials)

  def __getitem__(self, index: Any) -> 'Population':
    return Population(
        list(np.asarray(self.trials)[index]),
        converters.DictOf2DArrays({k: v[index] for k, v in self.xs.items()}),
        self.ys[index], self.cs[index], self.ages[index])

  def __add__(self, other: 'Population') -> 'Population':

    def concat_dict(d1, d2):
      return {k: np.concatenate([d1[k], d2[k]], axis=0) for k in d1}

    def concat(a1, a2):
      return np.concatenate([a1, a2], axis=0)

    return Population(self.trials + other.trials,
                      converters.DictOf2DArrays(concat_dict(self.xs, other.xs)),
                      concat(self.ys, other.ys), concat(self.cs, other.cs),
                      concat(self.ages, other.ages))


def _select_by(ys: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
  """Returns a boolean index array for the top `target` elements of ys.

  This method is tough to parse. Please improve the API if you see a better
  design!

  Args:
    ys: Array of shape [M]. Entries are expected to have a small set of unique
      values.
    target: Count to return.

  Returns:
    A tuple of two bolean index arrays `top` and `border`.
     * `ys[top]` has length less than or equal to `target`. They are within
       top `target`.
     * `ys[top | border]` has length greater than or equal to `target`.
     * `ys[border]` have all-identical entries. Callers should break ties
       among them.
     * `top & border` is all False.
  """
  if ys.shape[0] <= target:
    return (np.ones(ys.shape[:1],
                    dtype=np.bool_), np.zeros(ys.shape[:1], dtype=np.bool_))
  unique, counts = np.unique(ys, return_counts=True)
  cutoffidx = np.argmax(np.cumsum(counts) > target)
  cutoffnumber = unique[cutoffidx]
  return ys < cutoffnumber, ys == cutoffnumber


def _linf_mutation(population: Population,
                   *,
                   norm: float = .1) -> converters.DictOf2DArrays:
  """Perturb by a uniform sample from l-inf ball.

  Sample uniformly from [0, 1] if the perturbation pushes a coordinate out
  of [0, 1] range. This is clearly not ideal if optimum is at the boundary.

  Args:
    population:
    norm: Norm of the l-inf ball to sample from.

  Returns:

  """
  arr = population.xs.asarray()
  arr += np.random.uniform(-norm, norm, arr.shape)
  arr = np.where((arr >= 0) & (arr <= 1.), arr,
                 np.random.uniform(.0, 1., arr.shape))
  return population.xs.dict_like(arr)


class NSGA2(algorithms.Designer):
  """NSGA2."""
  _metadata_key: str = 'xs'

  # TODO: Allow an injection of selection_fn, so that users can
  # enforce their own selection mechanisms such as removing members that have
  # been in the pool for very long.
  def __init__(
      self,
      search_space: vz.SearchSpace,
      metrics: Iterable[vz.MetricInformation],
      population_size: int = 50,
      *,
      mutation_fn: Callable[[Population],
                            converters.DictOf2DArrays] = _linf_mutation,
      first_survival_after: Optional[int] = None,
      ranking_fn: Callable[[np.ndarray], np.ndarray] = _pareto_rank,
      eviction_limit: Optional[int] = None,
      metadata_namespace: str = 'nsga2'):
    """Init.

    Args:
      search_space:
      metrics:
      population_size:
      mutation_fn: Mutation function. Takes population as input and returns a
        feature dictionary.
      first_survival_after: Apply the survival step after observing this many.
        Defaults to twice the population size.
      ranking_fn: Takes (number of population) x (number of metrics) array of
        floating numbers and returns (number of population) array of integers,
        representing the pareto rank aka number of points it is dominated by. If
        you are using a large population size, plug in your XLA-compiled
        function.
      eviction_limit: Evict a gene that has been alive for this many
        generations.
      metadata_namespace: Metadata namespace to use.

    Raises:
      ValueError:
    """
    if search_space.is_conditional:
      raise ValueError('This algorithm does not support conditional search.')
    self._population_size = population_size
    self._total_seen = 0

    metrics = vz.MetricsConfig(metrics)
    self.objective_metrics = metrics.of_type(vz.MetricType.OBJECTIVE)
    self.safe_metrics = metrics.of_type(vz.MetricType.SAFETY)
    self.metrics = self.objective_metrics + self.safe_metrics

    self._mutation_fn = mutation_fn
    self._ranking_fn = ranking_fn
    self._eviction_limit = eviction_limit
    self._metadata_namespace = metadata_namespace

    def create_input_converter(pc):
      return converters.DefaultModelInputConverter(
          pc, scale=True, max_discrete_indices=0, onehot_embed=True)

    def create_metric_converter(mc):
      return converters.DefaultModelOutputConverter(
          mc,
          flip_sign_for_minimization_metrics=True,
          shift_safe_metrics=True,
          raise_errors_for_missing_metrics=True)

    self.converter = converters.DefaultTrialConverter(
        [create_input_converter(pc) for pc in search_space.parameters],
        [create_metric_converter(mc) for mc in self.metrics])

    self._pool = self._trials_to_population([])
    self._first_survival_after = first_survival_after or (
        self._population_size * 2)

  def _trials_to_population(self, trials: List[vz.Trial]) -> Population:
    """Converts trials into population. Accepts an empty list."""
    ys = self.converter.to_labels_array(trials)

    loaded_arrays = []
    notconverted_trials = []
    for t in trials:
      metadata = t.metadata.ns(self._metadata_namespace)
      if self._metadata_key in metadata:
        loaded_arrays.append(
            pickle.loads(
                bytes(metadata.get(self._metadata_key, cls=str), 'latin-1')))
      else:
        notconverted_trials.append(t)

    xs = converters.DictOf2DArrays(
        self.converter.to_features(notconverted_trials))
    if loaded_arrays:
      xs += xs.dict_like(np.asarray(loaded_arrays))

    return Population(trials, xs, ys[:, :len(self.objective_metrics)],
                      ys[:,
                         len(self.objective_metrics):], np.zeros(ys.shape[:1]))

  def _survival(self, population: Population) -> Population:
    """Applies survival process.

    Sorted all points by 3-tuple
    1. Descending order of safety constraint violation score. Zero
      means no violations.
    2. Descending order of crowding distance.
    3. Ascending order of how many points it's dominated by.

    Args:
      population:

    Returns:
      Population of size self._population_size.
    """
    if len(population.trials) <= self._population_size:
      return population

    selected = self._trials_to_population([])
    # Sort by the safety constraint.
    if self.safe_metrics:
      top, border = _select_by(
          _constraint_violation(population.cs), target=self._population_size)
      selected += population[top]
      population = population[border]

    # Sort by the pareto rank.
    pareto_ranks = self._ranking_fn(population.ys)
    top, border = _select_by(
        pareto_ranks, target=self._population_size - len(selected))
    selected += population[top]
    population = population[border]

    # Sort by the distance. Include the points that are already selected for
    # the computation.
    # Flip the sign so it works with ascending sort.
    distance = -_crowding_distance((selected + population).ys)
    sids = np.argsort(distance)
    # Selected points have fewer constraint violations or better pareto rank.
    # Regardless of the distance, they remain selected. Rank the remainder only.
    sids = sids[sids >= len(selected)] - len(selected)
    selected += population[sids[:self._population_size - len(selected)]]
    selected.ages += 1
    return selected

  def update(self, delta: algorithms.CompletedTrials) -> None:
    completed_trials = delta.completed
    if not completed_trials:
      return
    completed_trials = list(completed_trials)
    self._total_seen += len(completed_trials)
    # Evict stale members.
    # TODO: Make eviction happen before offsprings are generated.
    if self._eviction_limit is not None:
      self._pool = self._pool[self._pool.ages < self._eviction_limit]
    candidates = self._pool + self._trials_to_population(completed_trials)
    if self._total_seen >= self._first_survival_after:
      self._pool = self._survival(candidates)

  def suggest(self, count: Optional[int] = None) -> List[vz.TrialSuggestion]:
    # TODO: Update mutation to accept count.
    count = self._population_size
    if self._total_seen < self._population_size * 2:
      num_features = self._pool.xs.asarray().shape[1]
      suggestion_array = np.random.random([self._population_size, num_features])
      suggestion_trials = [
          vz.TrialSuggestion(p) for p in self.converter.to_parameters(
              self._pool.xs.dict_like(suggestion_array))
      ]
    else:
      mutated = self._mutation_fn(self._pool)
      suggestion_array = mutated.asarray()
      suggestion_trials = [
          vz.TrialSuggestion(p) for p in self.converter.to_parameters(mutated)
      ]

    # Mutations may happen in the continuous embedding space. Save that
    # representation to accumulate noise over generations.
    for idx, t in enumerate(suggestion_trials):
      t.metadata.ns(
          self._metadata_namespace)[self._metadata_key] = pickle.dumps(
              suggestion_array[idx]).decode('latin-1')
    return suggestion_trials
