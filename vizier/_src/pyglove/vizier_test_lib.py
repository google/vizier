# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Libraries for testing different Tuners on the shared backend."""
# pylint:disable=protected-access
# pylint:disable=invalid-name
import inspect
import threading
from typing import Type

from absl import flags
from absl import logging
import pyglove as pg
from vizier._src.pyglove import backend

from absl.testing import absltest

FLAGS = flags.FLAGS


class RandomAlgorithm(pg.DNAGenerator):
  """Dummy algorithm for testing."""

  def setup(self, dna_spec: pg.DNASpec):
    super().setup(dna_spec)
    self._random = pg.geno.Random(seed=1)
    self._random.setup(dna_spec)

  def _propose(self) -> pg.DNA:
    return self._random.propose()


@pg.members([
    ('max_trial_id', pg.typing.Int()),
    ('max_step', pg.typing.Int()),
])
class SizeLimitingStoppingPolicy(pg.tuning.EarlyStoppingPolicy):
  """Policy that stops trials when either ID or step is larger."""

  def _on_bound(self):
    super()._on_bound()
    self.requested_trial_steps = []
    self.stopped_trial_steps = []

  def should_stop_early(self, trial: pg.tuning.Trial) -> bool:
    if trial.status == 'COMPLETED':
      return False

    measurement = trial.measurements[-1]
    self.requested_trial_steps.append((trial.id, measurement.step))
    should_stop = (
        trial.id > self.max_trial_id or measurement.step > self.max_step
    )
    if should_stop:
      self.stopped_trial_steps.append((trial.id, measurement.step))
    return should_stop


class VizierTest(absltest.TestCase):
  """Base class for Vizier-based tests."""

  def __init__(
      self,
      backend_class: Type[backend.VizierBackend], *args, **kwargs
  ):
    self._backend_class = backend_class
    super().__init__(*args, **kwargs)


class SampleTest(VizierTest):
  """Tests for PyGlove sampling with Vizier backend."""

  def setUp(self):
    super().setUp()
    self._backend_class.use_study_prefix(self.id())

  def __init__(
      self,
      backend_class: Type[backend.VizierBackend],
      *args,
      builtin_multiobjective_algorithm_to_test: str,
      **kwargs):
    self._builtin_multiobjective_algorithm_to_test = (
        builtin_multiobjective_algorithm_to_test)
    super().__init__(backend_class, *args, **kwargs)

  def testSample(self):
    """Test `pg.sample` with Vizier backend."""
    self._backend_class.use_study_prefix('distributed_sampling')

    rewards_a = []
    algorithm = RandomAlgorithm()
    for a, fa in pg.sample(
        pg.Dict(x=pg.oneof([1, 2, 3])),  # Outer search space.
        algorithm=algorithm,
        num_examples=2,
        name='a',
    ):
      rs = []
      for b, fb in pg.sample(
          pg.Dict(y=pg.oneof([3, 4, 5, 6, 7])),  # Inner space.
          algorithm=backend.BuiltinAlgorithm('DEFAULT'),
          num_examples=3,
          name='a%d_b' % fa.id,
      ):
        logging.info('Sampling a=%s, fa=%s, b=%s, fb=%s', a, fa, b, fb)
        r = a.x + b.y
        fb(r)
        rs.append(r)
      ra = sum(rs) / len(rs)
      fa(ra)
      rewards_a.append(ra)

    # Check alogrithm state are updated.
    self.assertEqual(algorithm.num_proposals, 2)

    # Check studies are created.
    logging.info('Check studies.')
    self._backend_class._get_study_resource_name('distributed_sampling.a')
    self._backend_class._get_study_resource_name('distributed_sampling.a1_b')
    self._backend_class._get_study_resource_name('distributed_sampling.a2_b')

    # Check tuning results from each study.
    logging.info('Check tuning results from each study.')
    trials_a = pg.tuning.poll_result('a').trials
    self.assertLen(trials_a, 2)
    self.assertEqual([t.final_measurement.reward for t in trials_a], rewards_a)
    self.assertLen(pg.tuning.poll_result('a1_b').trials, 3)
    self.assertLen(pg.tuning.poll_result('a2_b').trials, 3)

    logging.info('Check that constant space raises an error.')
    # Non-deterministic value.
    with self.assertRaisesRegex(ValueError, "'space' is a constant value"):
      next(
          pg.sample(
              pg.Dict(x=1),  # a fixed value.
              algorithm=pg.geno.Random(seed=1),
              name='c',
          )
      )

  @absltest.skip('Pythia cannot accept empty trials.')
  def testSampleWithSearchAlgorithmStopIteration(self):
    """Test `pg.sample` with algorithm-side stop iteration."""
    self._backend_class.use_study_prefix('algorithm-stop-iteration')
    for sd, f in pg.sample(
        pg.Dict(x=pg.oneof(range(3)), y=pg.oneof(range(4))), pg.geno.Sweeping()
    ):
      f(sd.x + sd.y)
    self.assertLen(pg.tuning.poll_result('').trials, 3 * 4)

    # Continue from previous one.
    # We create a new Sweeping instance to test recovery of controller state.
    for _, _ in pg.sample(
        pg.Dict(x=pg.oneof(range(3)), y=pg.oneof(range(4))), pg.geno.Sweeping()
    ):
      assert False, 'should never happen'

    self.assertLen(pg.tuning.poll_result('').trials, 3 * 4)

  def testSampleWithRaceCondition(self):
    """Test `pg.sample` with race condition among co-workers."""
    self._backend_class.use_study_prefix('coworkers-with-race-conditions')
    _, f = next(pg.sample(pg.oneof([1, 2, 3]), pg.geno.Random(seed=1)))

    f(1)
    with self.assertRaisesRegex(
        pg.tuning.RaceConditionError,
        '.*Measurements can only be added to.*',
    ):
      f.add_measurement(0.1)

    with f.ignore_race_condition():
      f.add_measurement(0.1)

  def testSampleWithControllerSideUpdateOfDNAMetadata(self):
    """Test `pg.sample` with controller-side DNA metadata update."""
    self._backend_class.use_study_prefix('sampling_with_evolution')

    def write_metadata(dna_list):
      return [dna.set_metadata('foo', 1) for dna in dna_list]

    algorithm = pg.evolution.Evolution(
        (
            pg.evolution.Identity()[-1]
            >> pg.evolution.mutators.Uniform()
            >> write_metadata
        ),
        population_init=(pg.geno.Random(), 5),
    )

    for i, (x, fx) in enumerate(
        pg.sample(
            pg.oneof(range(100)),
            algorithm=algorithm,
            num_examples=10,
        )
    ):
      if i >= 5:
        self.assertEqual(fx.dna.metadata.foo, 1)
      fx(x)

    # Check alogrithm state are updated.
    self.assertEqual(algorithm.num_proposals, 10)

    # The feedback operation on the last trial is not yet called, since
    # it will be triggered by the next call to `GetNewSuggestions`.
    self.assertEqual(algorithm.num_feedbacks, 9)

    # Check studies are created.
    self._backend_class._get_study_resource_name('sampling_with_evolution')

    # Check the DNA metadata for each trial.
    trials = pg.tuning.poll_result('').trials
    self.assertLen(trials, 10)
    for i, t in enumerate(trials[:-1]):
      expected_metadata = {
          'proposal_id': i + 1,
          'generation_id': max(i - 5 + 1, 0) + 1,
          'feedback_sequence_number': i + 1,
          'initial_population': i < 5,
          'reward': float(t.dna.value),
      }
      if i >= 5:
        expected_metadata['foo'] = 1
      self.assertEqual(t.dna.metadata, expected_metadata)

    self.assertEqual(
        trials[-1].dna.metadata,
        {
            'proposal_id': 10,
            'generation_id': 6,
            'initial_population': False,
            'foo': 1,
        },
    )

  def testSamplingWithControllerSideEvaluation(self):
    """Test sampling with controller evaluated reward."""
    self._backend_class.use_study_prefix(
        'sampling_with_controller_side_evaluation'
    )

    def controller_side_evaluate(dna_list):
      return [pg.evolution.set_fitness(dna, 1.0) for dna in dna_list]

    algorithm = pg.evolution.Evolution(
        (
            pg.evolution.Identity()[-1]
            >> pg.evolution.mutators.Uniform()
            >> controller_side_evaluate
        ),
        population_init=(pg.geno.Sweeping(), 5),
    )

    client_side_evaluated = []
    for x, fx in pg.sample(
        pg.oneof(range(100)), algorithm=algorithm, num_examples=10
    ):
      client_side_evaluated.append(x)
      fx(x)

    # Check the DNA metadata for each trial.
    trials = pg.tuning.poll_result('').trials
    self.assertLen(trials, 10)
    for i, t in enumerate(trials[:-1]):
      self.assertEqual(
          t.dna.metadata,
          {
              'proposal_id': i + 1,
              'generation_id': max(i - 5 + 1, 0) + 1,
              'feedback_sequence_number': i + 1,
              'initial_population': i < 5,
              'reward': float(t.dna.value) if i < 5 else 1.0,
          },
          msg=f'for i={i}, t={t}',
      )

    # Check alogrithm state are updated.
    self.assertEqual(algorithm.num_proposals, 10)

    # The feedback operation on the last trial is not yet called, since
    # it will be triggered by the next call to `GetNewSuggestions`.
    self.assertEqual(algorithm.num_feedbacks, 9)

    # Check studies are created.
    self._backend_class._get_study_resource_name(
        'sampling_with_controller_side_evaluation'
    )

    # Only the initial population is evaluated at client side.
    self.assertEqual(client_side_evaluated, list(range(5)))

  def testSamplingWithMultiObjectiveAlgorithm(self):
    """Test sampling with multi-objective alogirthm."""
    # Test sampling with Vizier built-in algorithm.
    self._backend_class.use_study_prefix('multi-objective-sampling-vizier')
    sample1 = pg.sample(
        pg.oneof([1, 2]),
        backend.BuiltinAlgorithm(
            self._builtin_multiobjective_algorithm_to_test),
        metrics_to_optimize=['accuracy', 'latency'],
    )
    _, f = next(sample1)
    f(metrics={'accuracy': 0.9, 'latency': 0.5})

    class DummyMultiObjectiveAlgorithm(pg.DNAGenerator):
      """Test sampling with custom multi-objective algorithm."""

      @property
      def multi_objective(self):
        return True

      def setup(self, dna_spec):
        super().setup(dna_spec)
        self.rewards = []

      def _propose(self):
        return pg.DNA(1)

      def _feedback(self, dna, reward):
        self.rewards.append(reward)

    self._backend_class.use_study_prefix('multi-objective-sampling-custom')
    algo = DummyMultiObjectiveAlgorithm()
    sample2 = pg.sample(
        pg.oneof([1, 2]),
        algo,
        metrics_to_optimize=['reward', 'accuracy', 'latency'],
    )

    _, f = next(sample2)
    f(reward=0.0, metrics={'accuracy': 0.9, 'latency': 0.5})

    # In Vizier, feedback is done while the next example is requested.
    # Therefore, we call next sample to trigger the feedback call to the
    # first example.
    _, _ = next(sample2)
    self.assertEqual(algo.rewards, [(0.0, 0.9, 0.5)])

  def testSampleWithCustomTermination(self):
    """Test `pg.sample` with custom termination."""
    self._backend_class.use_study_prefix(None)
    hyper_value = pg.Dict(x=pg.oneof([1, 2, 3]))
    for x, f in pg.sample(
        hyper_value,
        algorithm=pg.geno.Random(seed=1),
        name='custom_termination',
    ):
      # Always invoke the feedback function in order to advance
      # to the next trial.
      if f.id == 1:
        f.skip()
      else:
        f.set_metadata('x', 'foo')
        f.set_metadata('y', True, per_trial=False)
        f.add_link('filepath', '/file/old_path')
        f(
            x.x,
            metrics={'accuracy': 0.9},
            metadata={'z': 'bar'},
            related_links={'filepath': '/file/path_%d' % f.id},
        )
      if f.id == 2:
        f.end_loop()
        logging.info('Ending loop')
        break

    result = pg.tuning.poll_result('custom_termination')
    self.assertFalse(result.is_active)
    self.assertEqual(result.metadata, {'y': True})
    self.assertLen(result.trials, 2)
    self.assertTrue(result.trials[0].infeasible)
    self.assertFalse(result.trials[1].infeasible)
    self.assertEqual(result.trials[1].measurements[0].reward, 3.0)
    self.assertEqual(
        result.trials[1].measurements[0].metrics,
        {
            # NOTE(daiyip): Vizier adds the reward to the metric with an empty
            # metric name.
            'reward': 3.0,
            'accuracy': 0.9,
        },
    )
    self.assertEqual(result.trials[1].metadata, {'x': 'foo', 'z': 'bar'})
    self.assertEqual(
        result.trials[1].related_links, {'filepath': '/file/path_2'}
    )
    self.assertEqual(
        str(result),
        inspect.cleandoc(
            """
        {
          'study': '%s',
          'status': {
            'COMPLETED': '2/2'
          },
          'best_trial': {
            'id': 2,
            'reward': 3.0,
            'step': 0,
            'dna': 'DNA(2)'
          }
        }"""
            % (
                self._backend_class._get_study_resource_name(
                    'custom_termination'
                )
            )
        ),
    )

    t = result.best_trial
    t.description = None
    self.assertEqual(
        str(t),
        inspect.cleandoc(
            """
        VizierTrial(
          id = 2,
          description = None,
          dna = DNA(
            value = 2,
            children = [],
            metadata = {}
          ),
          status = 'COMPLETED',
          final_measurement = Measurement(
            step = 0,
            elapse_secs = %s,
            reward = 3.0,
            metrics = {
              accuracy = 0.9,
              reward = 3.0
            },
            checkpoint_path = None
          ),
          infeasible = False,
          measurements = [
            0 : Measurement(
              step = 0,
              elapse_secs = %s,
              reward = 3.0,
              metrics = {
                accuracy = 0.9,
                reward = 3.0
              },
              checkpoint_path = None
            )
          ],
          metadata = {
            x = 'foo',
            z = 'bar'
          },
          related_links = {
            filepath = '/file/path_2'
          },
          created_time = %d,
          completed_time = %d
        )"""
            % (
                t.final_measurement.elapse_secs,
                t.measurements[0].elapse_secs,
                t.created_time,
                t.completed_time,
            )
        ),
    )

  def testSampleWithEarlyStopping(self):
    """Test sample with early stopping."""
    # Stop trial early if either the trial id or trial-step is greater than 1.
    early_stopping_policy = SizeLimitingStoppingPolicy(
        max_trial_id=1, max_step=1
    )
    actually_stopped = []
    for x, f in pg.sample(
        pg.oneof([1, 2, 3]),
        algorithm=RandomAlgorithm(),
        early_stopping_policy=early_stopping_policy,
        num_examples=2,
    ):
      skipped = False
      for i in range(4):
        f.add_measurement(x, step=i)
        # Add a minimal sleep to ensure that the measurement is added to DB.
        if f.should_stop_early():
          f.skip()
          skipped = True
          actually_stopped.append((f.id, i))
          break
      if not skipped:
        f.done()

    self.assertEqual(
        early_stopping_policy.requested_trial_steps,
        [(1, 0), (1, 1), (1, 2), (2, 0)],
    )
    self.assertEqual(
        early_stopping_policy.stopped_trial_steps, actually_stopped
    )

    self.assertEqual(
        early_stopping_policy.stopped_trial_steps, [(1, 2), (2, 0)]
    )

  def testSampleWithMultipleWorkers(self):
    """Test `pg.sample` with multiple workers."""
    self._backend_class.use_study_prefix(None)

    hyper_value = pg.Dict(x=pg.oneof([1, 2, 3]))

    # Create a new study for sample using Random.
    sample1 = pg.sample(
        hyper_value,
        algorithm=pg.geno.Random(seed=1),
        name='distributed_sampling2',
        group=0,
    )

    # `sample2` works on the same sampling queue with `sample1`
    # but with different trials by having the same `name` with
    # a different `group`.
    sample2 = pg.sample(
        hyper_value,
        algorithm=pg.geno.Random(seed=1),
        name='distributed_sampling2',
        group=1,
    )

    # `sample3` works on the same sampling queue and the same trials
    # with `sample1` by having the same `name` and `group`.
    sample3 = pg.sample(
        hyper_value,
        algorithm=pg.geno.Random(seed=1),
        name='distributed_sampling2',
        group=0,
    )

    # Make sure sampling with different worker IDs get different trial IDs.
    _, f1 = next(sample1)
    self.assertEqual(f1.id, 1)
    f1.set_metadata('x', 1)
    f1.set_metadata('y', RandomAlgorithm(), per_trial=False)

    # X is not serializable via `pg.to_json_str()`.
    class X:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert local class .* to JSON'
    ):
      f1.set_metadata('z', X)

    _, f2 = next(sample2)
    self.assertEqual(f2.id, 2)
    self.assertIsNone(f2.get_metadata('x'))
    self.assertEqual(f2.get_metadata('y', per_trial=False), RandomAlgorithm())

    _, f3 = next(sample3)
    self.assertEqual(f3.id, 1)
    self.assertEqual(f3.get_metadata('x'), 1)
    self.assertEqual(f3.get_metadata('y', per_trial=False), RandomAlgorithm())
    # Update the value of 'x'.
    f3.set_metadata('x', 2)
    f3.set_metadata('z', 'foo')
    f3.add_measurement(0.1, step=1)

    # Make sure sampling within the same worker get the same trial IDs before
    # feedback.
    _, f1b = next(sample1)
    self.assertEqual(f1b.id, 1)

    # Make sure `f1.trial` reflect the measurement added by f3.
    trial = f1.get_trial()
    self.assertLen(trial.measurements, 1)
    self.assertEqual(
        trial.measurements[0].reward,
        0.1,
        msg=f'Measurement was: {trial.measurements}',
    )

    logging.info('Trial metadata: %s', trial.metadata)
    self.assertEqual(
        trial.metadata.z, 'foo', msg=f'Metadata was: {trial.metadata}'
    )
    # Mark trial 1 as done.
    f1(0.5)
    # Check the value of metadata 'x' is updated (by f3).
    self.assertEqual(f1.get_metadata('x'), 2)
    self.assertEqual(f3.get_trial().status, 'COMPLETED')

    # NOTE(daiyip): trial 1 is now COMPLETED after `f1(0)`, therefore, calling
    # `f1b(reward)` will attempt to add a new measurement to the completed
    # trial, triggering a Vizier error.
    with self.assertRaisesRegex(
        pg.tuning.RaceConditionError,
        '.*Measurements can only be added to.*',
    ):
      f1b(1)

    # Make sure sampling within the same worker get different trial IDs after
    # previous trial is done.
    _, f1c = next(sample1)
    self.assertEqual(f1c.id, 3)
    _, f3b = next(sample3)
    self.assertEqual(f3b.id, 3)

    # After calling `end_loop`, all samplings (even with pending ones)
    # shall stop.
    f2.end_loop()
    with self.assertRaises(StopIteration):
      next(sample1)

    with self.assertRaises(StopIteration):
      next(sample2)

    with self.assertRaises(StopIteration):
      next(sample3)

  def testSampleWithDifferentSearchSpace(self):
    """Test client-side search space mismatch with server-side search space."""
    self._backend_class.use_study_prefix('different_search_space_vizier')

    ssd1 = pg.Dict(x=pg.oneof([1, 2, 3]))
    sample1 = pg.sample(ssd1, pg.geno.Random())
    x, f = next(sample1)
    f(x.x)

    ssd2 = pg.Dict(x=pg.oneof([1, 2]))
    sample2 = pg.sample(ssd2, pg.geno.Random())
    with self.assertRaisesRegex(ValueError, '.*different.*search space.*'):
      next(sample2)

  def testSampleWithCustomHyper(self):
    """Test sample with custom hyper."""
    self._backend_class.use_study_prefix('custom_hyper_vizier')

    class VariableString(pg.hyper.CustomHyper):
      """A custom decision type that represents a variable-length string."""

      def custom_decode(self, dna):
        return dna.value

    @pg.geno.dna_generator
    def init_population(dna_spec):
      yield pg.DNA('abc', spec=dna_spec)

    class StringRepeater(pg.evolution.Mutator):

      def mutate(self, dna):  # type: ignore
        return pg.DNA(dna.value + dna.value, spec=dna.spec)

    algo = pg.evolution.Evolution(
        StringRepeater(),
        population_init=init_population.partial(),
        population_update=pg.evolution.selectors.Last(1),
    )

    sample = pg.sample(VariableString(), algo)
    x, f = next(sample)
    self.assertEqual(x, 'abc')
    f(len(x))

    x, f = next(sample)
    self.assertEqual(x, 'abcabc')
    f(len(x))

    x, f = next(sample)
    self.assertEqual(x, 'abcabcabcabc')
    f(len(x))

  def testAllProposedTrialAreDeliveredToWorkers(self):
    """Test that all proposed trials are delivered to the workers."""
    self._backend_class.use_study_prefix('proposal_delivery')
    searchable_list = pg.List(
        [
            pg.one_of([1, 2, 3]),
            pg.one_of([-1, 0, 1]),
            pg.one_of([1, 2, 0]),
        ]
        * 5
    )
    algorithm = pg.evolution.regularized_evolution()

    def worker_fun():
      for l, f in pg.sample(searchable_list, algorithm, num_examples=100):
        reward = float(sum(l))
        f(reward)

    workers = [threading.Thread(target=worker_fun) for _ in range(10)]
    for w in workers:
      w.start()
    for w in workers:
      w.join()

    result = pg.tuning.poll_result('')
    for t in result.trials:
      self.assertEqual(t.id, t.dna.metadata.proposal_id)
