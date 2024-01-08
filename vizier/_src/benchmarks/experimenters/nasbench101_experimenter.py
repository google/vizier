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

"""NAS-Bench-101 Experimenter.

NOTE: Since our installation uses tensorflow>=2.0.0, you will need to change the
following in the nasbench package code (which assumes tensorflow-1 API):

```
tf.train.SessionRunHook -> tf.estimator.SessionRunHook
tf.train.CheckpointSaverListener -> tf.estimator.CheckpointSaverListener
tf.train.NanLossDuringTrainingError -> tf.estimator.NanLossDuringTrainingError
tf.python_io.tf_record_iterator -> tf.compat.v1.io.tf_record_iterator
```

See https://github.com/google-research/nasbench/issues/27.
"""
from typing import Sequence

from absl import logging
# from nasbench import api
import numpy as np

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter

logging.warning(
    'NASBENCH-101 assumes tensorflow<2.0. Please see above in this file for what to replace in nasbench code to make the benchmark work.'
)


class NASBench101Experimenter(experimenter.Experimenter):
  """This suggests model specs in the form of a matrix (for DAG topology) and ops (for convolutions).

  Search space is a union of binary strings and categoricals.
  """

  def __init__(self, nasbench):
    self._nasbench = nasbench
    self._num_vertices = 7
    self._allowed_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
    self._op_spots = self._num_vertices - 2
    self._input_op = 'input'
    self._output_op = 'output'

    self._metric_names = [
        'trainable_parameters', 'training_time', 'train_accuracy',
        'validation_accuracy', 'test_accuracy'
    ]

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    """Evaluates and completes the Trial using NASBench-101 querying."""

    for trial in suggestions:
      spec = self._trial_to_model_spec(trial)
      if self._nasbench.is_valid(spec):
        results = self._nasbench.query(spec)
        trial.complete(
            pyvizier.Measurement(
                metrics={k: results[k] for k in self._metric_names}))
      else:
        trial.complete(
            pyvizier.Measurement(), infeasibility_reason='Not in search space.')

  def _trial_to_model_spec(self, trial: pyvizier.Trial):
    matrix = np.zeros((self._num_vertices, self._num_vertices), dtype=int)
    for y in range(self._num_vertices):
      for x in range(self._num_vertices):
        if y > x:
          matrix[x][y] = int(
              trial.parameters['{}_{}'.format(x, y)].value == 'True')

    base_ops = []
    for i in range(self._op_spots):
      base_ops.append(trial.parameters['ops_{}'.format(i)].value)
    ops = [self._input_op] + base_ops + [self._output_op]

    return api.ModelSpec(matrix=matrix, ops=ops)

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root

    for y in range(self._num_vertices):
      for x in range(self._num_vertices):
        if y > x:
          root.add_bool_param(name='{}_{}'.format(x, y))

    for i in range(self._op_spots):
      root.add_categorical_param(
          name='ops_{}'.format(i), feasible_values=self._allowed_ops)

    # 'test_accuracy' also can be used for objective value.
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='validation_accuracy',
            goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))
    return problem_statement
