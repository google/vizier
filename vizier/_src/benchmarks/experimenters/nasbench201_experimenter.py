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

"""NAS-Bench-201 Experimenter.

Note that 'NATS' is an alias of 201.

Currently we only consider the topology search space (TSS) as it is much more
frequently used, as opposed to the size search space (SSS) which checks
performance by varying depths.
"""

from typing import Sequence
import nats_bench

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


def _model_tss_spc(ops: Sequence[str], num_nodes: int) -> str:
  """Converts ops and nodes to a string format recognized by NASBENCH-201."""
  nodes, k = [], 0
  for i in range(1, num_nodes):
    xstrs = []
    for j in range(i):
      xstrs.append('{:}~{:}'.format(ops[k], j))
      k += 1
    nodes.append('|' + '|'.join(xstrs) + '|')
  return '+'.join(nodes)


class NASBench201Experimenter(experimenter.Experimenter):
  """NASBENCH-201."""

  def __init__(self,
               nasbench: nats_bench.NATStopology,
               datastr_str: str = 'cifar10',
               validation_set_reporting_epoch: int = 12):
    self._nasbench = nasbench
    self._dataset_str = datastr_str
    tss_raw_config = nats_bench.search_space_info('nats-bench', 'tss')

    self._allowed_ops = tss_raw_config['op_names']
    self._num_nodes = tss_raw_config['num_nodes']
    self._op_spots = 6

  @property
  def nasbench(self) -> nats_bench.NATStopology:
    return self._nasbench

  def _trial_to_model_spec(self, trial: pyvizier.Trial):
    ops = [
        trial.parameters['op_{}'.format(i)].value for i in range(self._op_spots)
    ]
    return _model_tss_spc(ops, self._num_nodes)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for trial in suggestions:
      spec = self._trial_to_model_spec(trial)
      valid_acc, latency, time_cost, total_time = self._nasbench.simulate_train_eval(
          arch=spec, dataset=self._dataset_str)
      trial.complete(
          pyvizier.Measurement(
              metrics={
                  'valid_acc': valid_acc,
                  'latency': latency,
                  'time_cost': time_cost,
                  'total_time': total_time
              }))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root
    for i in range(self._op_spots):
      root.add_categorical_param(
          name='op_{}'.format(i), feasible_values=self._allowed_ops)
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='valid_acc', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))
    return problem_statement
