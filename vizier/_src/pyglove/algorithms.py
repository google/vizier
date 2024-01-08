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

"""Classes for specifying Vizier algorithms."""
import pyglove as pg


class PseudoAlgorithm(pg.DNAGenerator):
  """Base class for algorithms that delegate actual logics to others."""

  def _setup(self) -> None:
    raise RuntimeError(f'{self!r} should be used with \'vizier\' backend.')

  def _propose(self) -> pg.DNA:
    self._should_never_trigger('propose')
    return pg.DNA(0)

  def _feedback(self, unused_dna: pg.DNA, unused_reward: float) -> None:
    self._should_never_trigger('feedback')

  def _should_never_trigger(self, method_name):
    raise RuntimeError(
        f'`{method_name}` of pseudo algorithm {self!r} should never be called.')


@pg.members(
    [(
        'name',
        pg.typing.Enum(
            'DEFAULT',
            [
                'DEFAULT',
                # Algorithms shared across Vizier platforms.
                'GRID_SEARCH',
                'RANDOM_SEARCH',
                # Google-Vizier only algorithms.
                # TODO: copybara away these algorithms for OSS.
                'CMAES',
                'GP_BANDIT',
                'LINEAR_COMBINATION_SEARCH',
                'HYPER_LCS',
                # OSS-Vizier only algorithms:
                'BOCS',
                'CMA_ES',
                'EMUKIT_GP_EI',
                'HARMONICA',
                'NSGA2',
                'QUASI_RANDOM_SEARCH',
            ]),
        'Name of Vizier predefined algorithm.')],
    metadata={'init_arg_list': ['name']})  # pylint: disable=bad-continuation
class BuiltinAlgorithm(PseudoAlgorithm):
  """Vizier built-in algorithm."""

  @property
  def name(self) -> str:
    return self.sym_init_args.name

  @property
  def multi_objective(self) -> bool:
    return self.name in ('DEFAULT', 'GP_BANDIT', 'LINEAR_COMBINATION_SEARCH',
                         'RANDOM_SEARCH')


# TODO: implement the details according to Vizier early stopping policy
# later. Move to stopping.py
class BuiltinEarlyStoppingPolicy(pg.tuning.EarlyStoppingPolicy):
  """Vizier built-in early stopping policy."""
