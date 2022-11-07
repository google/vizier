# Copyright 2022 Google LLC.
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

"""Tests for grid."""
from vizier import pythia
from vizier import pyvizier
from vizier._src.algorithms.designers import grid
from vizier._src.algorithms.policies import designer_policy
from absl.testing import absltest


class GridSearchPolicyTest(absltest.TestCase):
  # TODO: Add conditional test case.

  def setUp(self):
    """Setups up search space."""
    self.search_space = pyvizier.SearchSpace()
    self.search_space.root.add_float_param(
        'double', min_value=-1.0, max_value=1.0)
    self.search_space.root.add_categorical_param(
        name='categorical', feasible_values=['a', 'b', 'c'])
    self.search_space.root.add_discrete_param(
        name='discrete', feasible_values=[0.1, 0.3, 0.5])
    self.search_space.root.add_int_param(name='int', min_value=1, max_value=5)
    self.search_space_size = grid.GRID_RESOLUTION * 3 * 3 * 5

    self.designer = grid.GridSearchDesigner(self.search_space)
    super().setUp()

  def test_make_grid_values(self):
    grid_values = grid._make_grid_values(self.search_space)
    self.assertLen(grid_values['double'], grid.GRID_RESOLUTION)
    double_list = [p.value for p in grid_values['double']]
    for d in double_list:
      self.assertBetween(d, -1.0, 1.0)
    self.assertContainsSubset([-1.0, 1.0], double_list)
    self.assertLen(grid_values['categorical'], 3)
    self.assertLen(grid_values['discrete'], 3)
    self.assertLen(grid_values['int'], 5)

  def test_make_grid_search_parameters(self):
    """Tests the internal grid_search_parameters function."""
    parameter_dict = grid._make_grid_search_parameters([0],
                                                       self.search_space)[0]
    self.assertLen(parameter_dict, 4)

  def test_make_suggestions(self):
    """Tests designer suggestion generation."""
    suggestions = self.designer.suggest(self.search_space_size)
    self.assertLen(suggestions, self.search_space_size)
    for suggestion in suggestions:
      self.assertLen(suggestion.parameters, 4)

    # Make sure we covered entire search space.
    distinct_suggestions = set([
        tuple(suggestion.parameters.as_dict().values())
        for suggestion in suggestions
    ])
    self.assertLen(distinct_suggestions, self.search_space_size)

  def test_policy_wrapping(self):
    problem = pyvizier.ProblemStatement()
    problem.search_space = self.search_space
    policy_supporter = pythia.InRamPolicySupporter(problem)
    policy = designer_policy.PartiallySerializableDesignerPolicy(
        problem, policy_supporter, grid.GridSearchDesigner.from_problem)

    # Make sure we covered entire search space.
    all_suggestions = []
    for _ in range(self.search_space_size):
      request = pythia.SuggestRequest(
          study_descriptor=policy_supporter.study_descriptor(), count=1)
      decisions = policy.suggest(request)
      all_suggestions.extend(decisions.suggestions)

    distinct_suggestions = set([
        tuple(suggestion.parameters.as_dict().values())
        for suggestion in all_suggestions
    ])
    self.assertLen(distinct_suggestions, self.search_space_size)


if __name__ == '__main__':
  absltest.main()
