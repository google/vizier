# Copyright 2023 Google LLC.
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

import json

from vizier import pyvizier as vz
from vizier.pyvizier.converters import string_converters

from absl.testing import absltest


class StringConvertersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.default_prompts = {'prompt1': 'def1', 'prompt2': 'def2'}
    self.config = string_converters.PromptTuningConfig(
        default_prompts=self.default_prompts
    )

  def test_augment_problem(self):
    problem = vz.ProblemStatement()
    tuning_problem = self.config.augment_problem(problem)
    for k, v in self.default_prompts.items():
      pconfig = tuning_problem.search_space.get(k)
      self.assertCountEqual(pconfig.feasible_values, [v])
      self.assertEqual(pconfig.default_value, v)

  def test_prompt_trials(self):
    trial = vz.Trial(parameters={'int': 3, 'float': 1.2, 'cat': 'test'})
    trial.metadata.ns(string_converters.PROMPT_TUNING_NS)['values'] = (
        json.dumps({'prompt1': 'test1', 'prompt2': 'test2'})
    )
    results = self.config.to_prompt_trials([trial])
    self.assertLen(results, 1)

    self.assertStartsWith(results[0].parameters['prompt1'].value, 'test1')
    self.assertEqual(results[0].parameters['prompt2'].value, 'test2')

  def test_valid_suggestions(self):
    problem = vz.ProblemStatement()
    tuning_problem = self.config.augment_problem(problem)
    suggestion = vz.TrialSuggestion(
        parameters={'prompt1': 'test1', 'prompt2': 'test2'}
    )
    valid_suggestions = self.config.to_valid_suggestions([suggestion])
    self.assertLen(valid_suggestions, 1)
    valid_suggestion = valid_suggestions[0]
    self.assertTrue(
        tuning_problem.search_space.contains(valid_suggestion.parameters)
    )

    # Test the reverse conversion retrieves original parameters.
    trial = valid_suggestion.to_trial().complete(vz.Measurement())
    prompt_trial = self.config.to_prompt_trials([trial])[0]
    self.assertCountEqual(prompt_trial.parameters, suggestion.parameters)


if __name__ == '__main__':
  absltest.main()
