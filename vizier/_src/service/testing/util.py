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

"""Below contain utilities for writing tests to reduce code duplication."""
from typing import List, Sequence

from vizier._src.service import resources
from vizier._src.service import study_pb2
from vizier._src.service import vizier_oss_pb2

from google.longrunning import operations_pb2


def generate_study(owner_id: str = 'my_username',
                   study_id: str = '1234',
                   display_name: str = 'cifar10',
                   **study_kwargs) -> study_pb2.Study:
  study_name = resources.StudyResource(owner_id, study_id).name
  return study_pb2.Study(
      name=study_name, display_name=display_name, **study_kwargs)


def generate_trials(trial_id_list: Sequence[int],
                    owner_id: str = 'my_username',
                    study_id: str = '1234',
                    **trial_kwargs) -> List[study_pb2.Trial]:
  """Generates arbitrary trials."""
  trials = []
  for trial_id in trial_id_list:
    trial = study_pb2.Trial(
        name=resources.TrialResource(owner_id, study_id, trial_id).name,
        id=str(trial_id),
        **trial_kwargs)
    trials.append(trial)
  return trials


def generate_all_states_trials(start_trial_index: int,
                               owner_id: str = 'my_username',
                               study_id: str = '1234',
                               **trial_kwargs) -> List[study_pb2.Trial]:
  """Generates a trial for each possible trial state."""
  trials = []
  for i, state in enumerate(study_pb2.Trial.State.keys()):
    trial_id = start_trial_index + i
    trial = study_pb2.Trial(
        name=resources.TrialResource(owner_id, study_id, trial_id).name,
        id=str(trial_id),
        state=state,
        **trial_kwargs)
    trials.append(trial)
  return trials


def generate_suggestion_operations(
    operation_numbers: Sequence[int],
    owner_id: str = 'my_username',
    study_id: str = 'cifar10',
    client_id: str = 'client0',
    **operation_kwargs) -> List[operations_pb2.Operation]:
  """Generates arbitrary suggestion operations."""
  operations = []
  for operation_number in operation_numbers:
    operation = operations_pb2.Operation(
        name=resources.SuggestionOperationResource(owner_id, study_id,
                                                   client_id,
                                                   operation_number).name,
        **operation_kwargs)
    operations.append(operation)
  return operations


def generate_early_stopping_operations(
    trial_id_list: Sequence[int],
    owner_id: str = 'my_username',
    study_id: str = '1234',
    **operation_kwargs) -> List[vizier_oss_pb2.EarlyStoppingOperation]:
  """Generates arbitrary early stopping operations."""
  operations = []
  for trial_id in trial_id_list:
    operation = vizier_oss_pb2.EarlyStoppingOperation(
        name=resources.EarlyStoppingOperationResource(owner_id, study_id,
                                                      trial_id).name,
        **operation_kwargs)
    operations.append(operation)
  return operations


def generate_all_four_parameter_specs(**study_spec_kwargs
                                     ) -> study_pb2.StudySpec:
  """All possible primitive parameter specs for testing."""
  double_value_spec = study_pb2.StudySpec.ParameterSpec.DoubleValueSpec(
      min_value=-1.0, max_value=1.0)
  double_parameter_spec = study_pb2.StudySpec.ParameterSpec(
      parameter_id='learning_rate', double_value_spec=double_value_spec)

  integer_value_spec = study_pb2.StudySpec.ParameterSpec.IntegerValueSpec(
      min_value=1, max_value=10)
  integer_parameter_spec = study_pb2.StudySpec.ParameterSpec(
      parameter_id='num_layers', integer_value_spec=integer_value_spec)

  categorical_value_spec = (
      study_pb2.StudySpec.ParameterSpec.CategoricalValueSpec(
          values=['relu', 'sigmoid']
      )
  )
  categorical_parameter_spec = study_pb2.StudySpec.ParameterSpec(
      parameter_id='nonlinearity',
      categorical_value_spec=categorical_value_spec)

  discrete_value_spec = study_pb2.StudySpec.ParameterSpec.DiscreteValueSpec(
      values=[1.0, 1.5, 3.0, 4.5])
  discrete_parameter_spec = study_pb2.StudySpec.ParameterSpec(
      parameter_id='discrete_unnamed', discrete_value_spec=discrete_value_spec)

  return study_pb2.StudySpec(
      parameters=[
          double_parameter_spec, integer_parameter_spec,
          categorical_parameter_spec, discrete_parameter_spec
      ],
      **study_spec_kwargs)
