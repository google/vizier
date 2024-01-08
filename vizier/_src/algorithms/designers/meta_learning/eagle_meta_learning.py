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

"""Eagle meta learner search space.
"""

from vizier import pyvizier as vz


def meta_eagle_search_space() -> vz.SearchSpace:
  """Returns the meta eagle search space."""
  search_space = vz.SearchSpace()
  # Perturbation
  search_space.root.add_float_param(
      name='perturbation',
      min_value=1e-4,
      max_value=1e2,
      default_value=1e-1,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='perturbation_lower_bound',
      min_value=1e-5,
      max_value=1e-1,
      default_value=1e-3,
      scale_type=vz.ScaleType.LOG,
  )
  # Gravity
  search_space.root.add_float_param(
      name='gravity',
      min_value=1e-2,
      max_value=1e2,
      default_value=1.0,
      scale_type=vz.ScaleType.LOG,
  )
  # Visibility
  search_space.root.add_float_param(
      name='visibility',
      min_value=3 * 1e-2,
      max_value=3 * 1e2,
      default_value=3.0,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='categorical_visibility',
      min_value=2.0 * 1e-3,
      max_value=2.0 * 1e1,
      default_value=2.0 * 1e-1,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='discrete_visibility',
      min_value=1e-2,
      max_value=1e2,
      default_value=1.0,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='categorical_perturbation_factor',
      min_value=2.5 * 1e-1,
      max_value=2.5 * 1e3,
      default_value=2.5 * 1e1,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='discrete_perturbation_factor',
      min_value=1e-1,
      max_value=1e3,
      default_value=1e1,
      scale_type=vz.ScaleType.LOG,
  )
  # Pool size.
  search_space.root.add_float_param(
      name='pool_size_factor',
      min_value=1.0,
      max_value=2.0,
      default_value=1.2,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='negative_gravity',
      min_value=2.0 * 1e-4,
      max_value=2.0,
      default_value=2 * 1e-2,
      scale_type=vz.ScaleType.LOG,
  )
  search_space.root.add_float_param(
      name='pure_categorical_perturbation',
      min_value=1e-3,
      max_value=1e1,
      default_value=1e-1,
      scale_type=vz.ScaleType.LOG,
  )

  return search_space
