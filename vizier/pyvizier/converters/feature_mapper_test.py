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

"""Tests for feature_mapper."""

from jax import config
import numpy as np
from vizier import pyvizier as vz
from vizier.pyvizier.converters import core
from vizier.pyvizier.converters import feature_mapper as fm

from absl.testing import absltest
from absl.testing import parameterized


class ContinuousCategoricalConverterTest(parameterized.TestCase):

  @parameterized.product(pad_oovs=[False, True])
  def test_categorical_only(self, pad_oovs):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_categorical_param('C1', ['a', 'b', 'c'])
    problem.search_space.root.add_categorical_param('C2', ['w', 'x', 'y', 'z'])
    problem.search_space.root.add_categorical_param('C3', ['0', '1'])
    problem.search_space.root.add_categorical_param('C4', ['*'])
    trial1 = vz.Trial(parameters={'C1': 'b', 'C2': 'z', 'C3': '0', 'C4': '*'})
    trial2 = vz.Trial(parameters={'C1': 'a', 'C2': 'z', 'C3': '0', 'C4': '*'})
    trial3 = vz.Trial(parameters={'C1': 'c', 'C2': 'x', 'C3': '1', 'C4': '*'})
    converter = core.TrialToArrayConverter.from_study_config(
        problem,
        pad_oovs=pad_oovs,
        max_discrete_indices=0,
    )
    features = converter.to_features([trial1, trial2, trial3])
    feature_mapper = fm.ContinuousCategoricalFeatureMapper(converter)
    res = feature_mapper.map(features)
    np.testing.assert_array_equal(
        res.categorical, np.array([[1, 3, 0, 0], [0, 3, 0, 0], [2, 1, 1, 0]])
    )
    np.testing.assert_array_equal(np.zeros((3, 0)), res.continuous)
    # Test un-mapping
    unmapped_features = feature_mapper.unmap(res)
    np.testing.assert_array_equal(unmapped_features, features)

  @parameterized.product(pad_oovs=[False, True], max_discrete_indices=[0, 10])
  def test_discrete_only(self, pad_oovs, max_discrete_indices):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_discrete_param('d1', [1, 10, 20])
    trial1 = vz.Trial(parameters={'d1': 10})
    trial2 = vz.Trial(parameters={'d1': 1})
    trial3 = vz.Trial(parameters={'d1': 20})
    converter = core.TrialToArrayConverter.from_study_config(
        problem, pad_oovs=pad_oovs, max_discrete_indices=max_discrete_indices
    )
    features = converter.to_features([trial1, trial2, trial3])
    feature_mapper = fm.ContinuousCategoricalFeatureMapper(converter)
    res = fm.ContinuousCategoricalFeatureMapper(converter).map(features)
    if max_discrete_indices == 10:
      # DISCRETE params are one-hot encoded. Should be mapped to categorical.
      np.testing.assert_array_equal(res.categorical, np.array([[1], [0], [2]]))
      np.testing.assert_array_equal(np.zeros((3, 0)), res.continuous)
    elif max_discrete_indices == 0:
      # DISCRETE params are continufied. Should be mapped to continuous.
      np.testing.assert_almost_equal(
          res.continuous, np.array([[(10 - 1) / (20 - 1)], [0.0], [1.0]])
      )
      np.testing.assert_array_equal(np.zeros((3, 0)), res.categorical)
    # Test un-mapping
    unmapped_features = feature_mapper.unmap(res)
    np.testing.assert_allclose(unmapped_features, features)

  def test_integer_only(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_int_param('i1', 0, 10)
    problem.search_space.root.add_int_param('i2', 0, 10)
    converter = core.TrialToArrayConverter.from_study_config(
        problem, pad_oovs=True, max_discrete_indices=0
    )
    trial1 = vz.Trial(parameters={'i1': 10, 'i2': 5})
    trial2 = vz.Trial(parameters={'i1': 1, 'i2': 3})
    trial3 = vz.Trial(parameters={'i1': 2, 'i2': 4})
    features = converter.to_features([trial1, trial2, trial3])
    feature_mapper = fm.ContinuousCategoricalFeatureMapper(converter)
    res = fm.ContinuousCategoricalFeatureMapper(converter).map(features)
    np.testing.assert_array_equal(
        res.continuous, np.array([[1.0, 0.5], [0.1, 0.3], [0.2, 0.4]])
    )
    np.testing.assert_array_equal(np.zeros((3, 0)), res.categorical)
    # Test un-mapping
    unmapped_features = feature_mapper.unmap(res)
    np.testing.assert_allclose(unmapped_features, features)

  def test_float_only(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = core.TrialToArrayConverter.from_study_config(
        problem, pad_oovs=True, max_discrete_indices=0
    )
    trial1 = vz.Trial(parameters={'f1': 1.0, 'f2': 0.5})
    trial2 = vz.Trial(parameters={'f1': 0.1, 'f2': 0.3})
    trial3 = vz.Trial(parameters={'f1': 0.2, 'f2': 0.4})
    features = converter.to_features([trial1, trial2, trial3])
    feature_mapper = fm.ContinuousCategoricalFeatureMapper(converter)
    res = fm.ContinuousCategoricalFeatureMapper(converter).map(features)
    np.testing.assert_allclose(
        res.continuous, np.array([[1.0, 0.5], [0.1, 0.3], [0.2, 0.4]])
    )
    np.testing.assert_array_equal(np.zeros((3, 0)), res.categorical)
    # Test un-mapping
    unmapped_features = feature_mapper.unmap(res)
    np.testing.assert_allclose(unmapped_features, features, atol=1e-5)

  @parameterized.product(pad_oovs=[False, True], max_discrete_indices=[0, 10])
  def test_mixed_space(self, pad_oovs, max_discrete_indices):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    problem.search_space.root.add_discrete_param('d1', [1, 10, 20])
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    trial1 = vz.Trial(parameters={'f1': 1.0, 'f2': 0.5, 'd1': 10, 'c1': 'b'})
    trial2 = vz.Trial(parameters={'f1': 0.1, 'f2': 0.3, 'd1': 10, 'c1': 'a'})
    trial3 = vz.Trial(parameters={'f1': 0.2, 'f2': 0.4, 'd1': 20, 'c1': 'c'})
    converter = core.TrialToArrayConverter.from_study_config(
        problem, pad_oovs=pad_oovs, max_discrete_indices=max_discrete_indices
    )
    features = converter.to_features([trial1, trial2, trial3])
    feature_mapper = fm.ContinuousCategoricalFeatureMapper(converter)
    res = fm.ContinuousCategoricalFeatureMapper(converter).map(features)
    if max_discrete_indices == 10:
      np.testing.assert_array_equal(
          res.continuous, np.array([[1.0, 0.5], [0.1, 0.3], [0.2, 0.4]])
      )
      np.testing.assert_array_equal(
          res.categorical, np.array([[1, 1], [1, 0], [2, 2]])
      )
    if max_discrete_indices == 0:
      np.testing.assert_allclose(
          res.continuous,
          np.array([
              [1.0, 0.5, (10 - 1) / (20 - 1)],
              [0.1, 0.3, (10 - 1) / (20 - 1)],
              [0.2, 0.4, 1.0],
          ]),
      )
      np.testing.assert_array_equal(res.categorical, np.array([[1], [0], [2]]))
    # Test un-mapping
    unmapped_features = feature_mapper.unmap(res)
    np.testing.assert_allclose(unmapped_features, features)


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  config.update('jax_enable_x64', True)
  absltest.main()
