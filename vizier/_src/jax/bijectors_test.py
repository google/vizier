"""Tests for bijectors."""

import numpy as np
from sklearn import preprocessing
from tensorflow_probability.substrates import jax as tfp

from vizier._src.jax import bijectors
from absl.testing import absltest
from absl.testing import parameterized


class PowerTransformTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(data_shape=(10,),),
      dict(data_shape=(10, 4),),
  )
  def test_yeojohnson(self, data_shape):
    """Test creation of Yeo-Johnson bijector."""
    data = np.random.random(data_shape).astype(np.float32) * 2 - 1
    data = data - tfp.stats.percentile(data, 50, axis=0)
    bijector = bijectors.optimal_power_transformation(data)
    transformed_data = bijector(data)
    np.testing.assert_allclose(transformed_data.mean(), 0.0, atol=1e-3)
    np.testing.assert_allclose(transformed_data.std(), 1.0, atol=1e-3)

    self.assertSequenceEqual(bijector.experimental_batch_shape(),
                             data_shape[1:])

    self.assertSequenceEqual(data.shape, transformed_data.shape)
    if len(data.shape) == 1:
      transformed_data = transformed_data[:, np.newaxis]
      data = data[:, np.newaxis]

    sklearn_transformed = preprocessing.PowerTransformer().fit(data).transform(
        data)
    np.testing.assert_allclose(transformed_data, sklearn_transformed, atol=1e-3)

  @parameterized.parameters(
      dict(transformation='box-cox', standardize=True),
      dict(transformation='box-cox', standardize=False),
      dict(transformation='yeo-johnson', standardize=True),
      dict(transformation='yeo-johnson', standardize=False),
  )
  def test_huge_values(self, transformation: str, standardize: bool):
    data = np.array(
        [[2201.5046], [2201.5046], [11985.564], [72969.02], [8150.587]],
        dtype=np.float32)
    bijector = bijectors.optimal_power_transformation(
        data, transformation, standardize=standardize)
    self.assertTrue(np.all(np.isfinite(bijector(data))))

  @parameterized.parameters(
      dict(data_shape=(10,),),
      dict(data_shape=(10, 4),),
  )
  def test_boxcox(self, data_shape):
    """Test creation of Box-Cox bijector."""
    data = np.random.random(data_shape).astype(np.float32) * 2
    bijector = bijectors.optimal_power_transformation(data, 'box-cox')
    transformed_data = bijector(data)
    np.testing.assert_allclose(transformed_data.mean(), 0.0, atol=1e-3)
    np.testing.assert_allclose(transformed_data.std(), 1.0, atol=1e-3)

    self.assertSequenceEqual(bijector.experimental_batch_shape(),
                             data_shape[1:])

    self.assertSequenceEqual(data.shape, transformed_data.shape)
    if len(data.shape) == 1:
      transformed_data = transformed_data[:, np.newaxis]
      data = data[:, np.newaxis]

    sklearn_transformed = preprocessing.PowerTransformer('box-cox').fit(
        data).transform(data)
    np.testing.assert_allclose(transformed_data, sklearn_transformed, atol=1e-3)


class SignFlipTest(parameterized.TestCase):

  def test_flip_sign_basic(self):
    flipped = bijectors.flip_sign(np.array([[False, True], [True,
                                                            False]]))([1., 2.])
    np.testing.assert_allclose(flipped, np.array([[1., -2.], [-1., 2.]]))


if __name__ == '__main__':
  absltest.main()
