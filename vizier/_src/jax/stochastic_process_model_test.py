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

"""Tests for stochastic_process_model."""

import functools
from unittest import mock

from absl.testing import parameterized
from flax import linen as nn

import jax
from jax import numpy as jnp
from jax import random
from jax.config import config
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import tree
from vizier._src.jax import stochastic_process_model as sp_model
from absl.testing import absltest

config.update('jax_enable_x64', True)

tfb = tfp.bijectors
tfd = tfp.distributions
tfpk = tfp.math.psd_kernels
tfde = tfp.experimental.distributions
tfpke = tfp.experimental.psd_kernels


def _continuous_kernel_coroutine(dtype=np.float64):
  constraint = sp_model.Constraint(
      bounds=(np.zeros([], dtype=dtype), None), bijector=tfb.Exp()
  )
  amplitude = yield sp_model.ModelParameter(
      init_fn=lambda k: random.exponential(k, dtype=dtype),
      regularizer=lambda x: dtype(1e-3) * x**2,
      constraint=constraint,
      name='amplitude',
  )
  inverse_length_scale = yield sp_model.ModelParameter.from_prior(
      tfd.Exponential(
          rate=np.ones([], dtype=dtype), name='inverse_length_scale'
      ),
      constraint=constraint,
  )
  return tfpk.ExponentiatedQuadratic(
      amplitude=amplitude,
      inverse_length_scale=inverse_length_scale,
      validate_args=True,
  )


def _categorical_kernel_coroutine(dtype=np.float64):
  amplitude = yield sp_model.ModelParameter(
      init_fn=lambda k: random.exponential(k, dtype=dtype),
      constraint=sp_model.Constraint(bounds=(np.zeros([], dtype=dtype), None)),
      regularizer=lambda x: 1e-3 * x**2,
      name='amplitude',
  )
  one = np.ones([], dtype=dtype)
  inverse_length_scale_continuous = yield sp_model.ModelParameter.from_prior(
      tfd.Exponential(rate=one, name='inverse_length_scale_continuous')
  )
  inverse_length_scale_categorical = yield sp_model.ModelParameter.from_prior(
      tfd.Exponential(rate=one, name='inverse_length_scale_categorical')
  )
  inverse_length_scale = tfpke.ContinuousAndCategoricalValues(
      inverse_length_scale_continuous, inverse_length_scale_categorical
  )
  kernel = tfpk.ExponentiatedQuadratic(amplitude=amplitude, validate_args=True)
  return tfpke.FeatureScaledWithCategorical(
      kernel, inverse_length_scale, validate_args=True
  )


def _test_coroutine(
    inputs=None,
    kernel_coroutine=_continuous_kernel_coroutine,
    num_tasks=1,
    dtype=np.float64,
):
  """A coroutine that follows the `ModelCoroutine` protocol."""
  kernel = yield from kernel_coroutine(dtype=dtype)
  if num_tasks == 1:
    return tfd.StudentTProcess(
        df=dtype(5.0),
        kernel=kernel,
        index_points=inputs,
        observation_noise_variance=np.ones([], dtype=dtype),
        validate_args=True,
    )

  multi_task_kernel = tfpke.Independent(num_tasks=num_tasks, base_kernel=kernel)
  return tfde.MultiTaskGaussianProcess(
      kernel=multi_task_kernel,
      index_points=inputs,
      observation_noise_variance=np.ones([], dtype=dtype),
      validate_args=True,
  )


def _make_inputs(key, dtype, num_tasks=1, num_observed=20, num_predictive=5):
  x_key, y_key, pred_key = random.split(key, num=3)
  dim = 3
  y_shape = (num_observed,) if num_tasks == 1 else (num_observed, num_tasks)
  x_observed = random.uniform(x_key, shape=(num_observed, dim), dtype=dtype)
  y_observed = random.uniform(y_key, shape=y_shape, dtype=dtype)
  x_predictive = random.uniform(
      pred_key, shape=(100, num_predictive, dim), dtype=dtype
  )
  return x_observed, y_observed, x_predictive


def _make_inputs_with_categorical(
    key, dtype, num_tasks=1, num_observed=15, num_predictive=8
):
  cont_obs_key, cat_obs_key, cont_pred_key, cat_pred_key, y_key = random.split(
      key, num=5
  )
  cont_dim = 5
  cat_dim = 3
  x_observed_cont = random.uniform(
      cont_obs_key, shape=(num_observed, cont_dim), dtype=dtype
  )
  x_observed_cat = random.randint(
      cat_obs_key, shape=(num_observed, cat_dim), minval=0, maxval=6
  )
  x_observed = tfpke.ContinuousAndCategoricalValues(
      x_observed_cont, x_observed_cat
  )
  y_shape = (num_observed,) if num_tasks == 1 else (num_observed, num_tasks)
  y_observed = random.uniform(y_key, shape=y_shape, dtype=dtype)

  x_predictive_cont = random.uniform(
      cont_pred_key, shape=(100, num_predictive, cont_dim), dtype=dtype
  )
  x_predictive_cat = random.randint(
      cat_pred_key, shape=(num_predictive, cat_dim), minval=0, maxval=6
  )
  x_predictive = tfpke.ContinuousAndCategoricalValues(
      x_predictive_cont, x_predictive_cat
  )
  return x_observed, y_observed, x_predictive


class StochasticProcessModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # TODO: Add a test case with categorical data.
      # TODO: Fix support for f32.
      {
          'testcase_name': 'continuous_only',
          'kernel_coroutine': _continuous_kernel_coroutine,
          'test_data_fn': _make_inputs,
          'num_tasks': 1,
          'dtype': np.float64,
      },
      {
          'testcase_name': 'multitask_continuous',
          'kernel_coroutine': _continuous_kernel_coroutine,
          'test_data_fn': _make_inputs,
          'num_tasks': 3,
          'dtype': np.float64,
      },
      {
          'testcase_name': 'continuous_categorical',
          'kernel_coroutine': _categorical_kernel_coroutine,
          'test_data_fn': _make_inputs_with_categorical,
          'num_tasks': 1,
          'dtype': np.float64,
      },
      {
          'testcase_name': 'continuous_categorical_multitask',
          'kernel_coroutine': _categorical_kernel_coroutine,
          'test_data_fn': _make_inputs_with_categorical,
          'num_tasks': 1,
          'dtype': np.float64,
      },
  )
  def test_stochastic_process_model(
      self, kernel_coroutine, test_data_fn, num_tasks, dtype
  ):
    init_key, data_key, sample_key = jax.random.split(random.PRNGKey(0), num=3)
    x_observed, y_observed, x_predictive = test_data_fn(
        data_key, dtype=dtype, num_tasks=num_tasks
    )
    model_coroutine = functools.partial(
        _test_coroutine,
        kernel_coroutine=kernel_coroutine,
        num_tasks=num_tasks,
        dtype=dtype,
    )
    model = sp_model.StochasticProcessModel(coroutine=model_coroutine)

    init_state = model.init(init_key, x_observed)
    dist, losses = model.apply(init_state, x_observed, mutable=('losses',))
    lp = dist.log_prob(y_observed)
    self.assertTrue(np.isfinite(lp))
    self.assertEqual(lp.dtype, dtype)

    # Check regularization loss values.
    gen = model_coroutine()
    p = next(gen)
    params = dict(init_state['params'])
    losses = dict(losses['losses'])
    try:
      while True:
        value = params[p.name]
        param_loss = losses[f'{p.name}_regularization']
        self.assertTrue(np.isfinite(value))
        self.assertEqual(value.dtype, dtype)
        self.assertTrue(np.isfinite(param_loss))
        self.assertAlmostEqual(p.regularizer(value), param_loss)
        p = gen.send(value)
    except StopIteration:
      pass

    _, pp_state = model.apply({'params': params},
                              x_observed,
                              y_observed,
                              method=model.precompute_predictive,
                              mutable=('predictive',))

    state = {'params': params, **pp_state}
    pp_dist = model.apply(
        state,
        x_predictive,
        x_observed,
        y_observed,
        method=model.posterior_predictive,
    )
    pp_log_prob = pp_dist.log_prob(pp_dist.sample(seed=sample_key))
    self.assertTrue(np.isfinite(pp_log_prob).all())
    self.assertEqual(pp_log_prob.dtype, dtype)

  @parameterized.named_parameters(
      {
          'testcase_name': 'continuous_only',
          'num_tasks': 1,
          'kernel_coroutine': _continuous_kernel_coroutine,
          'test_data_fn': _make_inputs,
      },
      {
          'testcase_name': 'multitask_continuous',
          'num_tasks': 3,
          'kernel_coroutine': _continuous_kernel_coroutine,
          'test_data_fn': _make_inputs,
      },
      {
          'testcase_name': 'multitask_continuous_categorical',
          'num_tasks': 3,
          'kernel_coroutine': _categorical_kernel_coroutine,
          'test_data_fn': _make_inputs_with_categorical,
      },
  )
  def test_jax_transformations(self, num_tasks, kernel_coroutine, test_data_fn):
    init_key, data_key, prior_key, post_key = jax.random.split(
        random.PRNGKey(0), num=4
    )
    num_observed = 20
    x_observed, y_observed, x_predictive = test_data_fn(
        data_key,
        num_tasks=num_tasks,
        num_observed=num_observed,
        dtype=np.float32,
    )
    model = sp_model.StochasticProcessModel(
        coroutine=functools.partial(
            _test_coroutine,
            kernel_coroutine=kernel_coroutine,
            num_tasks=num_tasks,
            dtype=np.float32,
        )
    )

    batch_size = 10
    keys = jax.random.split(init_key, num=batch_size)
    init_state = jax.vmap(lambda k: model.init(k, x_observed))(keys)
    s = jax.jit(
        jax.vmap(
            lambda s: model.apply(  # pylint: disable=g-long-lambda
                s, x_observed, mutable=('losses',)
            )[0].sample(seed=prior_key)
        )
    )(init_state)
    sample_shape = (
        (batch_size, num_observed)
        if num_tasks == 1
        else (batch_size, num_observed, num_tasks)
    )
    self.assertEqual(s.shape, sample_shape)

    _, pp_state = jax.jit(
        jax.vmap(
            lambda p: model.apply(  # pylint: disable=g-long-lambda
                p,
                x_observed,
                y_observed,
                method=model.precompute_predictive,
                mutable=('predictive',),
            )
        )
    )(init_state)

    state = {**init_state, **pp_state}

    @jax.jit
    @jax.vmap
    def _posterior_sample_and_log_prob(s):
      dist = model.apply(
          s,
          x_predictive,
          x_observed,
          y_observed,
          method=model.posterior_predictive,
      )[0]
      sample = dist.sample(seed=post_key)
      return dist.log_prob(sample)

    pp_log_prob = _posterior_sample_and_log_prob(state)
    log_prob_grad = jax.grad(
        lambda s: jnp.sum(_posterior_sample_and_log_prob(s))
    )(state)
    for g in log_prob_grad['params'].values():
      self.assertTrue(np.isfinite(g).all())
    self.assertTrue(np.isfinite(pp_log_prob).all())
    self.assertEqual(pp_log_prob.shape, (batch_size,))

  def test_cholesky_not_recomputed(self):
    # Need num_observed == num_predictive so the mock Cholesky works for both
    # observed and predictive.
    x_obs, y_obs, x = _make_inputs(
        jax.random.PRNGKey(5), np.float32, num_observed=12, num_predictive=12
    )
    chol = np.eye(12, dtype=np.float32)
    mock_cholesky_fn = mock.Mock(return_value=chol)

    def _coro_with_mock_cholesky(inputs=None):
      stp = yield from _test_coroutine(inputs=inputs, dtype=np.float32)
      # The `copy` method of TFP distributions rebuilds the distribution object
      # with the given constructor kwarg(s) replaced.
      return stp.copy(cholesky_fn=mock_cholesky_fn)

    model = sp_model.StochasticProcessModel(coroutine=_coro_with_mock_cholesky)
    keys = jax.random.split(jax.random.PRNGKey(0), num=8)
    params = jax.vmap(lambda k: model.init(k, x_obs))(keys)
    _, pp_state = jax.vmap(
        lambda p: model.apply(  # pylint: disable=g-long-lambda
            p,
            x_obs,
            y_obs,
            method=model.precompute_predictive,
            mutable=('predictive',),
        )
    )(params)
    mock_cholesky_fn.assert_called_once()

    # Sampling from the posterior predictive should invoke cholesky_fn only
    # once (on the kernel matrix for the predictive index points).
    mock_cholesky_fn.reset_mock()
    state = {**params, **pp_state}
    _ = jax.jit(
        jax.vmap(
            lambda s: model.apply(  # pylint: disable=g-long-lambda
                s, x, x_obs, y_obs, method=model.posterior_predictive
            ).sample(seed=jax.random.PRNGKey(0))
        )
    )(state)
    mock_cholesky_fn.assert_called_once()

    # Sample new data and compute the posterior predictive.
    mock_cholesky_fn.reset_mock()
    x_obs2, y_obs2, x2 = _make_inputs(
        jax.random.PRNGKey(6), np.float32, num_observed=12, num_predictive=12
    )
    params2 = model.init(jax.random.PRNGKey(7), x_obs2)
    _, pp_state2 = model.apply(
        params2,
        x_obs2,
        y_obs2,
        method=model.precompute_predictive,
        mutable=('predictive',),
    )
    _ = jax.jit(
        lambda s: model.apply(  # pylint: disable=g-long-lambda
            s, x2, x_obs2, y_obs2, method=model.posterior_predictive
        ).sample(seed=jax.random.PRNGKey(0))
    )({**params2, **pp_state2})
    # Two Cholesky calls; one for observed, one for predictive.
    self.assertEqual(mock_cholesky_fn.call_count, 2)

    # Assert that `model.posterior_predictive`, when called on the original
    # dataset, does not recompute the Cholesky.
    mock_cholesky_fn.reset_mock()
    state = {**params, **pp_state}
    _ = jax.jit(
        jax.vmap(
            lambda s: model.apply(  # pylint: disable=g-long-lambda
                s, x, x_obs, y_obs, method=model.posterior_predictive
            ).sample(seed=jax.random.PRNGKey(0))
        )
    )(state)
    mock_cholesky_fn.assert_called_once()

  def test_stochastic_process_model_with_mean_fn(self):

    obs_key, pred_key, init_key, vmap_key = random.split(
        random.PRNGKey(0), num=4)
    dim = 3
    num_observed = 20
    x_observed = random.uniform(obs_key, shape=(num_observed, dim))
    y_observed = x_observed.sum(axis=-1)
    x_predictive = random.uniform(pred_key, shape=(100, 5, dim))

    # Use `jnp.squeeze` to remove the singleton dimension of the output of
    # `mean_fn`, so that it has shape `(num_observed,)`.
    mean_fn = nn.Sequential([nn.Dense(5), nn.Dense(1), jnp.squeeze])
    model = sp_model.StochasticProcessModel(
        coroutine=_test_coroutine, mean_fn=mean_fn)

    init_state = model.init(init_key, x_observed)
    stp = model.apply(init_state, x_observed, mutable=False)
    lp = stp.log_prob(y_observed)
    self.assertTrue(np.isfinite(lp))

    # The mean of the GP should equal `mean_fn` evaluated at the index points.
    np.testing.assert_allclose(
        stp.mean(),
        mean_fn.apply({'params': init_state['params']['mean_fn']}, x_observed))

    _, pp_state = model.apply({'params': init_state['params']},
                              x_observed,
                              y_observed,
                              method=model.precompute_predictive,
                              mutable=('predictive',))

    state = {'params': init_state['params'], **pp_state}
    pp_dist = model.apply(
        state,
        x_predictive,
        x_observed,
        y_observed,
        method=model.posterior_predictive,
    )
    pp_mean = pp_dist.mean()

    self.assertTrue(
        np.isfinite(pp_dist.log_prob(x_predictive.sum(axis=-1))).all())
    self.assertSequenceEqual(pp_mean.shape, (100, 5))
    self.assertTrue(np.isfinite(pp_mean).all())

    # Test that vmap works.
    def log_prob(p):
      return model.apply(
          p, x_observed, mutable=('losses',))[0].log_prob(y_observed)

    batch_size = 20
    keys = random.split(vmap_key, num=batch_size)
    params = jax.vmap(lambda k: model.init(k, x_observed))(keys)
    lp = jax.vmap(log_prob)(params)
    self.assertLen(lp, batch_size)
    self.assertTrue(np.isfinite(lp).all())


class VectorToArrayTreeTest(absltest.TestCase):

  def test_vector_to_array_tree(self):
    k0, k1 = random.split(random.PRNGKey(0))
    params = {
        'foo': random.uniform(k0, shape=(2, 3)),
        'bar': np.array(0.3),
        'baz': random.uniform(k1, shape=(6,))
    }
    bijector = sp_model.VectorToArrayTree(params)
    self.assertEqual(bijector.inverse(params).shape, (13,))

    v = np.ones([13])
    struct = bijector.forward(v)
    self.assertSameElements(list(struct.keys()), list(params.keys()))
    for k in params:
      self.assertEqual(struct[k].shape, params[k].shape)


class ModelParameterTest(absltest.TestCase):

  def test_parameter_from_prior(self):
    prior = tfd.Gamma(concentration=1., rate=1.)
    param = sp_model.ModelParameter.from_prior(prior=prior)
    samples = param.init_fn(random.PRNGKey(0))
    regularization = param.regularizer(samples)
    x = np.linspace(-10.0, 10.0, 20)
    self.assertTrue(np.isfinite(regularization))
    self.assertEmpty(regularization.shape)
    self.assertNotEqual(regularization, 0.)
    self.assertTrue((param.constraint.bijector(x) > 0.0).all())


class ConstraintTest(parameterized.TestCase):

  @parameterized.parameters(
      (-1.0, None),
      (0.0, 10.0),
      (None, 1.0),
      ({'a': -1.0, 'b': 0.0}, {'a': 1.0, 'b': 10.0}),
      ({'a': None, 'b': 0.0}, {'a': 1.0, 'b': None}),
      (None, {'a': 1.0, 'b': 2.0}),
  )
  def test_create_constraint(self, lower, upper):
    make_array = lambda x: None if x is None else jnp.array(x)
    lower = jax.tree_util.tree_map(make_array, lower)
    upper = jax.tree_util.tree_map(make_array, upper)
    constraint = sp_model.Constraint.create(
        bounds=(lower, upper), bijector_fn=tfb.SoftClip
    )
    x_part = jnp.linspace(-5.0, 5.0, 10)
    x = tree.map_structure(lambda _: x_part, lower or upper)
    y = constraint.bijector(x)
    for y_, b_ in zip(tree.flatten(y), tree.flatten(lower)):
      if b_ is not None:
        self.assertTrue((y_ > b_).all())
    for y_, b_ in zip(tree.flatten(y), tree.flatten(upper)):
      if b_ is not None:
        self.assertTrue((y_ < b_).all())

  def test_get_constraints(self):
    constraint = sp_model.get_constraints(_test_coroutine)
    x_part = np.linspace(-5.0, 5.0, 10)
    x = {'amplitude': x_part, 'inverse_length_scale': x_part}
    y = constraint.bijector(x)
    lower, upper = constraint.bounds

    self.assertSameElements(list(x.keys()), list(y.keys()))
    for y_, b_ in zip(tree.flatten(y), tree.flatten(lower)):
      if b_ is not None:
        self.assertTrue((y_ > b_).all())
    for y_, b_ in zip(tree.flatten(y), tree.flatten(upper)):
      if b_ is not None:
        self.assertTrue((y_ < b_).all())


if __name__ == '__main__':
  absltest.main()
