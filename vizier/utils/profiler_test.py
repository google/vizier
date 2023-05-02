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

"""Tests for profiler."""
import jax
import jax.numpy as jnp

from vizier.utils import profiler
from absl.testing import absltest


class PerformanceUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    profiler.Storage().reset()

  def test_no_args(self):
    class TestClass:

      @profiler.record_runtime(also_log=True)
      def add_one(self, x: int) -> int:
        return x + 1

    t = TestClass()
    t.add_one(1)
    self.assertIn('add_one', profiler.Storage().runtimes())
    runtimes = profiler.Storage().runtimes().get('add_one')
    self.assertLen(runtimes, 1)
    self.assertGreaterEqual(runtimes[0].total_seconds(), 0)

  def test_with_name(self):
    class TestClass3:

      @profiler.record_runtime(name='add')
      def add_one(self, x: int) -> int:
        return x + 1

    t = TestClass3()
    t.add_one(1)
    self.assertNotIn('add_one', profiler.Storage().runtimes())
    self.assertIn('add', profiler.Storage().runtimes())
    runtimes = profiler.Storage().runtimes().get('add')
    self.assertLen(runtimes, 1)
    self.assertGreaterEqual(runtimes[0].total_seconds(), 0)

  def test_with_prefix(self):
    class TestClass4:

      @profiler.record_runtime(name_prefix='foo')
      def add_one(self, x: int) -> int:
        return x + 1

    t = TestClass4()
    t.add_one(1)
    self.assertIn('foo.add_one', profiler.Storage().runtimes())
    self.assertNotIn('add_one', profiler.Storage().runtimes())
    runtimes = profiler.Storage().runtimes().get('foo.add_one')
    self.assertLen(runtimes, 1)
    self.assertGreaterEqual(runtimes[0].total_seconds(), 0)

  def test_with_prefix_and_name(self):
    class TestClass5:

      @profiler.record_runtime(name_prefix='foo', name='add')
      def add_one(self, x: int) -> int:
        return x + 1

    t = TestClass5()
    t.add_one(1)
    self.assertIn('foo.add', profiler.Storage().runtimes())
    self.assertNotIn('add_one', profiler.Storage().runtimes())
    runtimes = profiler.Storage().runtimes().get('foo.add')
    self.assertLen(runtimes, 1)
    self.assertGreaterEqual(runtimes[0].total_seconds(), 0)

  def test_with_jax(self):
    @profiler.record_runtime(also_log=True)
    @jax.jit
    def selu(x, alpha=1.67, lambda_=1.05):
      return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    x = jnp.arange(1000000)
    # The first call should jit compile, so should be slow.
    selu(x)
    self.assertIn('selu', profiler.Storage().runtimes())
    microseconds = profiler.Storage().runtimes().get('selu')[0].microseconds
    self.assertGreaterEqual(microseconds, 1000)
    # The second call should use the jitted function.
    selu(x)
    self.assertLen(profiler.Storage().runtimes().get('selu'), 2)
    jt_microseconds = profiler.Storage().runtimes().get('selu')[1].microseconds
    # Should be at least 7 times faster (rough estimate so it's not flaky).
    self.assertLess(jt_microseconds, microseconds / 7)

  def test_with_jax_reverse_decorator_order(self):
    # If jax.jit decorator is first, only the first runtime is recorded.
    @jax.jit
    @profiler.record_runtime(also_log=True)
    def selu(x, alpha=1.67, lambda_=1.05):
      return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    x = jnp.arange(1000000)
    # The first call should jit compile, so should be slow.
    selu(x)
    self.assertIn('selu', profiler.Storage().runtimes())
    microseconds = profiler.Storage().runtimes().get('selu')[0].microseconds
    self.assertGreaterEqual(microseconds, 1000)
    # The second call should use the jitted function.
    selu(x)
    self.assertLen(profiler.Storage().runtimes().get('selu'), 1)
    jt_microseconds = profiler.Storage().runtimes().get('selu')[0].microseconds
    # Should be at least 10 times faster.
    self.assertEqual(jt_microseconds, microseconds)

  def test_context_manager(self):
    x = 0
    with profiler.record_runtime_context(name='add'):
      x += 1
    self.assertIn('add', profiler.Storage().runtimes())
    runtimes = profiler.Storage().runtimes().get('add')
    self.assertLen(runtimes, 1)
    self.assertGreaterEqual(runtimes[0].total_seconds(), 0)

    with profiler.record_runtime_context(name='add'):
      x += 1
    runtimes = profiler.Storage().runtimes().get('add')
    self.assertLen(runtimes, 2)
    self.assertGreaterEqual(runtimes[1].total_seconds(), 0)


if __name__ == '__main__':
  absltest.main()
