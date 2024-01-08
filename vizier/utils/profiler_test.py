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

"""Tests for profiler."""

import equinox as eqx
import jax
import jax.numpy as jnp
from vizier.utils import profiler

from absl.testing import absltest


class MyModule(eqx.Module):

  @profiler.record_tracing
  def add_one(self, x: int) -> int:
    return x + 1


@profiler.record_tracing
def add_one(x):
  return x + 1


class PerformanceUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.clear_caches()

  def test_eqx_jit(self):
    """`record_tracing` is tested on a bound method that is `filter_jit`-ed."""

    @profiler.record_runtime(name='return_two', also_log=True)
    def return_two():
      one = eqx.filter_jit(MyModule().add_one)(jnp.array(0.0))
      two = eqx.filter_jit(MyModule().add_one)(one)
      return two

    with profiler.collect_events() as events:
      return_two()
      return_two()

    # tracing is logged once, timer is logged twice.
    self.assertLen(events, 3)

    self.assertEqual(events[0].etype, profiler.EventType.JIT_TRACING)
    self.assertEqual(events[1].etype, profiler.EventType.TIMER)
    self.assertEqual(events[2].etype, profiler.EventType.TIMER)

    # Tracing happens within the timed functions' scope.
    self.assertIn(events[1].scope, events[0].scope)

    latencies = profiler.get_latencies_dict(events)
    self.assertIn('return_two', latencies)
    self.assertLen(latencies['return_two'], 2)

  def test_regular_jit(self):
    """`record_tracing` is tested on an unbound method."""

    @profiler.record_runtime(name='return_two', also_log=True)
    def return_two():
      one = jax.jit(add_one)(jnp.array(0.0))
      two = jax.jit(add_one)(one)
      return two

    with profiler.collect_events() as events:
      return_two()

    # tracing is logged once, timer is logged once.
    self.assertLen(events, 2)
    self.assertEqual(events[0].etype, profiler.EventType.JIT_TRACING)
    self.assertEqual(events[1].etype, profiler.EventType.TIMER)

    # Tracing happens within the timed functions' scope.
    self.assertIn(events[1].scope, events[0].scope)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
