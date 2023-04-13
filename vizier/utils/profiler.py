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

"""Utility functions for performance tracking.

A typical use case is:

class MyDesigner:

  def _compute(self, x: int):
    pass

  @profiler.record_runtime
  def suggest(self) -> None:
    # An example of how to annotate an inline function call, so the function
    # name `_compute` is used. Note that the name must be specified.
    with profiler.record_runtime_context(name='compute'):
      self._compute(x=1)

    # An example of how to annotate an inline code block:
    with profiler.record_runtime_context(name='compute_twice'):
      self._compute(x=2)
      self._compute(x=2)

  @profiler.record_runtime(name_prefix='designer')
  def suggest2(self) -> None:
    pass

  @profiler.record_runtime(name_prefix='designer', name='suggest')
  @jax.jit
  def suggest3(self) -> None:
    pass


designer = MyDesigner()
designer.suggest()

profiler.Storage().runtimes() returns a dict with these keys:
* 'suggest' (the name of the suggest method)
* 'compute' (from the call with x=1)
* 'compute_twice' (from the calls with x=2)
* 'designer.suggest2' (the name of the suggest2 method with the prefix)
* 'designer.suggest' (the specified decorator name for suggest3 with the prefix)
values will be lists of datetime.timedelta objects.

profiler.Storage().runtimes('suggest') will return a list with
datetime.timedelta objects for only `suggest` function calls.

NOTE that @profiler.record_runtime should appear FIRST, before the
@jax.jit decorator, or only the first runtime will be recorded.
"""
import collections
import contextlib
import datetime
import functools
from typing import Any, Generator

from absl import logging
import jax


class Storage:
  """Storage for function call runtimes."""

  def __new__(cls):
    if not hasattr(cls, '_instance'):
      cls._instance = super(Storage, cls).__new__(cls)
      # Add attributes below this line.
      # -------------------------------
      # A dict to track runtime performance.
      # Keys are computation names, values are computation duration.
      cls._instance.storage = collections.defaultdict(list)
    return cls._instance

  def record(self, name: str, duration: datetime.timedelta) -> None:
    """Records the runtime for the function."""
    self._instance.storage[name].append(duration)

  def reset(self) -> None:
    """Removes all performance information collected."""
    self._instance.storage.clear()

  def runtimes(self) -> dict[str, list[datetime.timedelta]]:
    """Returns the stored runtimes.

    Returns:
      A dict with all function names and their timedelta runtimes.
    """
    return dict(self._instance.storage)


def record_runtime(
    func=None,
    *,
    name_prefix: str = '',
    name: str = '',
    also_log: bool = False,
) -> Any:
  """A decorator for recording the runtime of functions.

  Args:
    func: Function being decorated.
    name_prefix: A prefix to add to the function name.
    name: The name to record. Defaults to func.__name__.
    also_log: Whether to also logging.info the runtime duration.

  Returns:
    The function return value.
  """
  # This is required for the decorator to work both with and without
  # optional arguments.
  if func is None:
    return functools.partial(
        record_runtime,
        name_prefix=name_prefix,
        name=name,
        also_log=also_log,
    )
  name = name or func.__name__
  if name_prefix:
    full_name = f'{name_prefix}.{name}'
  else:
    full_name = name

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    """Calculates runtime of the given function."""
    start = datetime.datetime.now()
    # This works fine for jitted functions as well.
    result = func(*args, **kwargs)
    # If running on an acceleartor, wait for the async dispatch result.
    # See https://jax.readthedocs.io/en/latest/async_dispatch.html
    result = jax.block_until_ready(result)
    duration = datetime.datetime.now() - start
    Storage().record(full_name, duration)
    if also_log:
      logging.info(
          '%s: Duration %s seconds', full_name, duration.total_seconds()
      )
    return result

  return wrapper


@contextlib.contextmanager
def record_runtime_context(
    name: str, also_log: bool = False
) -> Generator[None, None, None]:
  """A context manager to record the runtime of a function call/code block.

  Args:
    name: The name of the function call or code block to record timing for
    also_log: Whether to also logging.info the runtime duration.

  Yields:
    Nothing.
  """
  start = datetime.datetime.now()
  yield
  duration = datetime.datetime.now() - start
  Storage().record(name, duration)
  if also_log:
    logging.info('%s: Duration %s seconds', name, duration.total_seconds())
