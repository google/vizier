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

"""Utility functions for performance tracking.

A typical use case is:

```
@jax.jit
@profiler.record_tracing
def _compute(self, x: jax.Array):
  return x + 1

class MyDesigner:

  @profiler.record_runtime
  def suggest(self) -> None:
    # An example of how to annotate an inline function call, so the function
    # name `_compute` is used. Note that the name must be specified.
    with profiler.timeit('compute'):
      _compute(jnp.array(1.))

with profiler.collect_events() as events:
  MyDesigner().suggest()
```

`events` is a list of `ProfileEvent`s. In this example, there are 3 events:
* Tracing of `_compute`, in scope `suggest::compute`
* Timing of `compute`, in scope `suggest`
* Timing of `suggest`, in empty scope.

`profiler.get_latencies_dict(events)` returns a dictionary whose values are
list of timedelta objects.

NOTE: @profiler.record_runtime and @profiler.record_tracing should appear
BEFORE the @jax.jit decorator, or only the first runtime will be recorded.
"""
import collections
import contextlib
import datetime
import enum
import functools
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence

from absl import logging
import attrs
import jax


class EventType(enum.Enum):
  JIT_TRACING = 'tracing'
  TIMER = 'timer'


@attrs.define
class ProfileEvent:
  """An event logged by the profiler and stored in _Storage.

  Attributes:
    etype: type of the event.
    scope: context of the event.
    data: For JIT_TRACING event, it is either a singleton "function_name: str"
      OR a tuple of "function_name: str, args: tuple, kwargs: dict". For TIMER
      event, it's a tuple of "name: str" and "duration: timedelta".
  """

  etype: EventType = attrs.field()
  scope: str = attrs.field(converter='::'.join)
  data: Any = attrs.field()


@attrs.define
class _Storage:
  """Storage for function call runtimes.

  There is a single `_Storage` instance used globally. The global instance
  stays in "inactive" state until code enters a `collect_events()`
  contextmanager.

  Attributes:
    active: If not True, `add` method is a no-op.
    events:
    scope: Currently active "scope". Each time code enters
      `_enter_profile_scope` contextmanager, the scope name is appended to the
      list. Makes it easier to track where things happened.
  """

  active: bool = attrs.field(default=False)
  events: List[ProfileEvent] = attrs.field(factory=list)
  scope: list[str] = attrs.field(factory=list)

  def add(self, etype: EventType, data: Any) -> ProfileEvent:
    """Create and add a new event and return the added event.

    Args:
      etype: See ProfileEvent.
      data: See ProfileEvent.

    Returns:
      Newly added event.
    """
    event = ProfileEvent(etype, self.scope, data)
    if self.active:
      self.events.append(event)
    return event


_GLOBAL_SOTRAGE: _Storage = _Storage()


@contextlib.contextmanager
def _enter_profile_scope(scope: str):
  try:
    _GLOBAL_SOTRAGE.scope.append(scope)
    yield current_scope()
  finally:
    _GLOBAL_SOTRAGE.scope.pop()


def current_scope() -> str:
  return '::'.join(_GLOBAL_SOTRAGE.scope)


@contextlib.contextmanager
def collect_events() -> Generator[List[ProfileEvent], None, None]:
  """Context manager for turning on the profiler."""
  try:
    if _GLOBAL_SOTRAGE.active:
      raise RuntimeError(
          'There can be only one `collect_events()` context manager active at'
          ' the same time.'
      )
    _GLOBAL_SOTRAGE.active = True
    yield _GLOBAL_SOTRAGE.events
  finally:
    _GLOBAL_SOTRAGE.active = False
    # It's important to create a new instance here as opposed to resetting
    # the same object, so the yieled `events` persist outside the context.
    _GLOBAL_SOTRAGE.events = list()


@contextlib.contextmanager
def timeit(
    name: str, also_log: bool = False
) -> Generator[Callable[[], datetime.timedelta], None, None]:
  """Context manager for measuring the timing.

  Example:
  ```
  with timeit('scope_name') as duration:
    ...

  duration() # returns the duration.
  ```

  Also see: record_runtime, which is the decorator equivalent of this.

  Args:
    name:
    also_log: If True, also create a log.

  Yields:
    A callable with zero input arguments. Returns the elapsed time until call,
    or the end of the context, whichever comes earlier.
  """
  start = datetime.datetime.now()
  duration = None
  try:
    with _enter_profile_scope(name):

      def func() -> datetime.timedelta:
        if duration is None:
          return datetime.datetime.now() - start
        return duration

      yield func

  finally:
    duration = datetime.datetime.now() - start
    event = _GLOBAL_SOTRAGE.add(EventType.TIMER, (name, duration))
    if also_log:
      logging.info(
          'Timed %s: took %s seconds. Full event string: %s',
          name,
          duration.total_seconds(),
          event,
      )


def get_latencies_dict(
    events: Sequence[ProfileEvent],
) -> Dict[str, list[datetime.timedelta]]:
  d = collections.defaultdict(list)
  for e in events:
    if e.etype == EventType.TIMER:
      d[e.data[0]].append(e.data[1])
  return dict(d)


def record_runtime(
    func: Optional[Callable[..., Any]] = None,
    *,
    name_prefix: str = '',
    name: str = '',
    also_log: bool = False,
    block_until_ready: bool = False,
) -> Any:
  """Decorates the function to record the runtime.

  Also see: timeit(), which is the context manager equivalent of this.

  Args:
    func: Function being decorated.
    name_prefix: A prefix to add to the function name.
    name: The name to record. Defaults to func.__qualname__.
    also_log: Whether to also logging.info the runtime duration.
    block_until_ready: If True and running on an acceleartor, wait for the async
      dispatch results from this function, if any. See
      https://jax.readthedocs.io/en/latest/async_dispatch.html

  Returns:
    Decorated function, or decorator.
  """
  # This is required for the decorator to work both with and without
  # optional arguments.
  if func is None:
    return functools.partial(
        record_runtime,
        name_prefix=name_prefix,
        name=name,
        also_log=also_log,
        block_until_ready=block_until_ready,
    )
  name = name or func.__qualname__
  full_name = name
  if name_prefix:
    full_name = f'{name_prefix}.{name}'

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    """Calculates runtime of the given function."""
    with timeit(name=full_name, also_log=also_log):
      result = func(*args, **kwargs)
      if block_until_ready:
        result = jax.block_until_ready(result)
    return result
  return wrapper


@attrs.define
class _Config:
  """Configuration for this module."""

  _include_args_in_trace_records: bool = attrs.field(default=False)

  def include_args_in_trace_records(self, v: Optional[bool] = None) -> bool:
    """(Setter+getter): Whether to include input arguments in tracing events.

    NOTE: Use this feature for debugging only. If the function decorated with
    `@record_tracing` is not jitted, then arguments and kwargs will be actual
    Array(Tree)s instead of Tracers. The logs may include full values of the
    array which can greatly slow down your binary.

    Args:
      v: If None, leaves the behavior unchanged.

    Returns:
      New include_args_in_trace_records value.
    """
    if v is not None:
      self._include_args_in_trace_records = v
    return self._include_args_in_trace_records


config = _Config()


def record_tracing(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: str = '',
    also_log: bool = True,
) -> Any:
  """Decorates the function to record the runtime of functions.

  Args:
    func: Function being decorated.
    name: The name to record. Defaults to func.__qualname__.
    also_log: Whether to also logging.info the runtime duration.

  Returns:
    Decorated function, or decorator.
  """
  # This is required for the decorator to work both with and without
  # optional arguments.
  if func is None:
    return functools.partial(
        record_tracing,
        name=name,
        also_log=also_log,
    )
  name = name or func.__qualname__

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    if config.include_args_in_trace_records():
      data = f'{name} with args={args} kwargs={kwargs}'
    else:
      data = name

    event = _GLOBAL_SOTRAGE.add(EventType.JIT_TRACING, data)
    if also_log:
      logging.info('Tracing %s. Full event string: %s', name, event)
    with _enter_profile_scope(name):
      result = func(*args, **kwargs)
    return result

  return wrapper
