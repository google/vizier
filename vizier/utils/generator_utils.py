# Copyright 2022 Google LLC.
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

"""Generator utilities."""

from collections.abc import Iterator
import enum
from typing import Generator, Generic, TypeVar
from typing import Optional

import attrs

_Y = TypeVar('_Y')
_S = TypeVar('_S')
_R = TypeVar('_R')


class _GeneratorState(enum.Enum):
  NEXT = 'next'
  DONE = 'done'
  SEND = 'send'


@attrs.define(init=False)
class BetterGenerator(Iterator[_Y], Generic[_Y, _S, _R]):
  """Variation of Generator where send() does not advance to next yield.

  This allows you to write:
    generator = BetterGenerator(coroutine)
    for element in generator:
      generator.send(my_fn(elment))
    result = generator.result

  as opposed to:
    generator = coroutine()
    try:
      element = next(generator)
      while True:
        element = generator.send(my_fn(elment))
      except StopIteration as e:
        result = e.value
  """

  _gen: Generator[_Y, _S, _R]
  _next: _Y
  _mode: _GeneratorState = attrs.field(init=False, default=_GeneratorState.NEXT)
  _result: Optional[_R] = attrs.field(init=False, default=None)

  def __init__(self, generator: Generator[_Y, _S, _R], /):
    self.__attrs_init__(generator, next(generator))

  @property
  def result(self) -> _R:
    if self._mode != _GeneratorState.DONE:
      raise RuntimeError('Generator is still not done')
    else:
      return self._result

  def __next__(self) -> _Y:
    if self._mode == _GeneratorState.DONE:
      raise StopIteration(self._result)
    if self._mode != _GeneratorState.NEXT:
      raise RuntimeError('Must call send.')
    self._mode = _GeneratorState.NEXT
    return self._next

  def send(self, x: _S) -> None:
    if self._mode != _GeneratorState.NEXT:
      raise RuntimeError('Must call next.')
    try:
      self._next = self._gen.send(x)
      print('next:', self._next)
      self._mode = _GeneratorState.NEXT
    except StopIteration as e:
      self._result = e.value
      self._mode = _GeneratorState.DONE
