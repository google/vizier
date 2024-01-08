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

"""Serializable interface."""

import abc
from typing import Type, TypeVar

from vizier import pyvizier as vz

_S = TypeVar('_S', bound='Serializable')


class DecodeError(Exception):
  """Parent class for errors during decoding."""
  pass


class HarmlessDecodeError(DecodeError):
  """Failed to load state from metadata, and the object was untouched."""


class FatalDecodeError(DecodeError):
  """Failed to load state from metadata, and the object became invalid."""


class PartiallySerializable(abc.ABC):
  """Partially serializable objects except initialization.

  As long as the object is created in the same manner (via __init__ or a
  factory @classmethod), `load()` recovers the exact same `State`.

  NOTE: `State` here refers to the behavioral state that can be verified by
  calling "public" (not prefixed with an underline) methods. Subclasses do not
  guarantee that all private variables are recovered precisely.


  class Foo(PartiallySerializable):
   ...

  foo = Foo(*init_args)
  # do something with foo
  ...
  md = foo.dump()

  # `foo` and `foo2` are indistinguishable using "public" methods of Foo.
  foo2 = Foo(*init_args).load(md)

  # `foo` and `foo3` can be arbitrarily different.
  foo3 = Foo(*different_init_args).load(md)
  """

  @abc.abstractmethod
  def load(self, metadata: vz.Metadata) -> None:
    """Recovers the object's state stored in metadata.

    Args:
      metadata:

    Raises:
      HarmlessDecodeError: The object is still valid and can be used.
      FatalDecodeError: The object became invalid due to failure during
        load. It should be discarded. Raising a FatalDecodeError is NOT
        recommended, unless doing so avoids a signifcant overhead in performance
        or code complexity.
    """
    pass

  @abc.abstractmethod
  def dump(self) -> vz.Metadata:
    pass


class Serializable(abc.ABC):
  """Objects that can be fully serialized.

  Compared to PartiallySerializable, Serializable.recover() is a _classmethod_
  not a regular method. The object's `State` in its entirety can be serialized
  and then recovered.

  NOTE: `State` here refers to the behavioral state that can be verified by
  calling "public" (not prefixed with an underline) methods. Subclasses do not
  guarantee that all private variables are recovered precisely.


  class Foo(Serializable):
   ...

  foo = Foo()
  # do something with foo
  ...
  md = foo.dump()
  foo2 = Foo.recover(md)

  Then `foo` and `foo2` are indistinguishable using "public" methods of Foo.
  """

  # NOTE: cannot declare it as an abstract method due to Pytype limitations.
  @classmethod
  def recover(cls: Type['_S'], metadata: vz.Metadata) -> '_S':
    """Creates a new Serializable object from metadata.

    Args:
      metadata:

    Raises:
      DecodeError: Object could not be recovered from the metadata.
    """
    raise NotImplementedError('')

  @abc.abstractmethod
  def dump(self) -> vz.Metadata:
    pass
