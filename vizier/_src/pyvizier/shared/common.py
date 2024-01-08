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

"""Common classes shared between Study and Trial."""

import collections
from collections import abc
from typing import DefaultDict, Dict, overload, Iterator
from typing import Iterable, List, Optional, Tuple, TypeVar, Union, Type

from absl import logging
import attr

from google.protobuf import any_pb2
from google.protobuf.message import Message

_M = TypeVar('_M', bound=Message)
T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
MetadataValue = Union[str, any_pb2.Any, Message]
_V = TypeVar('_V', bound=MetadataValue)

# Namespace Encoding.
#
# By definition, ∀ ns ∈ Namespace, Namespace.decode(ns.encode()) == ns.
# The tricky part of that definition is handling namespaces with components
# that are empty strings.  Notably, we want to make sure that
# Namespace(()).encode() != Namespace(('',)).encode().
# So, we set up the mapping:
# Namespace(()).encode() -> ''
# Namespace((s,)).encode() -> ':s'
# Namespace((s, s)).encode() -> ':s:s',
# et cetera, and note that every tuple gets a unique encoding, even if $s is the
# empty string.  (As long as we escape colons properly.)
#
# So, ns.encode() is a bijection, therefore it has an inverse which we call
# Namespace.decode(s).


def _parse(arg: str) -> Tuple[str, ...]:
  """Parses an encoded namespace string into a namespace tuple."""
  # The tricky part here is that arg.split('') has a length of 1, so it can't
  # generate a zero-length tuple; we handle that corner case manually.
  if not arg:
    return ()
  # And, then, once we've handled the case of _parse(''), we note that all the
  # other encoded strings begin with a colon.  It thus contains no information
  # and we can remove it.
  # TODO: Once we're on Python 3.9, use: arg = arg.removeprefix(':')
  if arg.startswith(':'):
    arg = arg[1:]
  # The rest of the algorithm is that we split on all colons, both
  # escaped and unescaped.  Then, we walk through the list of fragments and
  # join back together the colons that were preceeded by an escape character,
  # dropping the escape character as we go.
  fragments = arg.split(':')
  output = []
  join = False
  for frag in fragments:
    if join and frag and frag[-1] == '\\':
      output[-1] += ':' + frag[:-1]
      join = True
    elif join:  # Doesn't end in an escape character.
      output[-1] += ':' + frag
      join = False
    elif frag and frag[-1] == '\\':  # Don't join to previous.
      output.append(frag[:-1])
      join = True
    else:  # Don't join to previous and doesn't end in an escape.
      output.append(frag)
      join = False
  return tuple(output)


@attr.frozen(eq=True, order=True, hash=True, auto_attribs=True, init=False)
class Namespace(abc.Sequence):
  r"""A namespace for the Metadata class.

  Namespaces represent a tree of Metadata; each namespace object can be thought
  of as a tuple of components obtained by walking the tree from the root.
  This makes it easy to give each part of your algorithm its own namespace,
  to avoid name collisions.  E.g. if your algorithm A uses sub-algorithms B and
  C, you might have namespaces ":A", ":A:B", and ":A:C".

  NOTE: The empty namespace is writeable by users via a RPC in Vizier's
    user-facing API; other namespaces are writeable only by Pythia algorithms.
    (Users can read all namespaces.) So, to minimize collisions, please avoid
    the empty namespace unless your algorithm needs to read user data.

  You can create a Namespace from a tuple of strings, e.g.
  Namespace(('a', 'b')).  Or, you can create a Namespace from a single string
  with Namespace.decode(s); this parses the string into components, splitting at
  colons.  For instance Namespace.decode(':a:b') gives you a two-component
  namespace, equivalent to Namespace(('a', 'b')).
  (Note that in the tuple case, the strings are not parsed and colons are
  treated as ordinary characters.)

  TLDR: If you decode() a namespace from a string, then ":" is a
    reserved character, but when constructing from a tuple, there are no
    reserved characters.

  Decoding the string form:
  * Initial colons don't matter: Namespace.decode(':a') == Namespace('a');
    this is a single-component namespace.
  * Colons separate components:
    Namespace.decode('a:b') == Namespace(('a', 'b')).
    (This is a two-component namespace.)
  * Colons are encoded as r'\:':
    Namespace.decode('a\\:b') == Namespace(('a:b',)).
    (This is a single-component namespace.)

  Conversions: For a Namespace x,
  * Namespace.decode(x.encode()) == x; here, x.encode() will be a string with
    colons separating the components.
  * Namespaces act as a Sequence[str], so Namespace(tuple(x)) == x and
    Namespace(x) == x.
  """

  _as_tuple: Tuple[str, ...] = attr.field(hash=True, eq=True, order=True)

  def __init__(self, arg: Iterable[str] = ()):
    """Generates a Namespace from its component strings.

    Args:
      arg: typically, a tuple of strings.
    """
    arg = tuple(arg)
    self.__attrs_init__(as_tuple=arg)

  _ns_repr_table = str.maketrans({':': r'\:'})

  @classmethod
  def decode(cls, s: str) -> 'Namespace':
    r"""Decode a string into a Namespace.

    For a Namespace x, Namespace.decode(x.encode()) == x.

    Args:
      s: A string where ':' separates namespace components, and colon is escaped
        as r'\:'.

    Returns:
      A namespace.
    """
    return Namespace(_parse(s))

  def encode(self) -> str:
    """Encodes a Namespace into a string.

    Given a Namespace x, Namespace.decode(x.encode()) == x.

    Returns:
      Colons are escaped, then Namespace components are joined by colons.
    """
    return ''.join(
        [':' + c.translate(self._ns_repr_table) for c in self._as_tuple]
    )

  def __len__(self) -> int:
    """Number of components (elements of the tuple form)."""
    return len(self._as_tuple)

  def __add__(self, other: Iterable[str]) -> 'Namespace':
    """Appends components onto the namespace."""
    return Namespace(self._as_tuple + tuple(other))

  @overload
  def __getitem__(self, key: int) -> str:
    ...

  @overload
  def __getitem__(self, key: slice) -> 'Namespace':
    ...

  def __getitem__(self, key):
    """Retrieves item by the specified key."""
    if isinstance(key, int):
      return self._as_tuple[key]
    return Namespace(self._as_tuple[key])

  def __str__(self) -> str:
    """Shows the namespace, fully escaped."""
    return self.encode()

  def __repr__(self) -> str:
    """Shows the namespace, fully escaped."""
    return f'Namespace({self.encode()})'

  def startswith(self, prefix: Iterable[str]) -> bool:
    """Returns True if this namespace starts with $prefix.

    So, if the current namespace is "a:b:c", then startswith() will return True
    when called with prefix=(), ('a',), ('a', 'b'), and ('a', 'b', 'c');
    otherwise False.

    Args:
      prefix: namespace components or a Namespace object.

    Returns:
    """
    ns_prefix = Namespace(prefix)
    return self[: len(ns_prefix)] == ns_prefix


class _MetadataSingleNameSpace(Dict[str, MetadataValue]):
  """Stores metadata associated with one namespace."""

  pass


class Metadata(abc.MutableMapping):
  """Metadata class: a key-value dict-like mapping, with namespaces.

  This is the main interface for reading metadata from a Trial or StudyConfig,
  or adding metadata to a Trial/StudyConfig.  Loosely speaking, within each
  namespace, Metadata acts like a dictionary mapping from a string key to a
  string or protobuf. (See more about namespaces below.)

  Metadata can be initialized from a dictionary:
    mm = Metadata({'foo': 'Foo'})
  And items can be retrieved with:
    mm.get('foo')  # Returns 'Foo'
    mm['foo']      # Returns 'Foo'

  More items can be added with:
    mm = Metadata({'foo': 'Foo'})
    mm['bar'] = 'Bar'  # Add a single item
    mm.update({'a': 'A'}, gleep='Gleep')  # Add two items

  By default, items are added to the root/empty namespace.
  Vizier users can only add metadata to the empty namespace (the Vizier service
  will reject attempts by users to add metadata elsewhere); Pythia algorithms
  can add metadata to any namespace, but should normally work in a single unique
  namespace, and should avoid the root namespace, unless they intend to
  pass data to/from Vizier users.

  1. Keys are namespaced. Each Metadata object only interacts with one
    Namespace.

    Namespaces form a tree, and you can walk down the tree.

    NOTE ns(s: str) takes one step down the namespace tree. For nearly all
    practical purposes, ignore abs_ns() which is only used for conversions
    to and from protobufs.

    mm = Metadata({'foo': 'foofoo'})
    # $mm is created with its current namespace equal to the root/empty
    # namespace.
    mm.ns('NewName')['bar'] = 'Bar'
    # We've added an item in the ":NewName" namespace, but $mm's current
    # namespace is unchanged.

  2. Values can be protobufs. If `metadata['foo']` is an instance of `MyProto`
    proto message or an `Any` proto that packs a `MyProto` message, then the
    proto can be recovered by calling:

    my_proto = metadata.get('foo', cls=MyProto)
    isinstance(my_proto, MyProto) # Returns `True`

    NOTE that the bracket operator doesn't work well for protobufs:
      metadata['foo'] will return an `Any` protobuf instead of a MyProto.
      For protos, you may wish to use
      `metadata.get_or_error('foo', cls=MyProto)` instead of the bracket form.

  3. An iteration over a Metadata object only shows you the data in the current
    namespace.  So,

    mm = Metadata({'foo': 'foofoo'})
    for k, v in mm.ns('gleep'):
      ...

    will not yield anything because there are no keys in the 'gleep' namespace.
    WARNING: Because of this behavior, if you iterate over Metadata(mm), you
      will quietly drop metadata from all but mm's current namespace.
    NOTE also that the type of $v is MetadataValue, which can carry strings and
      protos; you may want to use mm.get_or_error(key, cls=proto_class) to
      unpack the contained proto.

  4. To iterate over all the keys in all the namespaces use

    mm = Metadata()
    mm.ns('gleep')['x'] = 'X'
    for ns, k, v in mm.all_items():
      # iteration will include ('gleep', 'x', 'X')
      # Be aware that type(v) is MetadataValue, which can carry either strings
      # or protos.
  """

  def __init__(
      self,
      *args: Union[
          Dict[str, MetadataValue], Iterable[Tuple[str, MetadataValue]]
      ],
      **kwargs: MetadataValue,
  ):
    """Construct; this follows dict(), and puts data in the root namespace.

    You can pass it a dict, or an object that yields (key, value)
    pairs, and those pairs will be put in the root namespace.

    Args:
      *args: A dict or an iterable the yields key-value pairs.
      **kwargs: key=value pairs to be added to the specified namespace.
    """
    self._stores: DefaultDict[Namespace, _MetadataSingleNameSpace] = (
        collections.defaultdict(_MetadataSingleNameSpace)
    )
    self._namespace = Namespace()
    self._store = self._stores[self._namespace]
    self._store.update(*args, **kwargs)

  def ns(self, component: str) -> 'Metadata':
    r"""Switches to a deeper namespace by appending one component.

    The entire tree of metadata is shared between $self and the returned value,
    but the returned value will have a deeper current namespace.  ($self is not
    modified.)

    Args:
      component: one component to be appended to the current namespace.

    Returns:
      A new Metadata object in the specified namespace; the new object shares
      metadata with $self.
    """
    new_ns: Namespace = self._namespace + (component,)
    return self._copy_core(new_ns)

  def __repr__(self) -> str:
    """Prints items in all namespaces."""
    itemlist: List[str] = []
    for namespace, store in self._stores.items():
      item_string = f'(namespace:{namespace}, items: {store})'
      itemlist.append(item_string)
    return 'Metadata({}, current_namespace={})'.format(
        ', '.join(itemlist), self._namespace.encode()
    )

  def __str__(self) -> str:
    """Prints items in the current namespace."""
    return 'namespace: {} items: {}'.format(str(self._namespace), self._store)

  def get_proto(self, key: str, *, cls: Type[_M]) -> Optional[_M]:
    """Deprecated: use get() instead."""
    logging.warning(
        'Metadata.get_proto() is deprecated, prefer Metadata.get().'
    )
    value = self._store.get(key, None)
    if value is None:
      return None

    if isinstance(value, cls):
      # Starting from 3.10, pytype supports typeguard, which obsoletes
      # the need for the `pytype:disable` clause.
      return value  # pytype: disable=bad-return-type
    if isinstance(value, any_pb2.Any):
      # `value` is an Any proto potentially packing `cls`.
      message = cls()
      success = value.Unpack(message)
      return message if success else None
    return None

  def get_or_error(self, key: str, *, cls: Type[T] = str) -> T:
    """Gets the metadata as type `cls`, or raises a KeyError.

    This acts like the square bracket operator, except that
    it lets you specify a class; it gets the metadata from the current
    namespace.

    Examples with string metadata:
      metadata = common.Metadata({'key': 'value'})
      assert metadata.get_or_error('key') == 'value'
      metadata.get_or_error('badkey')  # raises KeyError

    Examples with numeric values:
      metadata = common.Metadata({'float': '1.2', 'int': '60'})
      assert metadata.get_or_error('int', cls=int) == 60
      assert metadata.get_or_error('float', cls=float) == 1.2
      metadata.get_or_error('badkey', cls=float)      # raises KeyError

    Example with `Duration` and `Any` proto values:
      duration = Duration(seconds=60)
      anyproto = Any()
      anyproto.Pack(duration)
      metadata = common.Metadata({'duration': duration, 'any': anyproto})
      assert metadata.get_or_error('duration', cls=Duration) == duration
      assert metadata.get_or_error('any', cls=Duration).seconds == 60

    Args:
      key:
      cls: Desired type of the value.

    Returns:
      The matching metadata value is parsed into type `cls`. For proto messages,
      it involves unpacking an Any proto.

    Raises:
      KeyError if the metadata item is not present.
      TypeError or other errors if the string can't be converted to $cls.
    """
    value = self._store[key]
    if isinstance(value, cls):
      # Starting from 3.10, pytype supports typeguard, which obsoletes
      # the need for the `pytype:disable` clause.
      return value  # pytype: disable=bad-return-type
    elif isinstance(value, any_pb2.Any):
      # `value` is an Any proto potentially packing `cls`.
      message = cls()
      if not value.Unpack(message):
        logging.warning(
            'Cannot unpack message to %s: %s', cls, str(value)[:100]
        )
        raise TypeError('Cannot unpack to %s' % cls)
      return message
    else:
      return cls(value)

  def get(
      self, key: str, default: T1 = None, *, cls: Type[T2] = str
  ) -> Union[T1, T2]:
    """Gets the metadata as type `cls`, or $default if not present.

    This returns $default if the specified metadata item is not found.
    Note that there's always a default value, and the $default defaults to None.
    This gets the data from the current namespace.

    For string values, this function behaves exactly like a
    regular string-to-string dict (within its namespace).
      metadata = common.Metadata({'key': 'value'})
      metadata.get('key')  # returns 'value'
      metadata.get('badkey')  # returns None
      assert metadata.get('badkey', 'badvalue') == 'badvalue'

    Examples with numeric values:
      metadata = common.Metadata({'float': '1.2', 'int': '60'})
      value = metadata.get('int', cls=int)
      if value is not None:
        assert value == 60
      #
      metadata.get('float', cls=float)       # returns 1.2
      metadata.get('badkey', cls=float)      # returns None
      metadata.get('int', cls=int)           # returns 60
      assert metadata.get('float', 0.0, cls=float) == 1.2
      assert metadata.get('badkey', 1, cls=int) == 1
      assert metadata.get('badkey', 0.2, cls=float) == 0.2

    Example with `Duration` and `Any` proto values:
      duration = Duration(seconds=60)
      anyproto = Any()
      anyproto.Pack(duration)
      metadata = common.Metadata({'duration': duration, 'any': anyproto})
      duration_out =  metadata.get('duration', cls=Duration)
      if duration_out is not None:
        assert duration_out == duration
      any_out =  metadata.get('any', cls=Duration)
      if any_out is not None:
        assert any_out == duration

    Args:
      key:
      default: Default value.
      cls: Desired type of the value.

    Returns:
      $default if the key does not exist. Otherwise, the matching value is
      parsed into type `cls`. For proto messages, it involves unpacking an
      Any proto.

    Raises:
      TypeError or other errors if the string can't be converted to $cls.
    """
    try:
      return self.get_or_error(key, cls=cls)
    except KeyError:
      return default

  # TODO: Rename to `abs_namespaces`
  def namespaces(self) -> List[Namespace]:
    """List all namespaces for which there is at least one key."""
    return [ns for ns, store in self._stores.items() if store]

  # TODO: Rename to `namespaces`
  def subnamespaces(self) -> Tuple[Namespace, ...]:
    """Returns relative namespaces that are at or below the current namespace.

    For all `ns` in the returned value, `self.abs_ns(md.current_ns() + ns)` is
    not empty.
    # Examples:
    md = Metadata()
    md.ns('foo').ns('bar')['A'] = 'b'
    md.subnamespaces() == (Namespace(['foo', 'bar']),)
    md.ns('foo').subnamespaces() == (Namespace(['bar']),)

    Returns:
      For all namespaces that begin with the current namespace and are
      non-empty, this returns a namespace object that contains the relative
      path from the current namespace.
    """
    return tuple(
        [
            Namespace(ns[len(self._namespace) :])
            for ns, store in self._stores.items()
            if store and ns.startswith(self._namespace)
        ]
    )

  def current_ns(self) -> Namespace:
    """Displays the object's current Namespace."""
    return self._namespace

  def all_items(self) -> Iterator[Tuple[Namespace, str, MetadataValue]]:
    """Yields an iterator that walks through all metadata items.

    This iterates through all the metadata items in all namespaces, vs.
    __iter__() which just iterates through all the items in the current
    namespace.

    Yields:
      Tuple of (namespace, key, value).
    """
    for ns in self.namespaces():
      for k, v in self.abs_ns(ns).items():
        yield (ns, k, v)

  def items_by_cls(self, *, cls: Type[_V]) -> Iterator[Tuple[str, _V]]:
    """Yields an iterator over items whose type=$cls in the current namespace.

    This iterates through the metadata items in the current namespace, like
    __iter__(), except that it only returns items of the specified type.

    Args:
      cls: What type of objects to filter for?

    Yields:
      Tuple of (key, value).
    """
    for k_v in self.items():
      if isinstance(k_v[1], cls):
        yield k_v

  # START OF abstract methods inherited from `MutableMapping` base class.
  def __getitem__(self, key: str) -> MetadataValue:
    return self._store.__getitem__(key)

  def __setitem__(self, key: str, value: MetadataValue):
    self._store[key] = value

  def __delitem__(self, key: str):
    del self._store[key]

  def __iter__(self):
    return iter(self._store)

  def __len__(self):
    """The number of elements in the current namespace."""
    return len(self._store)

  def __bool__(self):
    """True if this instance contains any metadata in _any_ namespace."""
    for s in self._stores.values():
      if s:
        return True
    return False

  def __copy__(self) -> 'Metadata':
    """Shallow copy -- metadata continues to be shared.

    Returns:
      A copy of the object.
    """
    return self._copy_core(self._namespace)

  # END OF Abstract methods inherited from `MutableMapping` base class.

  def abs_ns(self, namespace: Iterable[str] = ()) -> 'Metadata':
    """Returns a metadata object set to the specified absolute namespace.

    NOTE Prefer using ns() instead in most cases.

    abs_ns() jumps to the root namespace and
    abs_ns(ns) jumps to the specified Namespace.

    (NOTE: ns() and abs_ns() take different argument types!)
    (NOTE: Neither ns() nor abs_ns() modify the Metadata object they are called
     on: they return a shallow copy that shares all metadata items, but
     which displays a different namespace.)

    # Use of abs_ns().
    mm.abs_ns(['NewName'])  # returns 'Bar'
    mmx = mm.ns('x')
    mmx.abs_ns(['NewName'])  # returns 'Bar2'
    mmx.abs_ns().get('foo')  # returns 'foofoo'

    # Multi-component namespaces.
    mm = Metadata()
    mm.ns('a').ns('b')['foo'] = 'AB-foo'
    mm.ns('a')['foo'] = 'A-foo'
    mm['foo']          # Throws a KeyError
    mm.ns('a')['foo']  # returns 'A-foo'
    mm.ns('a').ns('b')['foo']  # returns 'AB-foo'
    # abs_ns() can be also used:
    mm.abs_ns(['a', 'b']).get('foo')  # Returns 'ab-foo'
    mm.abs_ns(Namespace.decode('a:b')).get('foo')  # Returns 'ab-foo'

    All the Metadata object's data is shared between $self and the returned
    object, but the new Metadata object will have a different current
    namespace.  (Note that $self is not modified, and the current namespace of
    $self doesn't matter.)

    NOTE: $namespace can be a Namespace object, because you can iterate over
      a Namespace to get strings.

    Args:
      namespace: a list of Namespace components.  (Defaults to the root, empty
        Namespace.)

    Returns:
      A new Metadata object that shares data with $self, but the current
      namespace is one level deeper.
    """
    if isinstance(namespace, str):
      raise ValueError(
          'Passing str to abs_ns() is rarely intended and therefore '
          'considered an error. Carefully read the class doc and prefer '
          'using ns(). If you do decide abs_ns() is the right method, '
          'expclitily pass abs_ns([namespace]).'
      )
    return self._copy_core(Namespace(namespace))

  def _copy_core(self, ns: Namespace) -> 'Metadata':
    """Shallow copy: metadata is shared, default namespace changes.

    Args:
      ns: the namespace to use for the new object.

    Returns:
      A copy of the object.
    """
    md = Metadata()
    md._namespace = ns  # pylint: disable='protected-access'
    md._stores = self._stores  # pylint: disable='protected-access'
    md._store = md._stores[md._namespace]  # pylint: disable='protected-access'
    return md

  def update(
      self,
      *args: Union[
          Dict[str, MetadataValue], Iterable[Tuple[str, MetadataValue]]
      ],
      **kwargs: MetadataValue,
  ) -> None:
    self._store.update(*args, **kwargs)

  def attach(self, other: 'Metadata') -> None:
    """Attach the $other metadata as a descendent of this metadata.

    More precisely, it takes the part of `other`'s namespace that is at or
    below `other`'s current namespace, and attaches it to `self`'s current
    namespace.
    * Tree structure is preserved and nothing is flattened.
    * Attached data overwrites existing data, item-by-item, not
      namepace-by-namespace.

    So, if we have
    other = Metadata()
    other.abs_ns(('x', 'y', 'z'))['foo'] = 'bar'
    m = Metadata()
    m.ns('w').attach(other.ns('x'))
    then
    m.abs_ns(('w', 'y', 'z'))['foo'] will contain 'bar'.

    Args:
      other: a Metadata object to copy from.
    """
    for ns in other.subnamespaces():
      self._stores[self._namespace + ns].update(
          other.abs_ns(other.current_ns() + ns)
      )
