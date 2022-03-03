"""Common classes shared between Study and Trial."""

import collections
from collections import abc
from typing import DefaultDict, Dict
from typing import Iterable, List, Optional, Tuple, TypeVar, Union, Type
import attr

from google.protobuf import any_pb2
from google.protobuf.message import Message

M = TypeVar('M', bound=Message)
T = TypeVar('T')
MetadataValue = Union[str, any_pb2.Any, Message]


def _parse(arg: str) -> Tuple[str, ...]:
  """Parses an encoded namespace string into a namespace tuple."""
  return tuple([s.replace('|c', ':').replace('||', '|')
                for s in arg.split(':')])


@attr.frozen(eq=True, order=True, hash=True, auto_attribs=True, init=False)
class Namespace:
  """A namespace for the Metadata class.

  Namespaces form a tree; a particular namespace is a string or a tuple of
  strings.

  Namespaces can be represented as a tuple of strings or a single encoded
  string; to convert, you can use string_form = repr(Namespace(tuple_form)), or
  tuple_form = tuple(Namespace(string_form)).
  E.g.  Namespace(('a', 'b')) == Namespace('a:b');
  Namespace((':',)) == Namespace('|c'); and
  Namespace(('|',)) == Namespace('||').  I.e. in the string form, colons are
  escaped, and components are separated by colons.

  Note that Namespace(':a') == Namespace('a'); initial colons don't matter in
  the string form.
  """

  _as_tuple: Tuple[str, ...] = attr.field(hash=True, eq=True, order=True)

  def __init__(self, arg: Union[str, Iterable[str]] = ''):
    """Generates a Namespace from a string or tuple.

    Args:
      arg: either a tuple or string representation of a namespace.
    """
    if isinstance(arg, str):  # string
      arg = arg.lstrip(':')
      if not arg:
        parsed: Tuple[str, ...] = ()
      else:
        parsed: Tuple[str, ...] = _parse(arg)
    else:
      parsed: Tuple[str, ...] = tuple(arg)
    self.__attrs_init__(parsed)

  _ns_repr_table = str.maketrans({':': '|c', '|': '||'})

  def __len__(self) -> int:
    """Number of components (elements of the tuple form) in the namespace."""
    return len(self._as_tuple)

  def __add__(self, other: Iterable[str]) -> 'Namespace':
    """Appends components onto the namespace."""
    return Namespace(self._as_tuple + tuple(other))

  def __iter__(self):
    return iter(self._as_tuple)

  def __str__(self) -> str:
    return ':'.join(self._as_tuple)

  def __repr__(self) -> str:
    """Given a Namespace x, Namespace(repr(x))==x."""
    return ':'.join([c.translate(self._ns_repr_table) for c in self._as_tuple])


class _MetadataSingleNameSpace(Dict[str, MetadataValue]):
  """Stores metadata associated with one namespace."""
  pass


_StoresType = DefaultDict[Namespace, _MetadataSingleNameSpace]


class Metadata(abc.MutableMapping):
  """Metadata class.

  This is the main interface for reading metadata from a Trial (writing metadata
  should typically be done via the MetadataUpdater class.)

  This behaves like a str->str dict, within a given namespace.
    mm = Metadata({'foo': 'Foo'})
    mm.get('foo')  # Returns 'Foo'
    mm['foo']      # Returns 'Foo'
    mm['bar'] = 'Bar'
    mm.update({'a': 'A'}, gleep='Gleep')

  1. Keys are namespaced. Each Metadata object only interacts with one
    Namespace, but a metadata object and its children share a
    common set of (namespace, key, value) triplets.

    Namespaces form a tree, and you can walk down the tree.  There are two
    namespace operators: ns(s) which adds a component to the namespace, and
    abs_ns() which replaces the entire namespace.

    A Metadata() object is always created at the root of the namespace tree,
    and the root is special (it's the only namespace that Vizier users can write
    or conveniently read).  Pythia algorithm developers should avoid the root
    namespace, unless they intend to pass data to/from Vizier users.

    mm = Metadata({'foo': 'foofoo'})
    mm.ns('NewName')['bar'] = 'Bar'
    mm['foo']               # Returns 'foofoo'
    mm['bar']               # Throws a KeyError
    mm.ns('NewName')['foo'] # Throws a KeyError
    mm.ns('NewName')['bar'] # Returns 'Bar'
    mm.ns('NewName').get('bar') # Returns 'Bar'
    # The above operations are identical if abs_ns() is used instead of ns().

    # Multi-component namespaces.
    mm = Metadata()
    mm.ns('a').ns('b')['foo'] = 'AB-foo'
    mm.ns('a')['foo'] = 'A-foo'
    mm['foo']          # Throws a KeyError
    mm.ns('a')['foo']  # returns 'A-foo'
    mm.ns('a').ns('b')['foo']  # returns 'AB-foo'
    mm.abs_ns(Namespace(('a', 'b'))).get('foo')  # Returns 'ab-foo'
    mm.abs_ns('a:b').get('foo')  # Returns 'ab-foo'

  2. Values can be protobufs. If `metadata['foo']` is an instance of `MyProto`
    proto message or `Any` proto that packs a `MyProto` message, then the proto
    can be recovered by calling:
      my_proto = metadata.get_proto('foo', cls=MyProto)
      isinstance(my_proto, MyProto) # Returns `True`

  3. An iteration over a Metadata object only shows you the data in the current
    namespace.  So,

    mm = Metadata({'foo': 'foofoo'})
    for k, v in mm.ns('gleep'):
      ...

    will not yield anything because there are no keys in the 'gleep' namespace.
    Be aware that type(v) is MetadataValue, not str.

    To iterate over all the keys in all the namespaces use the namespaces()
    method.

    mm : Metadata
    for ns in mm.namespaces():
      for k, v in mm.abs_ns(ns).items():
        ...
  """

  def __init__(
      self,
      *args: Union[Dict[str, MetadataValue],
                   Iterable[Tuple[str, MetadataValue]]],
      **kwargs: MetadataValue):
    """Construct; this follows dict(), and puts data in the root namespace.

    You can pass it a dict, or an object that yields (key, value)
    pairs, and those pairs will be put in the root namespace.

    Args:
      *args: A dict or an iterable the yields key-value pairs.
      **kwargs: key=value pairs to be added to the specified namespace.
    """
    self._stores: _StoresType = collections.defaultdict(
        _MetadataSingleNameSpace)
    self._namespace = Namespace()
    self._store = self._stores[self._namespace]
    self._store.update(*args, **kwargs)

  def abs_ns(self, namespace: Union[str, Namespace] = '') -> 'Metadata':
    """Switches to a specified absolute namespace.

    All the Metadata object's data is shared between $self and the returned
    object, but they have a different default namespaces.

    Args:
      namespace: a string is parsed into a Namespace object.  Note that
                 abs_ns() with no argument goes to the root namespace.

    Returns:
      A new Metadata object in the specified namespace; the new object shares
      data (except the namespace) with $self.
    """
    if isinstance(namespace, Namespace):
      ns = namespace
    else:
      ns = Namespace(namespace)
    return self._copy_core(ns)

  def ns(self, namespace: str) -> 'Metadata':
    """Switches to a deeper namespace by appending $namespace.

    This adds a single component to the namespace; len() increases by one;
    $namespace is not parsed.  All the metadata is shared between $self and the
    returned value, but they have a different current namespace.

    Args:
      namespace: A namespace component.

    Returns:
      A new Metadata object in the specified namespace; the new object shares
      data (except the namespace) with $self.
    """
    # pylint: disable='protected-access'
    new_ns: Namespace = self._namespace + (namespace,)
    # pylint: enable='protected-access'
    return self._copy_core(new_ns)

  def __repr__(self) -> str:
    itemlist: List[str] = []
    for namespace, store in self._stores.items():
      item_string = '(namespace: {}, items: {}'.format(repr(namespace),
                                                       repr(store))
      itemlist.append(item_string)
    items = ', '.join(itemlist)
    items += f', current_namespace = {repr(self._namespace)}'
    return f'Metadata({items})'

  def __str__(self) -> str:
    return 'namespace: {} items: {}'.format(str(self._namespace), self._store)

  def get_proto(self, key: str, *, cls: Type[M]) -> Optional[M]:
    """Deprecated.

    Use get() instead.

    Gets the metadata as type `cls`, or None if not possible.

    Args:
      key:
      cls: Pass in a proto ***class***, not a proto object.

    Returns:
      Proto message, if the value associated with the key exists and
      can be parsed into cls; None otherwise.
    """
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

  def get(self,
          key: str,
          default: Optional[T] = None,
          *,
          cls: Type[T] = str) -> Optional[T]:
    """Gets the metadata as type `cls`, or None if not possible.

    Given regular string values, this function behaves exactly like a
    regular string-to-string dict (within its namespace).
      metadata = common.Metadata({'key': 'value'})
      assert metadata.get('key') == 'value'
      assert metadata.get('badkey', 'badvalue') == 'badvalue'

    Example with numeric string values:
      metadata = common.Metadata({'float': '1.2', 'int': '60'})
      assert metadata.get('float', cls=float) == 1.2
      assert metadata.get('badkey', 0.2, cls=float) == 0.2
      assert metadata.get('int', cls=int) == 60
      assert metadata.get('badkey', 1, cls=int) == 1

    Example with `Duration` and `Any` proto values:
      duration = Duration(seconds=60)
      anyproto = Any()
      anyproto.Pack(duration)
      metadata = common.Metadata({'duration': duration, 'any': anyproto})
      assert metadata.get('duration', cls=Duration) == duration
      assert metadata.get('any', cls=Duration) == duration

    Args:
      key:
      default: Default value.
      cls: Desired type of the value.

    Returns:
      Default if the key does not exist. Otherwise, the matching value is
      parsed into type `cls`. For proto messages, it involves unpacking
      Any proto.
    """
    try:
      value = self._store[key]
    except KeyError:
      return default
    if isinstance(value, cls):
      # Starting from 3.10, pytype supports typeguard, which obsoletes
      # the need for the `pytype:disable` clause.
      return value  # pytype: disable=bad-return-type
    if isinstance(value, any_pb2.Any):
      # `value` is an Any proto potentially packing `cls`.
      message = cls()
      success = value.Unpack(message)
      return message if success else None
    return cls(value)

  def namespaces(self) -> Tuple[Namespace, ...]:
    """Returns all namespaces for which there is at least one key."""
    return tuple([ns for ns, store in self._stores.items() if store])

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
    return len(self._store)

  def __copy__(self) -> 'Metadata':
    """Shallow copy -- metadata continues to be shared.

    (Functionally, this is equivalent to a deep copy.)

    Returns:
      A copy of the object.
    """
    # pyline: disable='protected-access'
    return self._copy_core(self._namespace)
    # pyline: enable='protected-access'

  # END OF Abstract methods inherited from `MutableMapping` base class.

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
      *args: Union[Dict[str, MetadataValue],
                   Iterable[Tuple[str, MetadataValue]]],
      **kwargs: MetadataValue) -> None:
    self._store.update(*args, **kwargs)
