"""Common classes shared between Study and Trial."""

import collections
from collections import abc
from typing import DefaultDict, Dict, overload
from typing import Iterable, List, Optional, Tuple, TypeVar, Union, Type
import attr

from google.protobuf import any_pb2
from google.protobuf.message import Message

M = TypeVar('M', bound=Message)
T = TypeVar('T')
MetadataValue = Union[str, any_pb2.Any, Message]

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

  Namespaces form a tree; a particular namespace can be thought of as a tuple of
  namespace components.

  You can create a Namespace from a string, with Namespace.decode(s), where
  the string is parsed into components, splitting at colons; decode('a:b') gives
  you a two-component namespace: ('a', 'b').
  Or, you can create that same Namespace from a tuple of strings/components
  e.g. by constructing Namespace(('a', 'b')).  In the tuple case, the strings
  are not parsed and colons are ordinary characters.

  TLDR: If you decode() a namespace from a string, then ":" is a
    reserved character, but when constructing from a tuple, there are no
    reserved characters.

  Decoding the string form:
  * Initial colons don't matter: Namespace.decode(':a') == Namespace('a');
    this is a single-component namespace.
  * Colons separate components:
    Namespace.decode('a:b') == Namespace(['a', 'b']).
    (This is a two-component namespace.)
  * Colons are encoded as r'\:':
    Namespace.decode('a\\:b') == Namespace(('a:b')).
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
      arg: a tuple representation of a namespace.
    """
    arg = tuple(arg)
    self.__attrs_init__(as_tuple=arg)

  _ns_repr_table = str.maketrans({':': r'\:'})

  @classmethod
  def decode(cls, s: str) -> 'Namespace':
    r"""Decode a string into a Namespace.

    For a Namespace x, Namespace.decode(x.encode()) == x.

    Args:
      s: A string where ':' separates namespace components, and colon is
        escaped as r'\:'.

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
        [':' + c.translate(self._ns_repr_table) for c in self._as_tuple])

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
    """Returns True if this namespace starts with prefix."""
    ns_prefix = Namespace(prefix)
    return self[:len(ns_prefix)] == ns_prefix


class _MetadataSingleNameSpace(Dict[str, MetadataValue]):
  """Stores metadata associated with one namespace."""
  pass


class Metadata(abc.MutableMapping):
  """Metadata class.

  This is the main interface for reading metadata from a Trial (writing metadata
  should typically be done via the MetadataUpdateContext class.)

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
    namespace operators: ns(s) which adds component(s) on to the namespace, and
    abs_ns() which specifies the entire namespace.

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

    # Use of abs_ns().
    mm = Metadata()
    mm.abs_ns(Namespace(('NewName',)))['bar'] = 'Bar'
    mm.abs_ns(Namespace(('NewName',)))  # returns 'Bar'

    # Multi-component namespaces.
    mm = Metadata()
    mm.ns('a').ns('b')['foo'] = 'AB-foo'
    mm.ns('a')['foo'] = 'A-foo'
    mm['foo']          # Throws a KeyError
    mm.ns('a')['foo']  # returns 'A-foo'
    mm.ns('a').ns('b')['foo']  # returns 'AB-foo'
    mm.abs_ns(Namespace(('a', 'b'))).get('foo')  # Returns 'ab-foo'
    mm.abs_ns(Namespace.decode('a:b')).get('foo')  # Returns 'ab-foo'

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
    WARNING: Because of this behavior, Metadata(mm) will quietly drop metadata
      from all but mm's current namespace.

    Be aware that type(v) is MetadataValue, which includes protos in addition to
    strings.

    To iterate over all the keys in all the namespaces use the namespaces()
    method.

    mm : Metadata
    for ns in mm.namespaces():
      for k, v in mm.abs_ns(ns).items():
        ...
  """

  def __init__(self, *args: Union[Dict[str, MetadataValue],
                                  Iterable[Tuple[str, MetadataValue]]],
               **kwargs: MetadataValue):
    """Construct; this follows dict(), and puts data in the root namespace.

    You can pass it a dict, or an object that yields (key, value)
    pairs, and those pairs will be put in the root namespace.

    Args:
      *args: A dict or an iterable the yields key-value pairs.
      **kwargs: key=value pairs to be added to the specified namespace.
    """
    self._stores: DefaultDict[
        Namespace, _MetadataSingleNameSpace] = collections.defaultdict(
            _MetadataSingleNameSpace)
    self._namespace = Namespace()
    self._store = self._stores[self._namespace]
    self._store.update(*args, **kwargs)

  def abs_ns(self, namespace: Iterable[str] = ()) -> 'Metadata':
    """Switches to a specified absolute namespace.

    All the Metadata object's data is shared between $self and the returned
    object, but the new Metadata object will have a different default
    namespace.

    Args:
      namespace: a list of Namespace components.  (Defaults to the root, empty
      Namespace.)

    Returns:
      A new Metadata object in the specified namespace; the new object shares
      data (except the namespace) with $self.
    """
    return self._copy_core(Namespace(namespace))

  def ns(self, component: str) -> 'Metadata':
    r"""Switches to a deeper namespace by appending a component.

    All the metadata is shared between $self and the returned value, but they
    have a different current namespace.

    Args:
      component: one component to be added to the current namespace.

    Returns:
      A new Metadata object in the specified namespace; the new object shares
      metadata (except the choice of namespace) with $self.
    """
    new_ns: Namespace = self._namespace + (component,)
    return self._copy_core(new_ns)

  def __repr__(self) -> str:
    itemlist: List[str] = []
    for namespace, store in self._stores.items():
      item_string = f'(namespace:{namespace}, items: {store})'
      itemlist.append(item_string)
    return 'Metadata({}, current_namespace={})'.format(', '.join(itemlist),
                                                       self._namespace.encode())

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

  # TODO: Rename to `abs_namespaces`
  def namespaces(self) -> Tuple[Namespace, ...]:
    """Get all namespaces for which there is at least one key.

    Returns:
      For all `ns` in `md.namespaces()`, `md.abs_ns(ns)` is not empty.
    """
    return tuple([ns for ns, store in self._stores.items() if store])

  # TODO: Rename to `namespaces`
  def subnamespaces(self) -> Tuple[Namespace, ...]:
    """Returns relative namespaces that are at or below the current namespace.

    For all `ns` in `md.subnamespaces()`, `md.abs_ns(md.current_ns() + ns)` is
    not empty.  E.g. if namespace 'foo:bar' is non-empty, and you're in
    namespace 'foo', then the result will contain namespace 'bar'.

    Returns:
      For namespaces that begin with the current namespace and are
      non-empty, this returns a namespace object that contains the relative
      path from the current namespace.
    """
    return tuple([
        Namespace(ns[len(self._namespace):])
        for ns, store in self._stores.items()
        if store and ns.startswith(self._namespace)
    ])

  def current_ns(self) -> Namespace:
    """Displays the object's current Namespace."""
    return self._namespace

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

    Returns:
      A copy of the object.
    """
    return self._copy_core(self._namespace)

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

  def update(self, *args: Union[Dict[str, MetadataValue],
                                Iterable[Tuple[str, MetadataValue]]],
             **kwargs: MetadataValue) -> None:
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
    other.abs_ns(Namespace.(('x', 'y', 'z'))['foo'] = 'bar'
    m = Metadata()
    m.ns('w').attach(other.ns('x'))
    then
    m.abs_ns(('w', 'y', 'z'))['foo'] will contain 'bar'.

    Args:
      other: a Metadata object to copy from.
    """
    for ns in other.subnamespaces():
      self._stores[self._namespace + ns].update(
          other.abs_ns(other.current_ns() + ns))
