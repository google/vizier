"""Tests for vizier.pyvizier.shared.common."""

import copy
from vizier._src.pyvizier.shared import common
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from absl.testing import absltest


class MetadataGetClsTest(absltest.TestCase):

  def test_get_proto(self):
    duration = duration_pb2.Duration(seconds=60)
    anyproto = any_pb2.Any()
    anyproto.Pack(duration)
    metadata = common.Metadata(duration=duration, any=anyproto)

    self.assertEqual(
        metadata.get_proto('duration', cls=duration_pb2.Duration), duration)
    self.assertEqual(
        metadata.get_proto('any', cls=duration_pb2.Duration), duration)
    self.assertEqual(
        metadata.get('duration', cls=duration_pb2.Duration), duration)
    self.assertEqual(metadata.get('any', cls=duration_pb2.Duration), duration)

  def test_get_int(self):
    metadata = common.Metadata({'string': '30', 'int': '60'})
    self.assertEqual(metadata.get('string', cls=int), 30)
    self.assertEqual(metadata.get('int', cls=int), 60)
    self.assertEqual(metadata.get('badkey', 1, cls=int), 1)


class MetadataNamespaceTest(absltest.TestCase):

  def test_basic(self):
    ns0 = common.Namespace()
    self.assertEmpty(ns0)
    self.assertEqual(str(ns0), '')
    self.assertEqual(ns0.encode(), '')
    self.assertEqual(ns0, common.Namespace.decode(''))
    n1t = common.Namespace(('aerer',))
    self.assertLen(n1t, 1)
    n1 = common.Namespace.decode('a78')
    self.assertLen(n1, 1)
    self.assertEqual(str(n1), ':a78')
    n2 = common.Namespace(('a78', 'bfe'))
    self.assertLen(n2, 2)
    n2s1 = common.Namespace.decode('a78:bfe')
    self.assertLen(n2s1, 2)
    self.assertEqual(n2.encode(), n2s1.encode())
    n2s2 = common.Namespace.decode(':a78:bfe')
    self.assertLen(n2s2, 2)
    self.assertEqual(n2.encode(), n2s2.encode())
    self.assertEqual(n2, n2s2)
    self.assertEqual(n2s1, n2s2)
    ns = common.Namespace(('a', 'b'))
    self.assertLen(ns, 2)
    self.assertEqual(tuple(ns), ('a', 'b'))
    self.assertEqual(str(ns), ':a:b')
    self.assertEqual(ns.encode(), ':a:b')

  def test_escape(self):
    s1 = 'a\\:A'
    ns1 = common.Namespace.decode(s1)
    self.assertLen(ns1, 1)
    self.assertEqual(str(ns1), ':a\\:A')
    self.assertEqual(ns1.encode(), ':' + s1)
    self.assertEqual(common.Namespace.decode(ns1.encode()), ns1)
    #
    s2 = 'b:B'
    ns2 = common.Namespace.decode(s2)
    self.assertLen(ns2, 2)
    self.assertEqual(str(ns2), ':' + s2)
    self.assertEqual(ns2.encode(), ':' + s2)
    self.assertEqual(common.Namespace.decode(ns2.encode()), ns2)
    #
    s1e1 = ':b\\B'
    ns1e1 = common.Namespace.decode(s1e1)
    self.assertLen(ns1e1, 1)
    self.assertEqual(ns1e1.encode(), s1e1)
    self.assertEqual(common.Namespace.decode(ns1e1.encode()), ns1e1)
    ns1e2 = common.Namespace((s1e1.lstrip(':'),))
    self.assertLen(ns1e2, 1)
    self.assertEqual(ns1e2.encode(), s1e1)
    self.assertEqual(ns1e2, ns1e1)
    self.assertEqual(common.Namespace.decode(ns1e2.encode()), ns1e2)
    #
    s1c = r':b\:B'
    ns1c = common.Namespace.decode(s1c)
    self.assertLen(ns1c, 1)
    # Initial colon is harmlessly removed.
    self.assertEqual(ns1c.encode(), s1c)
    self.assertEqual(common.Namespace.decode(ns1c.encode()), ns1c)
    self.assertEqual(common.Namespace(('b:B',)), ns1c)


class MetadataTest(absltest.TestCase):

  def create_test_metadata(self):
    md = common.Metadata({'bar': 'bar_v'}, foo='foo_v')
    md.ns('Name').update(foo='Name_foo_v', baz='Name_baz_v')
    return md

  def test_empty_namespaces(self):
    md = common.Metadata()
    self.assertEmpty(list(md.namespaces()))
    md = common.Metadata().ns('ns')
    self.assertEmpty(list(md.namespaces()))

  def test_nonempty_namespaces(self):
    mm = self.create_test_metadata()
    self.assertLen(mm.namespaces(), 2)

  def test_getters_are_consistent_when_item_is_in_dict(self):
    mm = self.create_test_metadata()
    self.assertEqual(mm['foo'], 'foo_v')
    self.assertEqual(mm.get('foo'), 'foo_v')

  def test_getters_are_consistent_when_item_is_not_in_dict(self):
    mm = self.create_test_metadata()
    self.assertIsNone(mm.get('baz'))
    with self.assertRaises(KeyError):
      _ = mm['baz']

  def test_separator_is_not_allowed_as_keys_after_init(self):
    mm = self.create_test_metadata()
    with self.assertRaises(KeyError):
      _ = mm['Name_foo']

  def test_namespace_works_as_intended(self):
    mm = self.create_test_metadata()
    self.assertEqual(mm.ns('Name')['foo'], 'Name_foo_v')
    self.assertIsNone(mm.ns('Name').get('bar'))

    mm_name = mm.ns('Name')
    self.assertEqual(mm_name['foo'], 'Name_foo_v')
    self.assertIsNone(mm_name.get('bar'))
    self.assertEqual(mm.ns('Name')['foo'], 'Name_foo_v')

  def test_create_new_namespace(self):
    # Calling ns() with an unexisting namespace should work fine.
    mm = self.create_test_metadata()
    mm.ns('NewName')['foo'] = 'NewName_foo_v'
    self.assertEqual(mm.ns('NewName')['foo'], 'NewName_foo_v')
    self.assertIsNone(mm.ns('NewName').get('bar'))

  def test_changing_namespace_copies_reference(self):
    mm = self.create_test_metadata()
    # Calling ns() copies by reference so any changes to the returned Metadata
    # object is reflected in the original object.
    mm_in_namespace = mm.ns('Name')
    mm_in_namespace['foofoo'] = 'Name_foofoo_v'
    self.assertEqual(mm.ns('Name')['foofoo'], 'Name_foofoo_v')

  def test_iterators(self):
    mm = self.create_test_metadata()
    self.assertSequenceEqual(list(mm.keys()), ['bar', 'foo'])
    self.assertSequenceEqual(
        list(mm.ns('Name').values()), ['Name_foo_v', 'Name_baz_v'])
    self.assertLen(list(mm.items()), 2)

  def test_repr_str(self):
    mm = self.create_test_metadata()
    self.assertNotEmpty(str(mm), '')
    self.assertNotEmpty(repr(mm), repr(''))

  def test_update(self):
    md = common.Metadata(foo='foo_v')
    md.ns('Name').update(foo='Name_foo_v', baz='Name_baz_v')

    md2 = common.Metadata()
    md2.ns('Name').update(foo='Name_foo_v2', bar='Name_bar_v2')

    md.ns('Name').update(md2.ns('Name'))

    self.assertLen(md.ns('Name'), 3)
    self.assertIn('bar', md.ns('Name'))

  def test_copy(self):
    # There's no useful distinction to be made between copy.copy() and
    # copy.deepcopy().
    mm = common.Metadata().ns('ns1')
    mm.update(foo='bar')
    mm_copy = copy.copy(mm)
    mm_deepcopy = copy.deepcopy(mm)
    # Check that copies match.
    self.assertEqual(mm['foo'], 'bar')
    self.assertEqual(mm_copy['foo'], 'bar')
    self.assertEqual(mm_deepcopy['foo'], 'bar')
    self.assertEqual(mm_deepcopy.namespaces(), mm.namespaces())
    self.assertEqual(mm_copy.namespaces(), mm.namespaces())
    # Check that the  deep copy is disconnected.
    mm_deepcopy['nerf'] = 'gleep'
    with self.assertRaises(KeyError):
      mm['nerf']  # pylint: disable=pointless-statement
    with self.assertRaises(KeyError):
      mm_copy['nerf']  # pylint: disable=pointless-statement
    # Check that the shallow copy shares the metadata store with the original.
    mm_copy['blip'] = 'tonk'
    self.assertEqual(mm['blip'], mm_copy['blip'])
    # ... but no sharing with the deep copy.
    with self.assertRaises(KeyError):
      mm_deepcopy['blip']  # pylint: disable=pointless-statement
    # Here's a test for a specific bug, where Metadata._store is improperly
    # disconnected from Metadata._stores.
    mx = common.Metadata()
    copy.copy(mx).ns('A')['a'] = 'Aa'
    self.assertEqual(mx.ns('A')['a'], 'Aa')

  def test_construction(self):
    # Test with iterables.
    m0i = common.Namespace([])
    self.assertEmpty(m0i)
    m0d = common.Namespace.decode('')
    self.assertEmpty(m0d)
    self.assertEqual(m0d, m0i)
    m1i = common.Namespace(['abc'])
    self.assertLen(m1i, 1)
    self.assertEqual(m1i, common.Namespace(tuple(m1i)))
    self.assertEqual(m1i, common.Namespace.decode(m1i.encode()))
    m2i = common.Namespace(['abc', 'def'])
    self.assertLen(m2i, 2)
    self.assertEqual(m2i, common.Namespace(tuple(m2i)))
    self.assertEqual(m2i, common.Namespace.decode(m2i.encode()))
    m3i = common.Namespace(['abc', 'de:f'])
    self.assertLen(m3i, 2)
    self.assertEqual(m3i, common.Namespace(tuple(m3i)))
    self.assertEqual(m3i, common.Namespace.decode(m3i.encode()))
    # Test with strings.
    m1sc = common.Namespace.decode(':abc')
    self.assertLen(m1sc, 1)
    self.assertEqual(m1sc, common.Namespace(tuple(m1sc)))
    self.assertEqual(m1sc, common.Namespace.decode(m1sc.encode()))
    m1s = common.Namespace.decode('abc')
    self.assertLen(m1s, 1)
    self.assertEqual(m1s, common.Namespace(tuple(m1s)))
    self.assertEqual(m1s, common.Namespace.decode(m1s.encode()))
    m2s = common.Namespace.decode('abc:def')
    self.assertLen(m2s, 2)
    self.assertEqual(m2s, common.Namespace(tuple(m2s)))
    self.assertEqual(m2s, common.Namespace.decode(m2s.encode()))
    m3s = common.Namespace.decode('abc:de\\f')
    self.assertLen(m3s, 2)
    self.assertEqual(m3s, common.Namespace(tuple(m3s)))
    self.assertEqual(m3s, common.Namespace.decode(m3s.encode()))

  def test_startswith(self):
    m1 = common.Namespace(['aa', 'bb'])
    self.assertTrue(m1.startswith(common.Namespace(['aa'])))
    self.assertTrue(m1.startswith(common.Namespace(['aa', 'bb'])))
    self.assertTrue(m1.startswith(m1))
    self.assertTrue(m1.startswith(common.Namespace(tuple(m1))))
    self.assertFalse(m1.startswith(common.Namespace(['bb'])))
    self.assertFalse(m1.startswith(common.Namespace(['aa', 'bb', 'cc'])))
    self.assertFalse(m1.startswith(common.Namespace(['bb', 'bb'])))
    self.assertFalse(m1.startswith(common.Namespace(['aa', 'aa'])))

  def test_subnamespace(self):
    mm = common.Metadata()
    mm.ns('ns1')['foo'] = 'bar'
    mm.ns('ns2')['foo'] = 'bar'
    mm.ns('ns1').ns('ns11')['foo'] = 'bar'
    mm.ns('ns1').ns('ns:11')['gleep'] = 'nerf'

    self.assertSequenceEqual(mm.subnamespaces(), [
        common.Namespace(['ns1']),
        common.Namespace(['ns2']),
        common.Namespace(['ns1', 'ns11']),
        common.Namespace(['ns1', 'ns:11']),
    ])
    self.assertSequenceEqual(
        mm.ns('ns1').subnamespaces(), [
            common.Namespace([]),
            common.Namespace(['ns11']),
            common.Namespace(['ns:11'])
        ])
    self.assertSequenceEqual(mm.ns('ns2').subnamespaces(), [common.Namespace()])
    self.assertSequenceEqual(mm.ns('ns3').subnamespaces(), [])

  def test_namespace_add(self):
    n0 = common.Namespace()
    self.assertEmpty(n0)
    self.assertEqual(n0 + (), common.Namespace([]))
    self.assertEqual(n0 + ('ab',), common.Namespace([
        'ab',
    ]))
    self.assertEqual(n0 + ('a:b',), common.Namespace(['a:b']))
    self.assertEqual(n0 + ('a:b',), common.Namespace(['a:b']))
    self.assertEqual(n0 + ('ab', 'cd'), common.Namespace(['ab', 'cd']))
    n1 = common.Namespace(['xy'])
    self.assertLen(n1, 1)
    self.assertEqual(n1 + ('ab',), common.Namespace(['xy', 'ab']))
    self.assertEqual(n1 + ('a:b',), common.Namespace(['xy', 'a:b']))
    self.assertEqual(n1 + ('a:b',), common.Namespace(['xy', 'a:b']))
    n2 = common.Namespace(['xy', 'zw'])
    self.assertLen(n2, 2)
    self.assertLen(n2 + ('ab',), 3)
    self.assertEqual(n2 + ('ab',), common.Namespace(['xy', 'zw', 'ab']))
    self.assertLen(n2 + ('ab', 'cd'), 4)
    self.assertEqual(n2 + ('ab', 'cd'), common.Namespace.decode('xy:zw:ab:cd'))

  def test_metadata_attach(self):
    # Set up a metadata tree.
    mm = common.Metadata()
    mm.ns('ns1').ns('ns:11').update(foo='bar')
    mm.ns('ns1').ns('ns12').update(foo='gleep')
    mm.ns('ns1').update(foo='nerf')
    mm.ns('ns|').update(foo='pag')
    # Attach that metadata tree to a branch of an empty tree.
    m1 = common.Metadata()
    m1.ns('ns0').ns('ns00').attach(mm)
    self.assertEmpty(m1.abs_ns())
    self.assertEqual(m1.ns('ns0').ns('ns00'), mm)
    self.assertEqual(m1.abs_ns(['ns0', 'ns00', 'ns1', 'ns:11'])['foo'], 'bar')
    self.assertEqual(m1.abs_ns(['ns0', 'ns00', 'ns1', 'ns12'])['foo'], 'gleep')
    self.assertEqual(m1.abs_ns(['ns0', 'ns00', 'ns1'])['foo'], 'nerf')
    self.assertEqual(m1.abs_ns(['ns0', 'ns00', 'ns|'])['foo'], 'pag')
    # Attach just part of $mm to a branch of a new, empty tree.
    m2 = common.Metadata()
    m2.ns('nsX').attach(mm.ns('ns1'))
    self.assertEqual(m2.abs_ns(['nsX', 'ns:11'])['foo'], 'bar')
    self.assertEqual(m2.abs_ns(['nsX', 'ns12'])['foo'], 'gleep')
    self.assertEqual(m2.abs_ns(['nsX'])['foo'], 'nerf')
    # Check that attach() overwrites key collisions, but preserves other data.
    m3 = common.Metadata()
    m3['foo'] = 'Y'  # This will be overwritten.
    m3['z'] = 'Z'  # This will not be overwritten.
    m3.attach(mm.ns('ns1').ns('ns:11'))
    self.assertEqual(m3['z'], 'Z')
    self.assertEqual(m3['foo'], 'bar')


if __name__ == '__main__':
  absltest.main()
