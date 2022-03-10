"""Tests for vizier.pyvizier.shared.common."""

import copy
from vizier.pyvizier.shared import common
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
    self.assertEqual(repr(ns0), '')
    n1t = common.Namespace(('aerer',))
    self.assertLen(n1t, 1)
    n1 = common.Namespace('a78')
    self.assertLen(n1, 1)
    self.assertEqual(str(n1), 'a78')
    n2 = common.Namespace(('a78', 'bfe'))
    self.assertLen(n2, 2)
    n2s1 = common.Namespace('a78:bfe')
    self.assertLen(n2s1, 2)
    self.assertEqual(repr(n2), repr(n2s1))
    n2s2 = common.Namespace(':a78:bfe')
    self.assertLen(n2s2, 2)
    self.assertEqual(repr(n2), repr(n2s2))
    self.assertEqual(n2, n2s2)
    self.assertEqual(n2s1, n2s2)
    ns = common.Namespace(('a', 'b'))
    self.assertLen(ns, 2)
    self.assertEqual(tuple(ns), ('a', 'b'))
    self.assertEqual(str(ns), 'a:b')
    self.assertEqual(repr(ns), 'a:b')

  def test_escape(self):
    ns1 = common.Namespace('a|A')
    self.assertLen(ns1, 1)
    self.assertEqual(str(ns1), 'a|A')
    self.assertEqual(repr(ns1), 'a||A')
    self.assertEqual(common.Namespace(repr(ns1)), ns1)
    #
    ns2 = common.Namespace('b:B')
    self.assertLen(ns2, 2)
    self.assertEqual(str(ns2), 'b:B')
    self.assertEqual(repr(ns2), 'b:B')
    self.assertEqual(common.Namespace(repr(ns2)), ns2)
    ns2 = common.Namespace(':b:B')
    self.assertLen(ns2, 2)
    self.assertEqual(str(ns2), 'b:B')
    self.assertEqual(repr(ns2), 'b:B')
    self.assertEqual(common.Namespace(repr(ns2)), ns2)
    #
    ns1e1 = common.Namespace(':b|B')
    self.assertLen(ns1e1, 1)
    self.assertEqual(repr(ns1e1), 'b||B')
    self.assertEqual(common.Namespace(repr(ns1e1)), ns1e1)
    ns1e2 = common.Namespace(('b|B',))
    self.assertLen(ns1e2, 1)
    self.assertEqual(repr(ns1e2), 'b||B')
    self.assertEqual(ns1e2, ns1e1)
    self.assertEqual(common.Namespace(repr(ns1e2)), ns1e2)
    #
    ns1c = common.Namespace(':b|cB')
    self.assertLen(ns1c, 1)
    self.assertEqual(repr(ns1c), 'b|cB')
    self.assertEqual(common.Namespace(repr(ns1c)), ns1c)
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
    self.assertNotEmpty(repr(mm), '')

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

  def test_subnamespace(self):
    mm = common.Metadata()
    mm.ns('ns1')['foo'] = 'bar'
    mm.ns('ns2')['foo'] = 'bar'
    mm.ns('ns1').ns('ns11')['foo'] = 'bar'

    self.assertSequenceEqual(mm.subnamespaces(), [
        common.Namespace('ns1'),
        common.Namespace('ns2'),
        common.Namespace(['ns1', 'ns11'])
    ])
    self.assertSequenceEqual(
        mm.ns('ns1').subnamespaces(),
        [common.Namespace(), common.Namespace('ns11')])

  def test_attach(self):
    mm = common.Metadata()
    mm.ns('ns1').ns('ns11').update(foo='bar')
    mm.ns('ns1').ns('ns12').update(foo='bar')
    m1 = common.Metadata()
    m1.ns('ns0').ns('ns00').attach(mm)
    self.assertEmpty(m1.abs_ns())
    self.assertEqual(m1.ns('ns0').ns('ns00'), mm)


if __name__ == '__main__':
  absltest.main()
