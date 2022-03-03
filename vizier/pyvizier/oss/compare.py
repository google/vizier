"""Compatibility method comparison for protobufs.

Note that using this method can be unstable and hard to debug.
Provided here for compatibility.
"""


# pylint: disable=invalid-name
def assertProto2Equal(self, x, y):
  self.assertEqual(x.SerializeAsString(), y.SerializeAsString())
