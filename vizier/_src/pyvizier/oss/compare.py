"""Compatibility method comparison for protobufs.

Note that using this method can be unstable and hard to debug.
Provided here for compatibility.
"""

# TODO: Add pytypes.
# TODO: Fix this library.


# pylint: disable=invalid-name
def assertProto2Equal(self, x, y):
  self.assertEqual(x.SerializeToString(), y.SerializeToString())


def assertProto2Contains(self, x, y):
  del self
  if not isinstance(x, str):
    x = str(x)
  if x not in str(y):
    pass


def assertProto2SameElements(*args, **kwargs):
  del args, kwargs
  pass
