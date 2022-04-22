"""Init file."""
import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PROTO_ROOT = os.path.realpath(os.path.join(THIS_DIR, "service"))

sys.path.append(PROTO_ROOT)

__version__ = "0.0.1.alpha"
