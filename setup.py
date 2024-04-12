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

"""Setup for pip package."""

import datetime
import itertools
import os
import sys
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command.build import build


def _get_version():
  with open('vizier/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g['__version__']
    raise ValueError('`__version__` not defined in `vizier/__init__.py`')


def _strip_comments_from_line(s: str) -> str:
  """Parses a line of a requirements.txt file."""
  requirement, *_ = s.split('#')
  return requirement.strip()


def _parse_requirements(requirements_txt_path: str) -> list[str]:
  """Returns a list of dependencies for setup() from requirements.txt."""

  # Currently a requirements.txt is being used to specify dependencies. In order
  # to avoid specifying it in two places, we're going to use that file as the
  # source of truth.
  with open(requirements_txt_path) as fp:
    # Parse comments.
    lines = [_strip_comments_from_line(line) for line in fp.read().splitlines()]
    # Remove empty lines and direct github repos (not allowed in PyPI setups)
    return [l for l in lines if (l and 'github.com' not in l)]


class BuildCmd(build):
  """Custom installation script to build the protos."""

  def run(self):
    current_path = os.path.dirname(os.path.realpath(__file__))
    sys.stdout.write('current_path: {}'.format(current_path))
    with os.scandir('.') as it:
      for entry in it:
        if entry.name.startswith('build_protos.sh'):
          sys.stdout.write('{}'.format(entry))
    if os.system('bash build_protos.sh'):
      raise OSError('Failed to run build_protos.sh')
    build.run(self)


_VERSION = _get_version()
_NAME = 'google-vizier'

if '--dev' in sys.argv:
  sys.argv.remove('--dev')
  _VERSION += '.dev' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  _NAME += '-dev'

extras_require = {
    'jax': _parse_requirements('requirements-jax.txt'),
    'tf': _parse_requirements('requirements-tf.txt'),
    'algorithms': _parse_requirements('requirements-algorithms.txt'),
    'benchmarks': _parse_requirements('requirements-benchmarks.txt'),
    'test': _parse_requirements('requirements-test.txt'),
}

extras_require['all'] = list(
    itertools.chain.from_iterable(extras_require.values())
)

setup(
    name=_NAME,
    version=_VERSION,
    url='https://github.com/google/vizier',
    license='Apache License 2.0',
    author='Vizier Team',
    description=(
        'Open Source Vizier: Distributed service framework for blackbox'
        ' optimization and research.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='oss-vizier-dev@google.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(
        include=['vizier*'], exclude=['*_test.py', 'examples']
    ),
    install_requires=_parse_requirements('requirements.txt'),
    extras_require=extras_require,
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build': BuildCmd},
    # PyPI package information.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    keywords=(
        'ai machine learning hyperparameter blackbox optimization framework'
    ),
)
