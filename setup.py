"""Setup for pip package."""
import os
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext


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


class BuildCmd(build_ext):
  """Custom installation script to build the protos."""

  def run(self):
    os.system('sh build_protos.sh')


_VERSION = _get_version()

setup(
    name='google-vizier',
    version=_VERSION,
    url='https://github.com/google/vizier',
    license='Apache License 2.0',
    author='Vizier Team',
    description='Vizier: Distributed service framework for blackbox optimization and research.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='oss-vizier-dev@google.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(
        include=['vizier*'], exclude=['*_test.py', 'examples']),
    install_requires=_parse_requirements('requirements.txt'),
    extras_require={
        'jax': _parse_requirements('requirements-jax.txt'),
        'tf': _parse_requirements('requirements-tf.txt'),
        'algorithms': _parse_requirements('requirements-algorithms.txt'),
        'benchmarks': _parse_requirements('requirements-benchmarks.txt')
    },
    requires_python='>=3.9',
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build_protos': BuildCmd},
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
    keywords='ai machine learning hyperparameter blackbox optimization framework',
)
