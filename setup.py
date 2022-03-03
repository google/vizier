"""Sets up Vizier."""
from setuptools import setup

setup(
    name='Vizier',
    version=1.0,
    description='Python-based interface for blackbox optimization and research, based on the internal Vizier service at Google.',
    url='https://github.com/google/vizier',
    author='Vizier Team',
    author_email='vizier-team@google.com',
    license='Apache License 2.0',
    keywords='ai machine learning hyperparameter blackbox optimization',
    packages=['vizier'],
)
