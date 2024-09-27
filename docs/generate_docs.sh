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

#!/bin/bash
# Minimal script to manually generate Sphinx documentation, used by Github
# integration testing as well. Installs necessary Sphinx packages via
# `requirements.txt`, and then uses `conf.py` to output HTML files. This script
# is not actually needed for generating the official website at
# (https://oss-vizier.readthedocs.io/en/latest/).

# Define output folder for build files.
OUTPUT_FOLDER=_build

# Install Sphinx.
pip install sphinx>=8.0.0  # Make sure up-to-date with ReadTheDocs server.
# Installs relevant Sphinx packages.
pip install -r requirements-docs.txt

# Build files (HTML, doctests, etc.) into `OUTPUT_FOLDER` directory.
rm -rf ${OUTPUT_FOLDER}  # Clear out original folder
sphinx-build -b $1 -a . ${OUTPUT_FOLDER}

# Optionally host the HTML folder. Access on browser `https://localhost:5000/`.
# python -m http.server --directory ${OUTPUT_FOLDER} 5000
