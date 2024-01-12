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
# Usage: `run_tests.sh (test_name)`
# where `test_name` can be the following case strings shown below.
#
# On a Github workflow, the `test_name` will be supplied from the YML file.

case $1 in
  "core")
    pip install -r requirements-jax.txt
    pytest -n auto vizier \
    --ignore=vizier/_src/benchmarks/ \
    --ignore=vizier/_src/algorithms/ \
    --ignore=vizier/_src/pyglove/ \
    --ignore=vizier/_src/jax/ \
    --ignore=vizier/_src/raytune/
    ;;
  "algorithms")
    pip install -r requirements-algorithms.txt \
    -r requirements-jax.txt
    echo "These tests are too slow."
    # pytest -n auto vizier/_src/algorithms/
    ;;
  "benchmarks")
    pip install -r requirements-jax.txt \
    -r requirements-tf.txt \
    -r requirements-benchmarks.txt
    pytest -n auto vizier/_src/benchmarks/
    ;;
  "clients")
    python vizier/service/clients/__init__.py
    ;;
  "pyglove")
    pip install -r requirements-jax.txt
    pip install pyglove
    pytest -n auto vizier/_src/pyglove/
    ;;
  "raytune")
    pip install -U ray[tune]
    pip install -r requirements-jax.txt
    pip install pyarrow
    pip install pandas
    pytest -n auto vizier/_src/raytune/
    ;;
esac
