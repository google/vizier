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
#
# Usage: `install_test_deps.sh (test_name)`
#     where `test_name` can be the following case strings shown below.
#
# In a GitHub workflow, the `test_name` will be supplied in the YAML file.

case $1 in
  "core")
    pip install -r requirements-jax.txt
    ;;
  "algorithms")
    pip install -r requirements-algorithms.txt -r requirements-jax.txt
    ;;
  "benchmarks")
    pip install \
        -r requirements-jax.txt \
        -r requirements-tf.txt \
        -r requirements-benchmarks.txt
    ;;
  "clients")
    ;;
  "pyglove")
    pip install -r requirements-jax.txt
    pip install pyglove
    ;;
  "raytune")
    pip install -U ray[tune]
    pip install -r requirements-jax.txt
    pip install pyarrow
    pip install pandas
    ;;
esac
