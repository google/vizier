# Copyright 2023 Google LLC.
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
# Runs all core Python unit tests in the Vizier package.
pytest vizier --ignore=vizier/_src/benchmarks/ --ignore=vizier/_src/algorithms/

# Run algorithm tests. Disclaimer: Algorithms use multiple external libraries, some of which may break.
pytest -n auto vizier/_src/algorithms/

# Run benchmark tests. Disclaimer: Benchmarks use multiple external libraries, some of which may break.
pytest -n auto vizier/_src/benchmarks/

