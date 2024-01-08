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

# Installs standard Google protos, then builds Vizier specific protos.
set -x

THIS_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
DEPOT="$THIS_DIR/../../../.."

cd "$THIS_DIR"


# Download the OSS shared protos from GitHub:
# https://github.com/googleapis/googleapis
TMPDIR=$(mktemp -d)
(cd "$TMPDIR"
  wget https://github.com/googleapis/googleapis/archive/master.tar.gz
  tar xzf master.tar.gz
  cp -r googleapis-master/google $THIS_DIR/vizier/_src/service/
)
rm -rf "$TMPDIR"

# No need to copy over well known types. They're linked into protoc.

# Generates the *.py files from the protos.
for proto_name in *.proto
do
  python3 -m grpc_tools.protoc \
    --python_out=vizier/_src/service \
    --grpc_python_out=vizier/_src/service \
    --proto_path=vizier/_src/service \
    --experimental_allow_proto3_optional \
    vizier/_src/service/$proto_name
done
