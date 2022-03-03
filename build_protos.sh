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
  tar xzvf master.tar.gz
  cp -r googleapis-master/google $THIS_DIR/vizier/service/
)
rm -rf "$TMPDIR"

# No need to copy over well known types. They're linked into protoc.

# Generates the *.py files from the protos.
for proto_name in *.proto
do
  python3 -m grpc_tools.protoc \
    --python_out=vizier/service \
    --grpc_python_out=vizier/service \
    --proto_path=vizier/service \
    vizier/service/$proto_name
done
