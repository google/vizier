#!/bin/bash

# Recommend potentially installing Python packages on a virtualenv.
# sudo apt-get install -y virtualenv python3-venv
# python3 -m venv vizierproject  # or virtualenv myproject
# source vizierproject/bin/activate

# This entire file installs core dependencies for OSS Vizier.
sudo apt-get install -y libprotobuf-dev  # Needed for proto libraries.

# Installs Python packages.
pip install -r requirements.txt --use-deprecated=legacy-resolver # Installs dependencies
pip install -e . # Installs Vizier

# Builds all .proto files.
./build_protos.sh
