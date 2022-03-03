#!/bin/bash

# Recommend potentially installing Python packages on a virtualenv.
# Commented out for now by default.
"""
sudo apt-get install -y virtualenv python3-venv
virtualenv myproject  # or python3 -m venv myproject
source myproject/bin/activate
"""

# This entire file installs core dependencies for OSS Vizier.
sudo apt-get install -y libprotobuf-dev  # Needed for proto libraries.


# Installs Python packages.
pip install -r requirements.txt # Installs dependencies
pip install -e . # Installs Vizier

# Builds all .proto files.
./build_protos.sh
