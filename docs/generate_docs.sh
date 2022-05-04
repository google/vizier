#!/bin/bash
# Minimal script to generate Sphinx documentation. Installs necessary Sphinx
# packages via `requirements.txt`, and then uses `conf.py` to output HTML files.
# See vizier/google/README.md for more details.

# Define output folder for build files.
OUTPUT_FOLDER=/tmp/output

# Install Sphinx.
sudo apt-get install python3-sphinx

# Installs relevant Sphinx packages.
pip install -r requirements.txt --use-deprecated=legacy-resolver

# Build HTML webpage files into `OUTPUT_FOLDER` directory.
rm -rf ${OUTPUT_FOLDER}  # Clear out original folder
sphinx-build -b html -a . ${OUTPUT_FOLDER}

# Optionally host the HTML folder. Access on browser `https://localhost:5000/`.
python -m http.server --directory ${OUTPUT_FOLDER} 5000
