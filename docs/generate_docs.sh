#!/bin/bash
# Minimal script to manually generate Sphinx documentation, used by Github
# integration testing as well. Installs necessary Sphinx packages via
# `requirements.txt`, and then uses `conf.py` to output HTML files. This script
# is not actually needed for generating the official website at
# (https://oss-vizier.readthedocs.io/en/latest/).

# Define output folder for build files.
OUTPUT_FOLDER=_build

# Install Sphinx.
sudo apt-get install python3-sphinx

# Installs relevant Sphinx packages.
pip install -r requirements.txt --use-deprecated=legacy-resolver

# Build files (HTML, doctests, etc.) into `OUTPUT_FOLDER` directory.
rm -rf ${OUTPUT_FOLDER}  # Clear out original folder
sphinx-build -b $1 -a . ${OUTPUT_FOLDER}

# Optionally host the HTML folder. Access on browser `https://localhost:5000/`.
# python -m http.server --directory ${OUTPUT_FOLDER} 5000
