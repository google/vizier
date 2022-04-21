#!/bin/bash
# Runs all Python unit tests in the Vizier package.
python -m unittest discover -s . -p '*_test.py'
