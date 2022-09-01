#!/bin/bash
# Runs all core Python unit tests in the Vizier package.
pytest vizier --ignore=vizier/_src/integration/ --ignore=vizier/_src/benchmarks/ --ignore=vizier/_src/algorithms/

# Run algorithm tests. Disclaimer: Algorithms use multiple external libraries, some of which may break.
pytest -n auto vizier/_src/algorithms/

# Run benchmark tests. Disclaimer: Benchmarks use multiple external libraries, some of which may break.
pytest -n auto vizier/_src/benchmarks/

