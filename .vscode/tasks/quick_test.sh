#!/bin/bash
set -ex

# black --check .

cd lindi
flake8 .
cd ..

pytest --cov=lindi --cov-report=xml --cov-report=term -m "not slow and not network" tests/
