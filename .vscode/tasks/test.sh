#!/bin/bash
set -ex

# black --check .
cd scratch/dev1
pyright
pytest --cov=lindi --cov-report=xml --cov-report=term tests/