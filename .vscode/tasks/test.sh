#!/bin/bash
set -ex

# black --check .
flake8 .
pyright
pytest --cov=lindi --cov-report=xml --cov-report=term tests/
