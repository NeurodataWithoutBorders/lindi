#!/bin/bash
set -ex

# black --check .
pyright
pytest --cov=lindi --cov-report=xml --cov-report=term tests/
