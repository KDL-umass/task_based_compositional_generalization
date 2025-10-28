#!/bin/bash
set -euxo pipefail

isort --check --profile black --diff --gitignore src/ scripts/
black --check src/ scripts/
flake8 src/ scripts/