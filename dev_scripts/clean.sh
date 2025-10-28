#!/bin/bash
set -euxo pipefail

isort --profile black --gitignore src/ scripts/
black src/ scripts/