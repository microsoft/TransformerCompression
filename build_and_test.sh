# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Create and activate virtual environment
rm -rf .venv/
python3 -m venv .venv
. .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
pip install -e .[quarot]  # Required for quarot tests, must be installed once torch, setuptools and packaging installed.

./test.sh

deactivate