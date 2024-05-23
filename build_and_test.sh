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

./test.sh

deactivate