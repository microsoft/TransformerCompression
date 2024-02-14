# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Style, type hints and lint checks
pre-commit run --all-files --hook-stage manual --verbose

# Run tests
python3 -m pytest -m "not (gpu or experiment)"