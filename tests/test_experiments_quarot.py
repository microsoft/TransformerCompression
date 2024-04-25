"""
Tests that run and validate some of the scripts in the experiments directory, not to be used on the build machine.
Run with `pytest -m quarot` on a GPU VM.
"""
import re

import numpy as np
import pytest
import torch
from test_experiments import get_test_dir, run_python_script


def verify(log: str, pattern: str, value: float | int) -> None:
    """Verify that the log contains the expected value for the provided pattern."""
    match = re.search(pattern, log)
    assert match, f'Expected to find pattern {pattern} in the log'
    assert np.isclose(
        float(match.group(1).replace(',', '')), value, atol=5e-1, rtol=1e-2
    ), f'Expected {value} but got {match.group(1)}'


def verify_run_quarot_no_quant(model: str, expected_ppl: float) -> None:
    """Test the run_quarot.py script with the provided parameters."""

    # test rotate, quantize and save model
    tests_dir = get_test_dir()
    script = tests_dir.parent / 'experiments' / 'run_quarot.py'

    args = ['--no-wandb', '--model', str(model)]

    log = run_python_script(script, args)
    verify(log, r'(?:QuaRot ppl:) (\d+\.\d+)', expected_ppl)

    # TODO: test load a quarot model


@pytest.mark.quarot
@pytest.mark.gpu
def test_llama2_7b():
    """Test run_quarot.py with the meta-llama/Llama-2-7b model."""
    assert torch.cuda.is_available()

    # Test QuaRot with no quantization.
    model = 'meta-llama/Llama-2-7b-hf'
    verify_run_quarot_no_quant(
        model=model,
        expected_ppl=5.47,
    )
