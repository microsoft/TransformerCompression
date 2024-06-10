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
        float(match.group(1).replace(',', '')), value, atol=1e-1, rtol=1e-2
    ), f'Expected {value} but got {match.group(1)}'


def verify_run_quarot(args: list[str], expected_ppl: float) -> None:
    """Test QuaRot computational invariance: ppl should be same as the base model."""
    tests_dir = get_test_dir()
    script = tests_dir.parent / 'experiments' / 'run_quarot.py'
    log = run_python_script(script, args)
    verify(log, r'(?:QuaRot ppl:) (\d+\.\d+)', expected_ppl)


@pytest.mark.quarot_experiment
@pytest.mark.gpu
def test_phi3_rotate():
    """Test applying QuaRot to microsoft/Phi-3-mini-4k-instruct."""
    assert torch.cuda.is_available()

    model = 'microsoft/Phi-3-mini-4k-instruct'
    args = ['--no-wandb', '--model', str(model), '--rotate']

    verify_run_quarot(
        args=args,
        expected_ppl=6.35,
    )


@pytest.mark.quarot_experiment
@pytest.mark.gpu
def test_phi3_weight_only_rtn():
    """Test QuaRot 4-bit asymmetric weight-only RTN with microsoft/Phi-3-mini-4k-instruct."""
    assert torch.cuda.is_available()

    model = 'microsoft/Phi-3-mini-4k-instruct'
    bits = 4
    args = ['--no-wandb', '--model', str(model), '--rotate', '--w-rtn', '--w-bits', str(bits), '--w-asym']

    verify_run_quarot(
        args=args,
        expected_ppl=8.57,
    )


@pytest.mark.quarot_experiment
@pytest.mark.gpu
def test_phi3_rtn():
    """Test QuaRot A4W4KV4 RTN with microsoft/Phi-3-mini-4k-instruct."""
    assert torch.cuda.is_available()

    model = 'microsoft/Phi-3-mini-4k-instruct'
    bits = 4
    args = [
        '--no-wandb',
        '--model',
        str(model),
        '--rotate',
        '--w-rtn',
        '--w-bits',
        str(bits),
        '--a-bits',
        str(bits),
        '--a-clip-ratio',
        '0.8',
        '--k-bits',
        str(bits),
        '--v-bits',
        str(bits),
    ]

    verify_run_quarot(
        args=args,
        expected_ppl=11.69,
    )
