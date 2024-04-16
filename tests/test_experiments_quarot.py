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


def verify_run_quarot(model: str, bits: int, rtn: bool, expected_ppl: float) -> None:
    """Test the run_quarot.py script with the provided parameters."""

    # test rotate, quantize and save model
    tests_dir = get_test_dir()
    script = tests_dir.parent / 'experiments' / 'run_quarot.py'
    save_dir = tests_dir / 'test_model_data'

    args = [
        '--no-wandb',
        '--model',
        str(model),
        '--rotate',
        '--w-bits',
        str(bits),
        '--w-clip',
        '--a-bits',
        str(bits),
        '--v-bits',
        str(bits),
        '--k-bits',
        str(bits),
        '--v-asym',
        '--k-asym',
        '--v-groupsize',
        '128',
        '--k-groupsize',
        '128',
        '--a-clip-ratio',
        '0.9',
        '--k-clip-ratio',
        '0.95',
        '--v-clip-ratio',
        '0.95',
    ]

    if rtn:
        args.append('--w-rtn')

    # ext_args = ['--save-dir', str(save_dir)]
    log = run_python_script(script, args)
    verify(log, r'(?:QuaRot ppl:) (\d+\.\d+)', expected_ppl)

    # TODO: test load a quarot model


@pytest.mark.quarot
@pytest.mark.gpu
def test_llama2_7b():
    """Test run_quarot.py with the meta-llama/Llama-2-7b model."""
    assert torch.cuda.is_available()

    # Test 4-bit RTN quantization
    model = 'meta-llama/Llama-2-7b-hf'
    bits = 4
    rtn = True
    verify_run_quarot(
        model=model,
        bits=bits,
        rtn=rtn,
        expected_ppl=8.37,
    )
