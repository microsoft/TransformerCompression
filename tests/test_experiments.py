"""
Tests that run and validate some of the scripts in the experiments directory, not to be used on the build machine.
Run with `pytest -m experiment` on a GPU VM.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch


def teardown_module(module):
    """Remove the log and model_data directories after the tests."""
    test_dir = get_test_dir()

    log_dir = test_dir / 'log'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    model_data_dir = test_dir / 'test_model_data'
    if os.path.exists(model_data_dir):
        shutil.rmtree(model_data_dir)


def get_test_dir() -> pathlib.Path:
    """Return the path to the directory containing these tests."""
    return pathlib.Path(__file__).parent


def run_python_script(script: str | pathlib.Path, script_args: list[str]) -> str:
    """
    Run the provided Python script and return its output.
    Use the same Python interpreter as the current process.
    """

    def run_shell_command(command: str, cmd_args: list[str]) -> str:
        """Run the provided shell command in the specified working directory and return its output."""
        output = subprocess.run(
            [command] + cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=get_test_dir(),
        )

        if output.returncode != 0:
            raise RuntimeError(
                f"Command {command} with args {cmd_args} failed with error code {output.returncode}:\n{output.stderr}"
            )

        return output.stdout.decode()

    python = str(sys.executable)
    py_args = [str(script)] + script_args
    return run_shell_command(python, py_args)


def check_ppl_in_log(log: str, expected_ppl: float | None, expected_parameters: int | None) -> None:
    """Verify that the log contains the expected perplexity and parameters result."""

    def verify(pattern: str, value: float | int) -> None:
        match = re.search(pattern, log)
        assert match, f'Expected to find pattern {pattern} in the log'
        assert np.isclose(
            float(match.group(1).replace(',', '')), value, atol=5e-1, rtol=1e-2
        ), f'Expected {value} but got {match.group(1)}'

    if expected_ppl is not None:
        verify(r'(?:After rotating and slicing|Loaded model perplexity:) (\d+\.\d+)', expected_ppl)

    if expected_parameters is not None:
        verify(r'Sliced model parameters: ([0-9,]+)', expected_parameters)


def verify_run_slicegpt_perplexity(model: str, sparsity: float, expected_ppl: float, expected_parameters: int) -> None:
    """Test the run_slicegpt_perplexity.py script with the provided parameters."""
    # test rotate, slice and save model
    tests_dir = get_test_dir()
    script = tests_dir.parent / 'experiments' / 'run_slicegpt_perplexity.py'
    save_dir = tests_dir / 'test_model_data'
    args = ['--no-wandb', '--model', str(model)]

    ext_args = ['--sparsity', str(sparsity), '--save-dir', str(save_dir)]
    log = run_python_script(script, args + ext_args)
    check_ppl_in_log(log, expected_ppl=expected_ppl, expected_parameters=expected_parameters)

    # test load and slice model
    model_path = save_dir / (model.split('/')[-1] + '_' + str(sparsity) + '.pt')
    ext_args = ['--sliced-model-path', str(model_path)]
    log = run_python_script(script, args + ext_args)
    check_ppl_in_log(log, expected_ppl=expected_ppl, expected_parameters=None)


@pytest.mark.experiment
@pytest.mark.gpu
def test_opt_125m():
    """Test the run_slicegpt_perplexity.py script with the facebook/opt-125m model."""
    assert torch.cuda.is_available()

    verify_run_slicegpt_perplexity(
        model='facebook/opt-125m',
        sparsity=0.2,
        expected_ppl=34.53,
        expected_parameters=138_281_664,
    )


@pytest.mark.experiment
@pytest.mark.gpu
def test_phi_2():
    """Test the run_slicegpt_perplexity.py script with the microsoft/phi-2 model."""
    assert torch.cuda.is_available()

    verify_run_slicegpt_perplexity(
        model='microsoft/phi-2',
        sparsity=0.2,
        expected_ppl=11.2691,
        expected_parameters=2_256_505_856,
    )
