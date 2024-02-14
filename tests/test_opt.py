import time

import pytest
import torch


@pytest.mark.gpu
def test_opt_125m():
    """A dummy test that requires GPU and runs for 3 minutes."""
    assert torch.cuda.is_available()

    # TODO: James, please add the actual test here.
    time.sleep(180)
