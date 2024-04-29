import torch

from quarot.hadamard_utils import _apply_fast_hadamard, _apply_slow_hadamard, fht_available, get_hadK


class OnlineHadamard(torch.nn.Module):
    def __init__(self, hadamard_dim, force_fp32=False):
        super().__init__()
        self.fp32_had = force_fp32
        had_rem_dim = get_hadK(hadamard_dim)
        self.register_buffer("had_rem_dim", had_rem_dim)
        if not self.fp32_had:
            self.had_rem_dim = self.had_rem_dim.to(torch.float16)

        if fht_available:
            self.had = _apply_fast_hadamard
        else:
            self.had = _apply_slow_hadamard

    def forward(self, x):
        x_dtype = x.dtype
        if self.fp32_had:
            x = x.float()

        x = self.had(x, self.had_rem_dim)  # NB this uses a CUDA kernel from fast_hadamard_transform

        x = x.to(x_dtype)
        return x
