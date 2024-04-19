import torch

from quarot.hadamard_utils import get_hadK, matmul_hadU_cuda


class OnlineHadamard(torch.nn.Module):
    def __init__(self, hadamard_dim, force_fp32=False):
        super().__init__()
        self.fp32_had = force_fp32
        had_rem_dim, self.rem_dim = get_hadK(hadamard_dim)
        if had_rem_dim is not None:
            self.register_buffer("had_rem_dim", had_rem_dim)
            if not self.fp32_had:
                self.had_rem_dim = self.had_rem_dim.to(torch.float16)
        else:
            self.had_rem_dim = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.fp32_had:
            x = x.float()

        x = matmul_hadU_cuda(
            x, self.had_rem_dim, self.rem_dim
        )  # NB this uses a CUDA kernel from fast_hadamard_transform

        x = x.to(x_dtype)
        return x
