# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from slicegpt import rotate


def test_pca():
    """Sanity checks for PCA calculation"""

    def sorted_by_dim(val, vec):
        # sort values and vectors by the dimension of the largest element in each vector
        order = torch.argsort(torch.argmax(vec.abs(), dim=0))
        return val[order], vec[:, order]

    def allclose_up_to_alignment(a, b, atol=1e-5):
        for i in range(a.shape[1]):
            if not torch.allclose(a[:, i], b[:, i], atol=atol) and not torch.allclose(a[:, i], -b[:, i], atol=atol):
                return False

        return True

    # Case 1: points already have maximum variance across standard axes => expect no rotation
    x = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float, device='cpu')
    x *= 0.5**0.5  # scale so that eigenvalues are 1.0
    eigen_val, eigen_vec = rotate.pca_calc([torch.unsqueeze(x, dim=0)])

    # TODO: this should not be required
    eigen_val = eigen_val.to(dtype=torch.float, device='cpu')
    eigen_vec = eigen_vec.to(dtype=torch.float, device='cpu')

    # TODO: these currently fail unless fix above
    assert eigen_val.device == x.device
    assert eigen_vec.device == x.device
    assert eigen_val.dtype == x.dtype
    assert eigen_vec.dtype == x.dtype

    # Eigenvalues should be 1.0
    assert torch.allclose(eigen_val, torch.full_like(eigen_val, 1.0, dtype=torch.float, device='cpu'), atol=1e-1)

    # Eigenvectors should be the identity matrix
    _, sorted_vec = sorted_by_dim(eigen_val, eigen_vec)
    assert allclose_up_to_alignment(sorted_vec, torch.eye(2, dtype=torch.float, device='cpu'), atol=1e-5)

    # A pi/4 rotation matrix
    angle = torch.tensor(torch.pi / 4, dtype=torch.float, device='cpu')
    c, s = torch.cos(angle), torch.sin(angle)
    rm = torch.tensor([[c, -s], [s, c]], dtype=torch.float, device='cpu')

    # Case 2: points lie on the main diagonal => expect a pi/4 rotation
    x = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float, device='cpu')
    eigen_val, eigen_vec = rotate.pca_calc([torch.unsqueeze(x, dim=0)])

    # TODO: this should not be required
    eigen_val = eigen_val.to(dtype=torch.float, device='cpu')
    eigen_vec = eigen_vec.to(dtype=torch.float, device='cpu')

    # eigenvalues
    assert torch.allclose(eigen_val, torch.tensor([4.0, 0.0], dtype=torch.float, device='cpu'), atol=1e-1)

    # eigenvectors
    assert allclose_up_to_alignment(eigen_vec, rm, atol=1e-5)
