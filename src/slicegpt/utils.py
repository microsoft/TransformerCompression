import torch


@torch.no_grad()
def pca_calc(X):
    torch.cuda.empty_cache()
    try:
        X = X.double().cuda()
        H = X.T @ X
    except:
        print("Out of memory, trying to calculate PCA on CPU!")
        X = X.cpu().double()
        H = X.T @ X
        H = H.cuda()
    del X
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).cuda()
    H[diag, diag] += damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec
