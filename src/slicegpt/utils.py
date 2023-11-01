# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import datetime
import pathlib

import torch


@torch.no_grad()
def pca_calc(X):
    torch.cuda.empty_cache()
    try:
        X = X.double().cuda()
        H = X.T @ X
    except:
        logging.info("Out of memory, trying to calculate PCA on CPU!")
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


def configure_logging(
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: str = 'log',
    level: int = logging.INFO,
) -> None:
    handlers = []

    if log_to_console:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handlers.append(handler)

    if log_to_file:
        path = pathlib.Path.cwd() / log_dir / f'{datetime.datetime.now():log_%Y-%m-%d-%H-%M-%S}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(f'%(asctime)s.%(msecs)04d %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        handlers.append(handler)

    logging.basicConfig(
        level=level,
        format=f'%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        encoding='utf-8',
        handlers=handlers,
    )
