from __future__ import annotations

import warnings
from contextlib import contextmanager, ExitStack
from typing import Iterator, List, Optional, Tuple

import torch
from botorch import settings
from botorch.exceptions import InputDataError, InputDataWarning
from botorch.settings import _Flag
from gpytorch import settings as gpt_settings
from gpytorch.module import Module
from torch import Tensor
from botorch.models.utils.assorted import detect_duplicates

def consolidate_duplicates(
    X: Tensor, Y: Tensor, Z: Tensor, rtol: float = 0, atol: float = 1e-8
) -> Tuple[Tensor, Tensor, Tensor]:
    """Drop duplicated Xs and update the indices tensor Y accordingly.
    Supporting 2d Tensor only as in batch mode block design is not guaranteed.

    Args:
        X: the datapoints tensor
        Y: the index tensor to be updated (e.g., multiwise comparisons)
        Z: choices
        rtol: relative tolerance
        atol: absolute tolerance

    Returns:
        consolidated_X: the consolidated X
        consolidated_Y: the consolidated Y (e.g., multiwise comparisons indices)
        consolidated_Z: the consolidated Z
        new_indices: new index of each original item in X, a tensor of size X.shape[-2]
    """
    if len(X.shape) != 2:
        raise ValueError("X must have 2 dimensions.")

    n = X.shape[-2]
    dup_map = dict(detect_duplicates(X=X, rtol=rtol, atol=atol))

    # Handle edge cases conservatively
    # If a item is in both dup set and kept set, do not remove it
    common_set = set(dup_map.keys()).intersection(dup_map.values())
    for k in list(dup_map.keys()):
        if k in common_set or dup_map[k] in common_set:
            del dup_map[k]

    if dup_map:
        dup_indices, kept_indices = zip(*dup_map.items())

        unique_indices = sorted(set(range(n)) - set(dup_indices))

        # After dropping the duplicates,
        # the kept ones' indices may also change by being shifted up
        new_idx_map = dict(zip(unique_indices, range(len(unique_indices))))
        new_indices_for_dup = (new_idx_map[idx] for idx in kept_indices)
        new_idx_map.update(dict(zip(dup_indices, new_indices_for_dup)))
        consolidated_X = X[list(unique_indices), :]
        consolidated_Y = torch.tensor(
            [[new_idx_map[item.item()] for item in row] for row in Y.unbind()],
            dtype=torch.long,
            device=Y.device,
        )
        consolidated_Z = torch.tensor(
            [new_idx_map[item.item()] for item in Z],
            dtype=torch.long,
            device=Z.device,
        )
        new_indices = (
            torch.arange(n, dtype=torch.long)
            .apply_(lambda x: new_idx_map[x])
            .to(Y.device)
        )
        return consolidated_X, consolidated_Y, consolidated_Z, new_indices
    else:
        return X, Y, Z, torch.arange(n, device=Y.device, dtype=Y.dtype)