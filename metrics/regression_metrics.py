# -*- coding: utf-8 -*-
r"""
Regression Metrics
==============
    Metrics to evaluate regression quality of estimator models.

    Shamelessly copied and adapted.
"""
import warnings

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

def t_kendalltau(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes Kendall correlation.
    :param x: predicted scores.
    :param y: ground truth scores.
    :return: Kendall Tau correlation value.
    """
    x = x.cpu().detach().numpy().flatten()
    y = y.cpu().detach().numpy().flatten()
    return torch.tensor(kendalltau(x, y)[0], dtype=torch.float32)


def t_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes Pearson correlation.
    :param x: predicted scores.
    :param y: ground truth scores.
    :return: Pearson correlation value.
    """
    x = x.cpu().detach().numpy().flatten()
    y = y.cpu().detach().numpy().flatten()
    return torch.tensor(pearsonr(x, y)[0], dtype=torch.float32)


def t_spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes Spearman correlation.
    :param x: predicted scores.
    :param y: ground truth scores.
    Return:
        - Spearman correlation value.
    """
    x = x.cpu().detach().numpy().flatten()
    y = y.cpu().detach().numpy().flatten()
    return torch.tensor(spearmanr(x, y)[0], dtype=torch.float32)