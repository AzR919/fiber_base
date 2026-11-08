"""Rank tests for the single-molecule analyses without scipy (not in the nibi venv). Normal approximations with tie
corrections, two-sided unless stated; fine for the n >= 10 used here."""
import math

import numpy as np
import pandas as pd


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon_p(x):
    """Wilcoxon signed-rank test of median 0 (zeros dropped), two-sided p."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    x = x[x != 0]
    n = len(x)
    if n < 5:
        return np.nan
    r = pd.Series(np.abs(x)).rank().values
    w = r[x > 0].sum()
    mu = n * (n + 1) / 4
    _, counts = np.unique(np.abs(x), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    return min(1.0, 2 * _norm_sf(abs(w - mu) / math.sqrt(var)))


def mannwhitney_greater_p(a, b):
    """Mann-Whitney U test, one-sided p for a > b."""
    a, b = np.asarray(pd.Series(a).dropna(), float), np.asarray(pd.Series(b).dropna(), float)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return np.nan
    r = pd.Series(np.r_[a, b]).rank().values
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    n = n1 + n2
    _, counts = np.unique(np.r_[a, b], return_counts=True)
    var = n1 * n2 / 12 * ((n + 1) - (counts ** 3 - counts).sum() / (n * (n - 1)))
    return _norm_sf((u - n1 * n2 / 2) / math.sqrt(var))


def spearman(a, b):
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))
