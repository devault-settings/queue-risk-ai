"""Temporal train/test split and rolling-origin validation folds.

Lecture 2: the split mirrors "learning from the past, predicting the future".
Lecture 6: with time-stamped data the split must be temporal, or future seasons leak in.
Lecture 4: model selection uses the training data only; the test set is "showtime",
used once at the end. Random k-fold would leak the future here, so the folds roll forward.
"""
import numpy as np
import pandas as pd

from . import config


def temporal_split(years: pd.Series, train_years=config.TRAIN_YEARS, test_years=config.TEST_YEARS):
    """Boolean masks (train, test) from each project's queue-entry year."""
    y = years.astype(int)
    train = y.between(*train_years).to_numpy()
    test = y.between(*test_years).to_numpy()
    assert not (train & test).any()
    return train, test


def rolling_folds(years: pd.Series, val_years=config.CV_VAL_YEARS):
    """List of (fit_positions, val_positions): fit on entries before t, validate on year t.

    Pass the list straight to scikit-learn as `cv=` (e.g. GridSearchCV(cv=rolling_folds(...))).
    `years` must be the training rows only.
    """
    y = years.astype(int).to_numpy()
    folds = []
    for t in val_years:
        fit, val = np.flatnonzero(y < t), np.flatnonzero(y == t)
        if len(fit) and len(val):
            folds.append((fit, val))
    return folds
