"""Comb-content analysis (paper Figures 1-5).

Reproduces the first author's ``CombContent_Analysis.ipynb`` pipeline:
seasonality of nest contents (Fig. 1), cell-level neighbourhood,
transitions and clustering (Fig. 2), comb- and nest-level composition
(Fig. 3), specialization (Fig. 4) and the shuffled-vs-control comparisons
(Fig. 5).  Shared constants/loaders/functions are here; one module per
figure group adds the figure functions.
"""

from . import (celllevel, comblevel, constants, data, nestlevel,
              seasonality, specialization, util)
from .constants import *  # noqa: F401,F403
from .data import load_masks, load_tables, px_to_cm

__all__ = [
    "constants", "data", "util",
    "load_masks", "load_tables", "px_to_cm",
    "seasonality",
    "celllevel",
    "comblevel",
    "nestlevel",
    "specialization",
]
