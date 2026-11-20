"""smitheebe -- autoregressive masked-convolutional model of honeybee nest contents.

Code and model weights for the masked-convolutional autoregressive model
described in Marting et al., "Morphogenesis, differentiation, and robust
multi-scale organization of the honeybee nest revealed by a novel 4D atlas",
Current Biology (2026).

See README.md for setup and usage, and models/MANIFEST.md for the provenance
of each committed checkpoint.
"""

from .dataset import load_grids
from .eval import build_eval_hives, evaluate_spatial_classifier
from .labels import map_all_labels
from .preprocessing import process_grids

__version__ = "0.1.0"

__all__ = [
    "build_eval_hives",
    "evaluate_spatial_classifier",
    "load_grids",
    "map_all_labels",
    "process_grids",
]
