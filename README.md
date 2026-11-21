# Honeybee Nest Morphogenesis — data and code

Analysis code, model weights, and data for the honeybee nest
4D atlas study in

> **Marting PR, Graving JM, et al.** *Morphogenesis, differentiation, and
> robust multi-scale organization of the honeybee nest revealed by a novel 4D
> atlas.* **Current Biology** 36, in press (2026).
> DOI: [10.1016/j.cub.2026.10.064](https://doi.org/10.1016/j.cub.2026.10.064)

This repository contains

* **`combinator/`** — a Python package for the study's 4D atlas data
  (12 colonies × weekly comb-content masks): the observational analysis
  (paper Figures 1–5) in the `combinator.content` sub-package, and the
  autoregressive masked-convolutional model with its data pipeline
  (training, evaluation, and figure generation; paper Figure 6), both
  ported from the original analysis notebooks.
* **`models/`** — the eight fitted paper model checkpoints
  (k=3 across all 7 one/two/three-axis configurations, plus the spatial
  null-model baseline, ≈62 MB total) with per-file provenance in
  [`models/MANIFEST.md`](models/MANIFEST.md).
* **`data/`** — the model's detailed 21-class colony array
  (`AllColoniesCombinedDS48v2.zip`, ~3.6 MB; zipped because the raw 156 MB
  `.npy` exceeds GitHub's 100 MB/file cap).
* **`results/`** — the full per-hive, per-voxel outputs of those eight models
  (ground truth, predictions, entropy and log-likelihood for all 18 hives:
  6 control, 6 physically-shuffled, 6 simulated; ≈400 MB in 145 small files)
  with a `manifest.json` of SHA-256 checksums.
* **`figures/`** — the model outputs of the paper (Figure 6 likelihood
  comparison, Figure 6B 3D visualization, kernel legend, and the supplemental
  per-configuration figure of all individual 1D/2D/3D contexts) plus the CSVs
  of the mean-likelihood values behind the averaged and per-configuration
  figures, all regenerated from the committed weights.
* **`combinator/content/`** — the comb-content **analysis** package
  (Figures 1–5): seasonality, cell/comb/nest-level organization,
  specialization, and shuffled-vs-control comparisons, ported module-by-module
  from the first author's notebooks with every paper number reproduced
  (see [`RESULTS.md`](RESULTS.md)).
* **`CombContent_Make.ipynb`** / **`CombContent_Analysis.ipynb`** — thin
  baked wrappers demonstrating the analysis package with all figures inline
  (consumes the deposited `CombMasks_*.npy` and `df_*.csv` files in the
  repository root; the original monolithic notebooks are in the git history).
* **`tests/`** — a pytest suite that re-derives **every number the paper
  text quotes for Figures 1–5** and fails if any drifts.
* **`figures/analysis/`** — vector PDFs of every reproducible paper panel
  (Figures 1–5), regenerated from the committed tables.
* **`autoregressive_model.ipynb`** — the model notebook, driven entirely by
  the `combinator` package: load/relabel, align, crop, a 2-epoch training
  smoke test (plus the full `train_all_v2` reproduction call behind a flag —
  the committed checkpoints are never re-trained by the notebook), evaluation
  of the eight committed checkpoints on the 18-hive set, and all paper
  figures displayed in-cell. Its stored outputs include the prediction run
  and every figure image.

---

## Installation

The environment is managed with [uv](https://github.com/astral-sh/uv)
(Python ≥ 3.11, e.g. the Python 3.12.3 used in the paper):

```bash
uv sync --all-extras     # core + comb-content notebook dependencies
# or, for just the core package — analysis + model (no OpenCV/landstats extras):
uv sync
```

`uv sync` reads `pyproject.toml` and [the lock file `uv.lock`](uv.lock), so
the paper's pins are reproduced exactly (e.g. `pytorch-lightning==2.5.0`).
On a machine with a CUDA GPU, install a CUDA-enabled PyTorch first
(or add it to the project) so training/evaluation can use the GPU:

```bash
uv sync --all-extras
uv pip install "torch>=2.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

`uv run python` (or activate `.venv`) from the repository root is all that
is needed to use the package; it is installed editable by `uv sync`.

## Analysis pipeline — Figures 1–5 (`combinator.content`)

The observational half of the paper (Figures 1–5: seasonality, cell
aggregation, comb/nest organization, specialization, and the shuffling
experiment) runs through `combinator.content`, one module per figure
group (`seasonality`, `celllevel`, `comblevel`, `nestlevel`,
`specialization`, `shuffled`), plus `make.py` — the ported generators
for the derived tables (`df_cellneib`, `df_cellfuture`,
`df_nearestdist*`).

```bash
uv run python -m combinator.make_analysis_figures   # figures/analysis/*.pdf
uv run python -m combinator.make_content_tables     # regenerate derived tables
uv run pytest -q                                    # every paper number, asserted
```

[`RESULTS.md`](RESULTS.md) maps each panel to its code, tabulates every
validated paper number against its recomputation, and lists — with
evidence — the parts of the analysis that the released data cannot
reproduce (the segmentation network behind Figure 1B, a few offline
statistics, and the landscape table behind the nest-level tables).  CI
(`.github/workflows/tests.yml`) runs the fast suite on every push.

## Quick start — the model (Figure 6)

All of the following runs on CPU; add `CUDA_VISIBLE_DEVICES=<n>` to use a
GPU (selecting the device with `torch.cuda.device_count()`-style indexing,
e.g. run with `CUDA_VISIBLE_DEVICES=1` to pin a specific card).

```python
import numpy as np
from combinator import load_grids, map_all_labels, process_grids, build_eval_hives
from combinator import evaluate_spatial_classifier
from combinator.models import load_any

# 1) Load the detailed 21-class masks and run the paper's spatial pipeline
#    (class remap -> phase-correlation alignment to a global mean wood
#    template -> mean-wood bounding-box crop -> (20, 52, 104) grids of the
#    6 simplified classes).
grids = load_grids("data/AllColoniesCombinedDS48v2.zip")      # per-colony stack (zipped .npy)
grids = map_all_labels(grids)
grids = process_grids(grids)            # takes ~30 s on CPU

# 2) Build the 18-hive evaluation corpus: 6 control + 6 physically
#    shuffled + 6 simulated (frame-pair-shuffled) hives.
hives = build_eval_hives(grids, num_control=6)

# 3) Load a paper model (filename encodes its architecture) and evaluate it.
model = load_any("models/spatial_classifier_v2_k3_dTrue_hTrue_wTrue_posTrue_ctxTrue_int1.pth")
results = evaluate_spatial_classifier(model, hives, device="cpu")
print(results["hives"][0]["predictions"][0].argmax())  # ...
```

Reproduce the committed figures and CSV from the committed weights
(default palette: the published model-figure colors — seaborn
"colorblind" with the wood class in red; `--palette seaborn` uses the
untouched seaborn palette, `--palette content` the nest-content palette
of Figures 1–5):

```bash
uv run python -m combinator.make_figures \
    --data data/AllColoniesCombinedDS48v2.zip \
    --models-dir models --outdir figures_out
CUDA_VISIBLE_DEVICES=1 uv run python -m combinator.make_figures \
    --data ... --device cuda:0   # GPU
```

Reconstruct the committed `results/` directory into a plain dict in memory
(with SHA-256 verification):

```python
from combinator.results_io import load_results
results = load_results("results")   # {config_tuple: {"hives": [...]}}
```

Validate an unpacked `results/` copy at any time:

```bash
uv run python -m combinator.results_io verify results
```

## Data and provenance

The model is trained and evaluated on the **detailed** per-colony grids.
These are produced by the comb-image segmentation pipeline and (in their
raw, un-aligned form) are large, so the model input is committed **zipped**
under [`data/`](data/) — `zip -9` reduces the 156 MB `.npy` to ~3.6 MB.
`load_grids()` accepts a `.zip` containing a single `.npy` directly, so no
download or unzip step is needed.

| File | Shape per colony (weeks×D×H×W) | Classes | Where | Purpose |
|---|---|---|---|---|
| `AllColoniesCombinedDS48v2.npy` (zipped) | (weeks, 20, 77, 116) | detailed (0–19, 255) | in this repo, as `data/AllColoniesCombinedDS48v2.zip` | **model** input; the 21-class DS48v2 masks (12 colonies, 228 weeks). Remapped to 6 classes, aligned, and cropped by the pipeline to (weeks, 20, 52, 104). The raw file is 156 MB (over GitHub's 100 MB/file cap), so it is committed zipped; `load_grids()` reads the zip directly |
| `CombMasks_Contents_aligned.npy` | (weeks, 20, 52, 100) | 0–10 | in this repo | deposited; comb-**content** masks (already aligned) |
| `CombMasks_CombTypes_aligned.npy` | (weeks, 20, 52, 100) | 0–5 | in this repo | deposited; comb-**type** labels (already aligned) |
| `CombMasks_Combined_aligned.npy` | (weeks, 20, 52, 100) | 0–19 | in this repo | deposited; content + type combined (already aligned) |
| `CombMasks_Contents_simple.npy` | (weeks, 20, 77, 116) | 0–4, 255 | in this repo | deposited; simplified content masks, unaligned |
| `CombMasks_Contents_crop.npy` | (weeks, 20, 42, 88) | 0–10 | in this repo | deposited; content masks, a tighter crop |
| `df_*.csv` | — | — | in this repo | per-cell / per-colony statistics derived from the masks, consumed by `CombContent_Analysis.ipynb` |
| `models/*.pth` | — | — | in this repo | eight fitted checkpoints (see [`models/MANIFEST.md`](models/MANIFEST.md)) |
| `results/*.npz` | — | — | in this repo | per-hive model outputs (see `results/manifest.json`) |

The two families should not be conflated: the **model** works on the
detailed DS48v2 masks (local). The **deposited** `CombMasks_*.npy` and
`df_*.csv` are the products of the comb-content analysis (`*_aligned` are
already aligned to a 52×100 bounding box; `*_simple`/`*_crop` are the
pre-alignment forms) and are the inputs to the two comb-content notebooks.
They carry different class taxonomies and are not the model's training input.

### Note on grid geometry

The positional-embedding weights of the committed checkpoints fix the model
input geometry to **(depth, height, width) = (20, 52, 104)**: 20 = 10
combined frame-sides (Side A + Side B of each of the 10 frames, stored
interleaved), and 52 × 104 is the mean-wood bounding-box crop (margin 1,
threshold 0.5) of the aligned 77 × 116 comb image — exactly what
`combinator.preprocessing.process_grids` computes. The weights (e.g. the
`positional_embedding` shape `(1, 128, 20, 52, 104)`) are authoritative and
the code matches them exactly; loading is strict (`weights_only=True`,
`load_state_dict` strict).

## Model

`SpatialClassifier` (see `combinator/models.py` and the first-author
notebook it ports) is a per-voxel autoregressive content classifier:

* **Class embedding**: a learned (classes × 128) table maps each voxel's
  integer label to a 128-d vector; a per-class bias is added.
* **Center-masked conv**: a single linear convolution with kernel
  (k, k, k), with `k` = 3 on *included* axes and 1 on excluded axes, whose
  center voxel is hard-zeroed so the true label never appears in the input.
* **Positional bias**: a learnable (1, 128, D', H', W') field, full size on
  included axes and 1 on excluded axes, broadcast over the grid.
* **First-order interaction**: a learned "context × position" term — two
  1×1×1 pointwise convolutions produce two feature vectors that are
  multiplied element-wise and added back to `context + position`
  (i.e. `output = context + position + context2 + (context * position)`).
* **Tied head**: logits = `output @ class_embedding.T`, softmaxed to a
  distribution over the 6 classes.

Training (paper, section "Masked-convolutional autoregressive model"):
AdamW (base lr 1e-3, weight decay 1e-5, betas (0.9, 0.95)); the class
embedding and positional-bias groups run at 10× the base lr; a 5-epoch
linear warmup then cosine annealing, 200 epochs total; batch size 8;
class-weighted cross-entropy with inverse-frequency weights.

### The 8 committed checkpoints

| File (see `models/` for full names) | Role in the paper |
|---|---|
| `..._v2_k1_..._int0` | Spatial null model (k=1, no axes) — the "no context" baseline (dashed line) |
| 3 × 1D: w, h, d | Receptive field on a single axis |
| 3 × 2D: d+h, d+w, h+w | Two axes |
| 1 × 3D: d+h+w | Full 3-D context — the main model |

Figure 6 averages the 1-D and 2-D configurations separately and shows the
3-D model and the k=1 baseline. The values for the *individual* 1D/2D/3D
configurations (the paper's supplemental per-configuration figure) are in
**`figures/mean_likelihood_per_config.csv`**, plotted in
**`figures/supplement_config_likelihood.pdf`** (mean-likelihood per
configuration and treatment panel, with the pure-chance baseline at
1/num\_classes); **`figures/mean_likelihood_figure6.csv`** holds the
dimension-averaged Figure-6 numbers.

## What "shuffling" means

* **Control (shuffled = "no")**: the 6 `DD*` colonies, frames in natural
  order.
* **Treatment (shuffled = "physical")**: the 6 `SH*` colonies, whose frames
  were physically moved within the hive during the experiment.
* **Simulated**: `DD1`…`DD6` with the ten frame-pairs (A+B) along the
  depth axis randomly permuted, breaking global depth order while
  keeping each pair of a frame together. Built with
  `build_eval_hives(..., seed=42)`, deterministic.

## Reproducibility checklist

* `pyproject.toml` / `uv.lock` pin the software stack (Python ≥ 3.11;
  e.g. `pytorch-lightning==2.5.0`, the version used in the paper).
* `models/MANIFEST.md` lists the hyper-parameters and training seed context
  of each committed checkpoint.
* `results/manifest.json` lists every result file with its SHA-256, so a
  downloaded copy can be validated byte-for-byte.
* `python -m combinator.make_figures --data data/AllColoniesCombinedDS48v2.zip
  --models-dir models --outdir figures` reproduces `figures/` from the
  committed weights plus the data, and `autoregressive_model.ipynb` walks
  through the same pipeline as the package API, cell by cell.
* `python -m combinator.make_analysis_figures` reproduces every
  reproducible panel of Figures 1–5 into `figures/analysis/`, and
  `python -m combinator.make_content_tables` regenerates the derived
  analysis tables from the comb masks (regeneration fidelity and the
  non-regenerable parts: [`RESULTS.md`](RESULTS.md)).
* `pytest` asserts all paper numbers of Figures 1–5; `.github/workflows/
  tests.yml` runs lint + tests on every push.

## License

MIT (see `LICENSE`). Code and weights; data files retain their original
provenance and sharing terms (see the table above).
