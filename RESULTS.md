# Reproducing the paper's analysis figures (Figures 1–5)

This file documents what the released code reproduces from Marting et al.,
*Morphogenesis, differentiation, and robust multi-scale organization of the
honeybee nest revealed by a novel 4D atlas*, Current Biology (2026) —
figure by figure, number by number — and, just as important, what it
**cannot** reproduce from the released data, with evidence.

The analysis pipeline was originally two Jupyter notebooks by the first
author (`CombContent_Make.ipynb`, `CombContent_Analysis.ipynb`).  It now
lives in the package sub-module `combinator.content` (one module per
figure group), the notebooks are thin baked wrappers around package
calls, and every quantitative claim checked below is asserted in
`tests/test_paper_numbers.py` — so a regression in any reproduced
number fails CI (bootstrap-median checks run under the `slow` marker;
everything else is in the default fast suite).

## Figure → code map

| Figure | Panel | What it is | Code | Status |
|---|---|---|---|---|
| 1 | A | Photographs / hive-box illustration | — | drawn, no code |
| 1 | B | Image-processing overview | — | segmentation network **not released** |
| 1 | C–F | Seasonality of nest contents | `combinator.content.seasonality` | reproduced + all 5 text stats exact |
| 2 | A | Neighbourhood-composition network | `content.celllevel.neighborhood_network` | reproduced |
| 2 | B | Week-to-week transition network | `content.celllevel.transition_network` | reproduced |
| 2 | C | Cluster-metric schematic | — | drawn, no code |
| 2 | D–G | Comb/nest cluster metrics | `content.celllevel` | reproduced + all 8 text stats exact |
| 3 | A | Comb PCA (10 traits) + loadings | `content.comblevel.comb_pca`, `figure_3_comb_pca` | reproduced (PC1/PC2 = 40%/24%) |
| 3 | B | Vertical stratification by comb size | `content.comblevel` | distributions reproduced; **KS p-values not** (gap 3) |
| 3 | C | Worker/drone comb across the nest | `content.nestlevel.figure_3_combtypes_across_nest` | plot reproduced; **polarization % not** (gap 4) |
| 3 | D | Simplified contents across the nest | `content.nestlevel.figure_3_contents_across_nest` | plot reproduced; **polarization % not** (gap 4) |
| 4 | A | Cell-sequence concept schematic | — | drawn, no code |
| 4 | B | Cell content-history sequences | `content.specialization.figure_4_cell_sequences` | reproduced (slow, ~3 min) |
| 4 | C | Specialist kernel densities on the comb | — | **no released code** (gap 5) |
| 4 | D | Specialist densities: vertical/horizontal/across-comb | `content.specialization` | reproduced |
| 4 | E | Observed vs randomized specialization | `content.specialization` | reproduced (paired t-tests via `add_sig_note`) |
| 4 | F | Comb specialization grid | `content.specialization` | reproduced |
| 5 | A | Shuffling-regime schematic | — | drawn, no code |
| 5 | B | Life-stage amounts (bootstrapped) | `content.shuffled` | reproduced (medians within CI) |
| 5 | C | Egg/larva developmental failure | `content.shuffled.brood_flow` | reproduced **exactly** |
| 5 | D | Content fluctuation per comb | `content.shuffled` | reproduced |
| 5 | E | Example control vs shuffled ridgelines | `content.shuffled` | reproduced |
| 5 | F, G | Comb specialization, control vs shuffled | `content.shuffled` | reproduced + 55.9% exact |

Vector PDFs of every reproducible panel: `figures/analysis/*.pdf`
(`uv run python -m combinator.make_analysis_figures`).  Inline rendered
copies live in the baked notebooks.

## Validation against the paper text

Every number the Results text quotes for Figures 1–5, its source of
computation here, and the recomputed value (unrounded).  "Exact" means
the paper's rounded value matches the recomputation to its last printed
digit.

### Figure 1 (week 10, mean ± 95% CI over 6 control colonies)

| Paper | Recomputed | Status |
|---|---|---|
| brood 32.7% ± 7.7% | 0.327 ± 0.077 | exact |
| pollen 8.5% ± 5.5% | 0.085 ± 0.055 | exact |
| nectar & honey 14.5% ± 3.2% | 0.145 ± 0.032 | exact |
| empty 44.3% ± 8.5% | 0.443 ± 0.085 | exact |
| drone comb 11.7% ± 4.8% | 0.117 ± 0.048 | exact |

### Figure 2 (cluster metric, control colonies)

| Paper | Recomputed | Status |
|---|---|---|
| 0.803 ± 0.145 (first 10 wk) | 0.803 ± 0.145 | exact |
| honey 0.959 ± 0.066 | 0.959 ± 0.066 | exact |
| week 1 0.912 ± 0.077 | 0.912 ± 0.077 | exact |
| week 4 0.801 ± 0.145 | 0.801 ± 0.145 | exact |
| after week 4 0.785 ± 0.146 | 0.785 ± 0.146 | exact |
| p < 0.05 for 7,746 of 7,765 | 7,746 / 7,765 | exact |
| p < 0.05 for 644 of 644 (nest) | 644 / 644 | exact |
| nest metric 0.849 ± 0.097 | 0.849 ± 0.097 | exact |

### Figure 4 (specialization)

| Paper | Recomputed | Status |
|---|---|---|
| 60.9% ± 3.9% of cells specialized | 60.9% ± 3.9% | exact |
| class spectrum (of nest): nectar 10.6% ± 5.0%, pollen 11.0% ± 4.9%, brood 18.9% ± 7.7%, empty 20.4% ± 5.9%; generalists 39.1% ± 3.9% | 10.6 ± 5.0 / 11.0 ± 4.9 / 18.9 ± 7.7 / 20.4 ± 5.9 / 39.1 ± 3.9 | exact |
| comb classes: pollen 16.7% ± 8.2% | 16.7% ± 8.2% | exact |
| comb classes: 56.7% ± 17.0%, nectar 17.5% ± 5.0%, brood 32.5% ± 9.6%, empty 13.3% ± 5.8% | 58.3% ± 22.3% / 15.0% ± 15.2% / 21.7% ± 18.3% / 5.0% ± 8.4% (built-comb denominators) | **not reproduced** — gap 5 |

### Figure 5 (shuffled vs control)

| Paper | Recomputed | Status |
|---|---|---|
| control eggs 1,825 ± 162; larvae 5,109 ± 404; pupae 10,133 ± 1,259; adults 13,816 ± 1,120 | bootstrap medians 1,827 / 5,105 / 10,122 / 13,805 | within quoted CIs |
| shuffled eggs 2,517 ± 167; larvae 5,831 ± 471; pupae 10,992 ± 1,397; adults 13,806 ± 1,033 | bootstrap medians 2,518 / 5,819 / 10,978 / 13,800 | within quoted CIs |
| egg difference 686.1 ± 510 | 691 | within CI |
| failed eggs: control 2,863 ± 488, shuffled 5,407 ± 1,966 | 2,862.7 ± 488.4 / 5,407.0 ± 1,965.6 | exact |
| failed larvae: control 7,260 ± 1,003, shuffled 8,225 ± 2,256 | 7,260.2 ± 1,003.3 / 8,224.5 ± 2,255.9 | exact (8,224.5 → 8,225) |
| 88.9% higher egg failure, 13.3% higher larval failure | 88.9% / 13.3% | exact |
| fluctuation 32.7% ± 3.1% vs 46.9% ± 1.2%; difference 14% ± 5.1% | medians 32.7 / 47.0; difference 14.3 | within CI |
| 55.9% fewer specialized combs (shuffled) | 34 → 15 specialists = 55.9% | exact (see note) |

**Note on the failure counts:** the paper's absolute numbers are the
per-colony totals of failed transitions summed over the nine weekly
transitions of the first 10 weeks (total ÷ 6 colonies), computed with
the notebook's transition tracker (`brood_flow`) — this is the only
aggregation that reproduces all four mean±SD pairs to the digit.

**Note on the 55.9%:** it holds when the never-built DD6 frame-10 comb
is excluded from the control count (the notebook classifies that
phantom comb — as nectar — and only blanks it visually in Figure 4F;
including it gives 35 vs 15 = 57.1%).

## Gaps — what cannot be reproduced, and why

1. **Figure 1A–B** — photograph/illustration panels, and the neural
   network that segmented comb photographs into the `CombMasks_*.npy`
   arrays (the released data starts *after* segmentation).
2. **`df_combcounts.csv`** — the master per-frame cell-count table
   (from the masks via a pylandstats-based script) has no released
   generator.  All Figures 1–3 statistics that consume counts therefore
   start from this committed table as canonical input.
3. **Figure 3B KS p-values** ("brood–pollen p < 0.01 …") — the notebook
   plots the vertical distributions but never runs any KS test.  No
   released code computes them; likewise the Figure 4C KS tests
   (brood–pollen p < 0.0001 …).  The underlying distributions
   reproduce, the p-values cannot.
4. **Figure 3C/D polarization percentages** ("89.9% ± 8.3% of the
   nest's pollen was on one half", "54.7% ± 5.4%", "82.5% ± 32.1%") —
   no released code defines the half-split, and no candidate definition
   tried reproduces all three simultaneously (closest: ≈91% ± 11 on the
   pollen-heavy occupied half).  Computed offline; documented, not
   reimplemented.
5. **Figure 4C and the comb-class percentages** — the 2-D kernel
   density maps of specialist positions (Fig 4C) appear in the paper but
   no notebook cell draws them.  The per-class comb percentages in the
   text ("56.7% ± 17.0% of combs specialized …") do not recompute
   exactly from the released `comb_specialists` pipeline under any
   denominator tried (pollen matches to the digit, other classes differ
   by a few points), so the text was evidently produced with a variant
   of the function that was not published.
6. **Nest-level cluster tables** (`df_nearestdist_nest*.csv`) — the
   generator (`make.nest_nearest_neighbour_table`, ported for
   completeness) needs `df_mergedlandscape.csv`, the per-frame
   landscape-metrics table that is not in the release; it was also
   unseeded.  The committed tables are canonical.
7. **Drawn schematics** (Figs 2C, 4A, 5A) — no code exists or is
   needed.
8. `3d_convolution_animation.gif` in the repository has no code that
   produces or consumes it (orphan artifact, kept for provenance).

## Provenance and re-run fidelity

- The committed `df_*.csv` tables are **canonical inputs**; CI and tests
  never re-run these generators.
- `python -m combinator.make_content_tables` re-runs the released
  generators from the mask arrays:
  - `df_cellneib` / `df_cellfuture`: identical except last-bit float
    rounding (max observed relative difference ~1e-14; the committed
    files were computed on a different build), validated row-for-row in
    tests.
  - `df_nearestdist(_sum)` (seeded `np.random.seed(42)` as in the
    notebook): every deterministic column reproduces **bit-for-bit** on
    the full 16,192-row run; the Monte-Carlo columns
    (`rand_nearestneibavg`, `p_value`, `dispersion_metric`) differ
    within sampling noise because the committed random draws came from
    a different numpy build.
- `make.cell_neighbour_table` / `cell_future_table` retain the
  notebook's always-true `(colony != 'DD6') | (colony != 'DD7') |
  (colony != 'SH4')` reorientation condition verbatim — the committed
  CSVs show `frame_reoriented = 11 − frame` for every colony,
  confirming the bug was in the run that produced them; the intended
  orientation lives in `df_mergedinsp['frame_RO']` and is what the
  figures actually use.
- Dead notebook code (e.g. the always-true OR written before the nest
  polarization summary, the polarization summary frame that is
  overwritten before use) is either omitted with a NOTE comment or kept
  only where committed outputs depend on it.
- All 8 model checkpoints in `models/` are used as-is; nothing is
  re-trained anywhere in this pipeline.

## Running everything

```bash
uv pip install -e ".[analysis,dev]"          # environment
uv run pytest -q                             # fast suite (~3 min, CPU)
uv run pytest -q -m slow                     # bootstraps/figures (~20 min)
uv run python -m combinator.make_analysis_figures   # figures/analysis/*.pdf
uv run python -m combinator.make_content_tables --tables nearestdist
uv run jupyter notebook CombContent_Analysis.ipynb  # baked demo, inline figures
```
