"""Figures for the masked-convolutional autoregressive model.

Ports the plotting cells from the original notebook so that the three
representative outputs of the paper --

* ``control_physical_simulated_likelihood.pdf``  (Figure 6: control vs
  treatment vs simulated, with hierarchical-bootstrap CIs),
* ``3d_predictions_likelihood.png``              (Figure 6B: 3D voxel view),
* ``spatial_legend_subset.pdf``                   (Figure S: kernel legend)

-- can be regenerated from a results dict (see ``smitheebe.eval``).

``results`` is a dict keyed by a model key ``(kernel, d, h, w, pos, ctx,
int)`` whose values are the per-model output of
``smitheebe.eval.evaluate_spatial_classifier``.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from tqdm import tqdm

SIMPLE_NAMES = ['background', 'wood', 'empty_comb', 'any_brood', 'pollen',
                'nectar_honey']
DEFAULT_NAMES = ["Background", "Wood", "Empty Comb", "Brood", "Pollen",
                 "Nectar/Honey"]


# ---------------------------------------------------------------------------
# Figure 6: likelihood with bootstrap, control / physical / simulated
# ---------------------------------------------------------------------------

def _spatial_groups():
    return {
        '1D': [(True, False, False), (False, True, False), (False, False, True)],
        '2D': [(True, True, False), (True, False, True), (False, True, True)],
        '3D': [(True, True, True)],
    }


def plot_likelihood_with_bootstrap(results, model_keys, num_classes, class_names,
                                   n_bootstrap=1000, ci=95, ax=None, title="",
                                   hive_slice=None):
    """Average the mean-likelihood over all 1D/2D/3D configurations (kernel 3)
    and bootstrap 95% CIs across colonies/weeks, with a global (k=1, no-spatial)
    baseline line.  With ``hive_slice``, only hives in that slice are used
    (avoids copying the large per-hive arrays for the 3-panel Figure 6).

    Note: for the *simulated* (manually shuffled) panel the baseline and the
    spatial models are evaluated on the same shuffled hives, which is
    intentionally different from the other two panels (see original cell 42).
    """
    class_colors = sns.color_palette("colorblind", num_classes)
    class_markers = ['o', 's', '^', 'D', 'v', '*']

    spatial_groups = _spatial_groups()
    x_labels = list(spatial_groups.keys())
    x = np.arange(len(x_labels))

    def _sliced(hives):
        return hives[hive_slice] if hive_slice is not None else hives

    # Baseline (k=1, no spatial dimensions).
    baseline_key = next(k for k in model_keys
                        if k[0] == 1 and k[1:4] == (False, False, False))
    baseline_hives = _sliced(results[baseline_key]['hives'])

    def get_likelihood_matrix(hives):
        matrix = []
        for hive in hives:
            gt = hive['ground_truth']
            loglik = hive['log_likelihood']
            probs = np.exp(loglik)
            hive_vals = []
            for t in range(gt.shape[0]):
                gt_t = gt[t].flatten()
                prob_t = probs[t].reshape(-1)
                frame_vals = [prob_t[gt_t == cls].mean()
                              if np.any(gt_t == cls) else np.nan
                              for cls in range(num_classes)]
                hive_vals.append(frame_vals)
            matrix.append(np.array(hive_vals))
        return matrix

    baseline_matrix = get_likelihood_matrix(baseline_hives)
    baseline_means = np.nanmean(np.concatenate(baseline_matrix, axis=0), axis=0)
    baseline_avg = np.nanmean(baseline_means)

    means = {cls: [] for cls in range(num_classes)}
    lowers = {cls: [] for cls in range(num_classes)}
    uppers = {cls: [] for cls in range(num_classes)}

    for group_label in x_labels:
        dhw_list = spatial_groups[group_label]
        cls_samples = {cls: [] for cls in range(num_classes)}
        for dhw in dhw_list:
            key = next((k for k in model_keys
                        if k[0] == 3 and k[1:4] == dhw), None)
            if key is None:
                continue
            hives = _sliced(results[key]['hives'])
            matrix = get_likelihood_matrix(hives)
            for hive_weeks in matrix:
                for week in hive_weeks:
                    for cls in range(num_classes):
                        if not np.isnan(week[cls]):
                            cls_samples[cls].append(week[cls])
        for cls in range(num_classes):
            samples = []
            for _ in range(n_bootstrap):
                if not cls_samples[cls]:
                    samples.append(np.nan)
                    continue
                sample_vals = np.random.choice(cls_samples[cls],
                                               size=len(cls_samples[cls]),
                                               replace=True)
                samples.append(np.mean(sample_vals))
            samples = [s for s in samples if not np.isnan(s)]
            if samples:
                mu = np.mean(samples)
                lo = np.percentile(samples, (100 - ci) / 2)
                hi = np.percentile(samples, 100 - (100 - ci) / 2)
                means[cls].append(mu)
                lowers[cls].append(mu - lo)
                uppers[cls].append(hi - mu)
            else:
                means[cls].append(np.nan)
                lowers[cls].append(np.nan)
                uppers[cls].append(np.nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    else:
        fig = ax.figure

    for cls in range(num_classes):
        y = means[cls]
        yerr = [lowers[cls], uppers[cls]]
        ax.errorbar(x, y, yerr=yerr, fmt=class_markers[cls] + '-',
                    label=class_names[cls], color=class_colors[cls], capsize=4)
    ax.axhline(baseline_avg, color='black', linestyle='dashed', linewidth=1.5,
               label="Baseline")
    ax.set_ylabel("Likelihood")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylim(-0.025, 1.025)
    ax.set_title(title, pad=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    return fig, ax


def plot_control_physical_simulated(results, model_keys, num_classes, class_names,
                                    n_bootstrap=1000, ci=95, out="control_physical_simulated_likelihood.pdf"):
    """Figure 6: three panels -- control (unshuffled), physically shuffled
    treatment, and simulated (manually) shuffled.  Hive blocks in ``results``
    must be ordered [control(6), physical(6), manual(6)].  Each panel calls
    ``plot_likelihood_with_bootstrap`` with the matching hive slice (no copy of
    the large per-hive arrays is made)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.333), dpi=300, sharey=True)

    panels = [
        (slice(0, 6), "Control (Unshuffled)"),
        (slice(6, 12), "Treatment (Shuffled)"),
        (slice(12, 18), "Simulated (Shuffled)"),
    ]
    for (hive_slice, title), ax in zip(panels, axes):
        plot_likelihood_with_bootstrap(results, model_keys, num_classes,
                                       class_names, ax=ax, title=title,
                                       n_bootstrap=n_bootstrap, ci=ci,
                                       hive_slice=hive_slice)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 1.12))
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    for ax in axes:
        ax.set_xlim(-0.25, 2.25)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 6B: 3D voxel visualization
# ---------------------------------------------------------------------------

def plot_model_scatter3d_simplified(results, model_keys, model_idx=0, hive_idx=0,
                                    week=0, out="3d_predictions_likelihood.png"):
    """Figure 6B: ground truth / predictions / likelihood as 3D voxel clouds,
    with frame pairs stacked along depth and Side A/B interleaved along width."""
    simple_names = ['background', 'wood', 'empty_comb', 'any_brood', 'pollen',
                    'nectar_honey']
    num_classes = len(simple_names)
    class_palette = sns.color_palette("colorblind", num_classes)

    config_key = model_keys[model_idx]
    model_output = results[config_key]
    hive_data = model_output['hives'][hive_idx]

    preds = hive_data['predictions'][week]
    log_likelihood = hive_data['log_likelihood'][week]
    ground_truth = hive_data['ground_truth'][week]
    likelihood = np.exp(log_likelihood)

    metrics = [('ground_truth', ground_truth, 'class', 'Ground Truth Data'),
               ('predictions', preds, 'class', 'Model Predictions'),
               ('likelihood', likelihood, 'scalar', 'Model Likelihood')]

    fig = plt.figure(figsize=(10, 4), dpi=300)
    spacing = 50.0

    for i, (metric_key, data, mode, title) in enumerate(metrics):
        D, H, W = data.shape
        xs, ys, zs, cs, alphas = [], [], [], [], []
        for d in tqdm(range(D), desc=f"Collecting points for {metric_key}"):
            pair_idx = d // 2
            is_back = d % 2
            interleave_shift = -0.1 * W
            side_shift = 0
            x_offset = is_back * (W - interleave_shift)
            y_offset = pair_idx * spacing + is_back * side_shift
            for y in range(H):
                for x in range(W):
                    gt_val = int(ground_truth[d, y, x])
                    if gt_val == 0:
                        continue
                    val = data[d, y, x]
                    if mode == 'class':
                        val = int(val)
                        if val < 0 or val >= num_classes or val == 0:
                            continue
                        color = class_palette[val]
                        alpha = 1.0
                    elif metric_key == 'likelihood':
                        color = class_palette[1] if gt_val == 1 else plt.cm.viridis(val)
                        alpha = 1.0
                    xs.append(x + x_offset)
                    ys.append(y_offset)
                    zs.append(H - y)
                    cs.append(color)
                    alphas.append(alpha)

        rgba_colors = [(*c[:3], a) for c, a in zip(cs, alphas)]
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.scatter(xs, ys, zs, c=rgba_colors, s=0.33, edgecolors='none')
        ax.set_xlim(0, 2 * W + 1 - interleave_shift)
        ax.set_ylim(0, (D // 2) * spacing)
        ax.set_zlim(0, H)
        ax.set_box_aspect([2 * W + 1, (D // 2) * spacing, H])
        ax.set_title(title, fontsize=10, pad=-100)
        ax.set_xticks([51.5 - 10, 51.5 * 3 - 10 - interleave_shift])
        ax.set_xticklabels(["Side A", "Side B"], fontsize=6)
        for label in ax.get_xticklabels():
            label.set_y(-10)
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_facecolor("white")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((1, 1, 1, 0))
            if "grid" in axis._axinfo:
                axis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            if "pane" in axis._axinfo:
                axis._axinfo["pane"]['edgecolor'] = 'black'
    fig.axes[0].set_ylabel("Depth (Back -> Front)", fontsize=8)

    class_labels = ['Wood', 'Empty\nComb', 'Brood', 'Pollen', 'Nectar/\nHoney']
    class_colors = class_palette[1:6]
    cbar_class_ax = fig.add_axes([0.25, 0.1, 0.2, 0.025])
    cbar_likelihood_ax = fig.add_axes([0.55, 0.1, 0.2, 0.025])
    sm_class = ScalarMappable(cmap=ListedColormap(class_colors),
                              norm=Normalize(vmin=0, vmax=len(class_labels)))
    sm_class.set_array([])
    sm_likelihood = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=0, vmax=1))
    sm_likelihood.set_array([])
    cbar_class = plt.colorbar(sm_class, cax=cbar_class_ax, orientation='horizontal',
                              ticks=np.arange(len(class_labels)) + 0.5)
    cbar_class.ax.set_xticklabels(class_labels, fontsize=6)
    cbar_class.set_label("Class Labels", fontsize=8)
    cbar_likelihood = plt.colorbar(sm_likelihood, cax=cbar_likelihood_ax,
                                   orientation='horizontal')
    cbar_likelihood.ax.tick_params(labelsize=6)
    cbar_likelihood.set_label("Likelihood", fontsize=8)

    plt.subplots_adjust(top=0.8, hspace=0.0)
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Kernel legend
# ---------------------------------------------------------------------------

def plot_masked_kernel(ax, dims):
    kx, ky, kz = dims
    offset = [3 - (kx // 2), 3 - (ky // 2), 3 - (kz // 2)]
    for x in range(offset[0], offset[0] + kx):
        for y in range(offset[1], offset[1] + ky):
            for z in range(offset[2], offset[2] + kz):
                is_center = (x == offset[0] + kx // 2 and
                             y == offset[1] + ky // 2 and
                             z == offset[2] + kz // 2)
                facecolor = (0, 0, 0, 0.5) if is_center else mcolors.to_rgba('gray', alpha=0.1)
                ax.bar3d(x - 0.5, y - 0.5, z - 0.5, 1, 1, 1,
                         color=facecolor, edgecolor=(0, 0, 0, 0.1),
                         linewidth=0.25)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_zlim(0, 7)
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")


def plot_kernel_legend_subset(out="spatial_legend_subset.pdf"):
    """Legend showing the 1D, 2D and 3D masked 3x3x3 receptive fields."""
    kernel_shapes = [(3, 1, 1), (3, 1, 3), (3, 3, 3)]
    x_labels = ["1D", "2D", "3D"]
    n = len(kernel_shapes)
    fig = plt.figure(figsize=(1 * n, 1.3), dpi=300)
    left_margin, right_margin = 0.005, 0.995
    spacing = (right_margin - left_margin) / n
    height, bottom = 0.8, 0.12
    for i, (dims, label) in enumerate(zip(kernel_shapes, x_labels)):
        left = left_margin + i * spacing
        ax = fig.add_axes([left, bottom, spacing * 2.0, height], projection='3d')
        ax.patch.set_alpha(0.0)
        plot_masked_kernel(ax, dims)
        ax.view_init(elev=30, azim=120)
        ax.dist = 7
        ax.text(1, 6.8, 8.5, label, ha='center', va='bottom', fontsize=7.5)
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
