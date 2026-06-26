import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_parallel_coordinates(study, save_path, params=None):
    """
    Build a readable parallel-coordinates plot of an Optuna study.

    Args:
        study: a completed optuna.Study
        save_path: path (str or Path) to save the PNG to
        params: optional list of parameter names to include, in order.
                Defaults to all params found in the study, in a fixed
                sensible order if present.
    """
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df = df[df["state"] == "COMPLETE"].copy()
    if df.empty:
        raise ValueError("No completed trials to plot.")

    param_cols = [c for c in df.columns if c.startswith("params_")]
    all_names = [c.replace("params_", "") for c in param_cols]

    if params is None:
        # Sensible default order, fall back to whatever exists
        preferred = [
            "model_type", "encoder_name", "loss_criterion",
            "starting_lr", "end_lr",
            "num_cyc_frozen", "num_cyc_unfrozen",
        ]
        params = [p for p in preferred if p in all_names]
        params += [p for p in all_names if p not in params]

    n_axes = len(params)
    n_trials = len(df)

    fig, host = plt.subplots(figsize=(1.8 * n_axes + 2, 6))

    # Build one twin axis per parameter, all sharing the same x positions
    axes = [host] + [host.twinx() for _ in range(n_axes - 1)]

    # Normalised (0-1) coordinates for each trial, per axis
    coords = np.zeros((n_trials, n_axes))
    tick_info = []  # (positions, labels) per axis, for display only

    for i, p in enumerate(params):
        raw = df[f"params_{p}"]
        is_numeric = np.issubdtype(raw.dtype, np.number)

        if is_numeric:
            vmin, vmax = raw.min(), raw.max()
            if vmax == vmin:
                vmax = vmin + 1e-9
            # detect roughly log-scaled params (e.g. learning rates)
            use_log = (raw > 0).all() and (vmax / max(vmin, 1e-12)) > 50
            if use_log:
                logvals = np.log10(raw)
                lo, hi = logvals.min(), logvals.max()
                norm = (logvals - lo) / (hi - lo)
                ticks = np.linspace(lo, hi, 5)
                tick_labels = [f"{10**t:.1e}" for t in ticks]
                tick_pos = (ticks - lo) / (hi - lo)
            else:
                norm = (raw - vmin) / (vmax - vmin)
                ticks = np.linspace(vmin, vmax, 5)
                tick_labels = [f"{t:.3g}" for t in ticks]
                tick_pos = (ticks - vmin) / (vmax - vmin)
            coords[:, i] = norm.values
            tick_info.append((tick_pos, tick_labels))
        else:
            categories = sorted(raw.unique().tolist())
            cat_to_pos = {c: j / max(len(categories) - 1, 1)
                          for j, c in enumerate(categories)}
            coords[:, i] = raw.map(cat_to_pos).values
            tick_info.append(
                ([cat_to_pos[c] for c in categories], categories)
            )

    x = np.arange(n_axes)
    values = df["value"].values
    vmin, vmax = values.min(), values.max()
    cmap = plt.cm.viridis
    norm_v = (values - vmin) / max(vmax - vmin, 1e-12)

    for row in range(n_trials):
        color = cmap(norm_v[row])
        host.plot(x, coords[row], color=color, linewidth=1.8, alpha=0.85)

    host.set_xlim(0, n_axes - 1)
    host.set_ylim(0, 1)
    host.set_xticks(x)
    host.set_xticklabels(params, rotation=25, ha="right", fontsize=11,
                          color="black")

    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5)

    for i, ax in enumerate(axes):
        ax.set_facecolor("none")  # avoid double-shaded background per axis
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if i == 0:
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            ax.spines["right"].set_position(("data", i))
        pos, labels = tick_info[i]
        ax.set_yticks(pos)
        ax.set_yticklabels(
            labels, fontsize=9, color="black",
        )
        for tick_label in ax.get_yticklabels():
            tick_label.set_bbox(label_bbox)

    host.set_facecolor("#f0f0f0")
    host.grid(False)
    for xi in x:
        host.axvline(xi, color="white", linewidth=1.5, zorder=0)

    host.set_title("Parallel Coordinate Plot", fontsize=14, pad=15)

    # Reserve dedicated space on the right for the colorbar, rather than
    # letting it overlap the last parameter axis.
    fig.subplots_adjust(left=0.06, right=0.88, top=0.9, bottom=0.22)
    cbar_ax = fig.add_axes([0.91, 0.22, 0.02, 0.68])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Objective value", fontsize=10)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)