import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def sanity_check(df):
    # ── Quick sanity check ────────────────────────────────────────────────────
    required_cols = ["J.D.-2400000", "rel_flux_T1", "rel_flux_err_T1"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}\nFound: {list(df.columns)}")

    bjd  = df["J.D.-2400000"]
    flux = df["rel_flux_T1"]
    err  = df["rel_flux_err_T1"]

    print(f"J.D.-2400000     : {bjd.min():.6f}  →  {bjd.max():.6f}")
    print(f"rel_flux_T1      : {flux.min():.6f}  →  {flux.max():.6f}")
    print(f"Median flux error: {err.median():.6f}")


def plot_light_curve(TARGET, NIGHT, bjd, flux, err, period=None, subtract_mean=False):
    """Plot a single-night light curve."""
    bjd_offset = int(bjd.min())
    if period is None:
        bjd_plot = bjd - bjd_offset
        x_label = f"J.D. - 2400000 - {bjd_offset}"
    else:
        t_0 = bjd[np.argmax(flux)]
        bjd_plot = (bjd - t_0) * 24 / period
        x_label = f"Phase (period = {period}h)"

    if subtract_mean:
        flux = flux - np.mean(flux)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        bjd_plot, flux, yerr=err,
        fmt="o", markersize=3,
        color="steelblue", ecolor="lightsteelblue",
        elinewidth=1, capsize=2, linewidth=0,
        label="rel_flux_T1"
    )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Relative Flux (rel_flux_T1)", fontsize=12)
    ax.set_title(f"Light curve - {TARGET} / {NIGHT}", fontsize=13, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_light_curve_all_nights(TARGET, df, night_col="night", period=None, merge_nights=False):
    """
    Plot all nights as separate subplots side by side, sharing the y-axis.
    Each panel covers only its own observation window — no dead space between nights.
    If period and merge_nights are provided, all nights are folded and plotted on a single graph.

    The nightly mean is always subtracted before plotting.

    Parameters
    ----------
    TARGET        : str         — target name, used in the figure title
    df            : DataFrame   — with columns J.D.-2400000, rel_flux_T1, rel_flux_err_T1, <night_col>
    night_col     : str         — column identifying each night (default: 'night')
    period        : float|None  — if given, x-axis shows phase in [0, 1] over one period (hours)
    merge_nights  : bool        — if True and period is given, overlay all nights on one plot
    """
    nights = sorted(df[night_col].unique())

    # ── Subtract nightly mean for all nights upfront ───────────────────────────
    df = df.copy()
    for night in nights:
        mask = df[night_col] == night
        df.loc[mask, "rel_flux_T1"] = df.loc[mask, "rel_flux_T1"] - df.loc[mask, "rel_flux_T1"].mean()

    # ── Global t_0: BJD of the brightest point across all nights ───────────────
    if period is not None:
        t_0 = df["J.D.-2400000"].iloc[np.argmax(df["rel_flux_T1"].values)]

    # ── Phase-folded single plot: period provided and merge_nights=True ────────
    if period is not None and merge_nights:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors

        for i, night in enumerate(nights):
            sub   = df[df[night_col] == night]
            bjd   = sub["J.D.-2400000"]
            flux  = sub["rel_flux_T1"]
            err   = sub["rel_flux_err_T1"]
            phase = ((bjd - t_0) * 24 / period) % 1

            ax.errorbar(
                phase, flux, yerr=err,
                fmt="o", markersize=3,
                color=colors[i % len(colors)],
                ecolor="lightgray",
                elinewidth=1, capsize=2, linewidth=0,
                label=night
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"Phase  (period = {period} h)", fontsize=12)
        ax.set_ylabel("Relative Flux − nightly mean", fontsize=12)
        ax.set_title(f"Phase-folded light curve — {TARGET} — All nights", fontsize=13, fontweight="bold")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)
        ax.legend(fontsize=10, title="Night")
        fig.tight_layout()
        plt.show()
        return

    # ── One subplot per night ──────────────────────────────────────────────────
    n = len(nights)

    if period is None:
        spans  = [(df[df[night_col] == night]["J.D.-2400000"].max() - df[df[night_col] == night]["J.D.-2400000"].min()) for night in nights]
        widths = [max(s, max(spans) * 0.05) for s in spans]
    else:
        widths = [1] * n   # all phase panels same width

    fig, axes = plt.subplots(
        1, n,
        figsize=(5.5 * n, 5),
        sharey=True,
        gridspec_kw={"width_ratios": widths}
    )
    if n == 1:
        axes = [axes]

    bjd_offset = int(df["J.D.-2400000"].min())

    for ax, night in zip(axes, nights):
        sub  = df[df[night_col] == night]
        bjd  = sub["J.D.-2400000"]
        flux = sub["rel_flux_T1"]
        err  = sub["rel_flux_err_T1"]

        if period is not None:
            x_data  = ((bjd - t_0) * 24 / period) % 1
            x_label = f"Phase  (period = {period} h)"
            ax.set_xlim(0, 1)
        else:
            x_data  = bjd - bjd_offset
            x_label = f"JD − {bjd_offset}"

        ax.errorbar(
            x_data, flux, yerr=err,
            fmt="o", markersize=3,
            color="steelblue", ecolor="lightsteelblue",
            elinewidth=1, capsize=2, linewidth=0
        )
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_title(night, fontsize=10, fontweight="bold")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)

        if ax is axes[0]:
            ax.set_ylabel("Relative Flux − nightly mean", fontsize=11)

    fig.suptitle(f"Light curve — {TARGET} — All observation nights", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()