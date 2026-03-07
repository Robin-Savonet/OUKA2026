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


def plot_light_curve(TARGET, NIGHT, bjd, flux, err):
    """Plot a single-night light curve."""
    bjd_offset = int(bjd.min())
    bjd_plot   = bjd - bjd_offset

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        bjd_plot, flux, yerr=err,
        fmt="o", markersize=3,
        color="steelblue", ecolor="lightsteelblue",
        elinewidth=1, capsize=2, linewidth=0,
        label="rel_flux_T1"
    )
    ax.set_xlabel(f"J.D.-2400000  −  {bjd_offset}", fontsize=12)
    ax.set_ylabel("Relative Flux  (rel_flux_T1)", fontsize=12)
    ax.set_title(f"Light curve  —  {TARGET}  /  {NIGHT}", fontsize=13, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_light_curve_all_nights(TARGET, df, night_col="night"):
    """
    Plot all nights as separate subplots side by side, sharing the y-axis.
    Each panel covers only its own observation window — no dead space between nights.

    Parameters
    ----------
    TARGET    : str   — target name, used in the figure title
    df        : DataFrame with columns J.D.-2400000, rel_flux_T1, rel_flux_err_T1, <night_col>
    night_col : str   — name of the column that identifies each night (default: 'night')
    """
    nights = sorted(df[night_col].unique())
    n      = len(nights)

    # Width of each panel proportional to its time span so panels look consistent
    spans  = [(df[df[night_col] == night]["J.D.-2400000"].max() - df[df[night_col] == night]["J.D.-2400000"].min()) for night in nights]
    widths = [max(s, max(spans) * 0.05) for s in spans]   # minimum width for tiny nights

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
        bjd  = sub["J.D.-2400000"] - bjd_offset
        flux = sub["rel_flux_T1"]
        err  = sub["rel_flux_err_T1"]

        ax.errorbar(
            bjd, flux, yerr=err,
            fmt="o", markersize=3,
            color="steelblue", ecolor="lightsteelblue",
            elinewidth=1, capsize=2, linewidth=0
        )
        ax.set_xlabel(f"JD − {bjd_offset}", fontsize=10)
        ax.set_title(night, fontsize=10, fontweight="bold")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)

        # Only show y-axis label on the leftmost panel
        if ax is axes[0]:
            ax.set_ylabel("Relative Flux  (rel_flux_T1)", fontsize=11)

    fig.suptitle(f"Light curve - {TARGET}  - All observation nights", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()