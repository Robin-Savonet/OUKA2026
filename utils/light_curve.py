import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.timeseries import LombScargle


def _subtract_mean(flux):
    """Return flux with its mean subtracted."""
    return flux - flux.mean()


def _compute_phase(bjd, t_0, period):
    """Return phase in [0, 1] given a BJD array, reference time t_0, and period in hours."""
    return ((bjd - t_0) * 24 / period) % 1


def _style_ax(ax):
    """Apply common axis styling."""
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)


def _errorbar(ax, x, flux, err, color="steelblue", ecolor="lightsteelblue", label=None):
    """Plot a standard errorbar series."""
    ax.errorbar(
        x, flux, yerr=err,
        fmt="o", markersize=3,
        color=color, ecolor=ecolor,
        elinewidth=1, capsize=2, linewidth=0,
        label=label
    )


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


def plot_light_curve(TARGET, NIGHT, bjd, flux, err, period=None, subtract_mean=True):
    """
    Plot a single-night light curve.

    Parameters
    ----------
    TARGET        : str        — target name
    NIGHT         : str        — night label
    bjd           : Series     — J.D.-2400000 values
    flux          : Series     — rel_flux_T1 values
    err           : Series     — rel_flux_err_T1 values
    period        : float|None — if given, x-axis shows phase in [0, 1] (hours)
    subtract_mean : bool       — if True (default), subtract the mean flux before plotting
    """
    flux = flux.copy()

    if subtract_mean:
        flux = _subtract_mean(flux)

    if period is None:
        bjd_offset = int(bjd.min())
        x_data  = bjd - bjd_offset
        x_label = f"J.D. - 2400000 - {bjd_offset}"
    else:
        t_0     = bjd.iloc[np.argmax(flux.values)]
        x_data  = _compute_phase(bjd, t_0, period)
        x_label = f"Phase  (period = {period} h)"

    y_label = "Relative Flux − mean" if subtract_mean else "Relative Flux (rel_flux_T1)"

    fig, ax = plt.subplots(figsize=(12, 5))
    _errorbar(ax, x_data, flux, err, label="rel_flux_T1")

    if period is not None:
        ax.set_xlim(-0.02, 1.02)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"Light curve — {TARGET} / {NIGHT}", fontsize=13, fontweight="bold")
    _style_ax(ax)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_light_curve_all_nights(TARGET, df, night_col="night", period=None, merge_nights=False):
    nights = sorted(df[night_col].unique())
    
    df = df.copy()

    # Normalise each night: (flux - nightly_mean) / nightly_mean
    # This gives fractional flux deviation, physically meaningful across nights
    for night in nights:
        mask = df[night_col] == night
        nightly_mean = df.loc[mask, "rel_flux_T1"].mean()
        df.loc[mask, "rel_flux_T1"] = (df.loc[mask, "rel_flux_T1"] - nightly_mean) / nightly_mean
        df.loc[mask, "rel_flux_err_T1"] = df.loc[mask, "rel_flux_err_T1"] / nightly_mean

    # Global t_0: BJD of the brightest point across all nights
    if period is not None:
        t_0 = df["J.D.-2400000"].iloc[np.argmax(df["rel_flux_T1"].values)]

    # ── Phase-folded single plot ───────────────────────────────────────────────
    if period is not None and merge_nights:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors
        for i, night in enumerate(nights):
            sub   = df[df[night_col] == night]
            phase = _compute_phase(sub["J.D.-2400000"], t_0, period)
            _errorbar(ax, phase, sub["rel_flux_T1"], sub["rel_flux_err_T1"],
                      color=colors[i % len(colors)], ecolor="lightgray", label=night)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"Phase  (period = {period} h)", fontsize=12)
        ax.set_ylabel("Relative Flux − global mean", fontsize=12)
        ax.set_title(f"Phase-folded light curve — {TARGET} — All nights", fontsize=13, fontweight="bold")
        _style_ax(ax)
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
        widths = [1] * n

    fig, axes = plt.subplots(
        1, n, figsize=(5.5 * n, 5),
        sharey=True, gridspec_kw={"width_ratios": widths}
    )
    if n == 1:
        axes = [axes]

    bjd_offset = int(df["J.D.-2400000"].min())
    for ax, night in zip(axes, nights):
        sub  = df[df[night_col] == night]
        bjd  = sub["J.D.-2400000"]
        if period is not None:
            x_data  = _compute_phase(bjd, t_0, period)
            x_label = f"Phase  (period = {period} h)"
            ax.set_xlim(-0.02, 1.02)
        else:
            x_data  = bjd - bjd_offset
            x_label = f"JD − {bjd_offset}"
        _errorbar(ax, x_data, sub["rel_flux_T1"], sub["rel_flux_err_T1"])
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_title(night, fontsize=10, fontweight="bold")
        _style_ax(ax)
        if ax is axes[0]:
            ax.set_ylabel("Relative Flux − global mean", fontsize=11)

    fig.suptitle(f"Light curve — {TARGET} — All observation nights", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()


def plot_lomb_scargle(TARGET, df, min_period=0.05, max_period=1.0):
    """
    Estimate the rotation period of the asteroid using the Lomb-Scargle periodogram.

    Parameters
    ----------
    TARGET      : str   — target name, used in the figure title
    df          : DataFrame with columns J.D.-2400000, rel_flux_T1, rel_flux_err_T1
    min_period  : float — minimum period to search, in days (default: 0.05  ~1.2 h)
    max_period  : float — maximum period to search, in days (default: 1.0   ~24 h)
    """
    t    = df["J.D.-2400000"].values
    flux = df["rel_flux_T1"].values
    err  = df["rel_flux_err_T1"].values

    frequency, power = LombScargle(t, flux, err).autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period
    )
    period = 1.0 / frequency

    best_period     = 1.0 / frequency[np.argmax(power)]
    rotation_period = 2.0 * best_period

    print(f"Dominant LS period : {best_period * 24:.3f} h  ({best_period:.5f} d)")
    print(f"Estimated rotation : {rotation_period * 24:.3f} h  ({rotation_period:.5f} d)  [= 2 × LS period]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(period * 24, power, color="steelblue", linewidth=0.8)
    ax.axvline(best_period * 24,     color="tomato",     linewidth=1.5, linestyle="--", label=f"LS peak: {best_period * 24:.3f} h")
    ax.axvline(rotation_period * 24, color="darkorange", linewidth=1.5, linestyle=":",  label=f"Rotation: {rotation_period * 24:.3f} h")
    ax.set_xlabel("Period (hours)", fontsize=12)
    ax.set_ylabel("Lomb-Scargle Power", fontsize=12)
    ax.set_title(f"Periodogram — {TARGET}", fontsize=12, fontweight="bold")
    _style_ax(ax)
    ax.legend(fontsize=10)

    ax = axes[1]
    phase = _compute_phase(t, t.min(), rotation_period / 24)
    _errorbar(ax, phase, flux, err)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"Phase  (period = {rotation_period * 24:.3f} h)", fontsize=12)