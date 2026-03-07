import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from astropy.timeseries import LombScargle


def plot_lomb_scargle(TARGET, df, min_period=0.05, max_period=1.0):
    """
    Estimate the rotation period of the asteroid using the Lomb-Scargle periodogram.

    Asteroids have symmetric light curves (two maxima and two minima per rotation),
    so the true rotation period is twice the dominant period found by Lomb-Scargle.
    Both the raw LS period and the doubled period are shown.

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

    # Frequency grid corresponding to the requested period range
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period

    frequency, power = LombScargle(t, flux, err).autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq
    )
    period = 1.0 / frequency

    # Best period from the periodogram peak
    best_freq   = frequency[np.argmax(power)]
    best_period = 1.0 / best_freq
    # Asteroid light curves are symmetric → true rotation period = 2 × LS period
    rotation_period = 2.0 * best_period

    print(f"Dominant LS period : {best_period * 24:.3f} h  ({best_period:.5f} d)")
    print(f"Estimated rotation : {rotation_period * 24:.3f} h  ({rotation_period:.5f} d)  [= 2 × LS period]")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: periodogram in period space
    ax = axes[0]
    ax.plot(period * 24, power, color="steelblue", linewidth=0.8)
    ax.axvline(best_period * 24,     color="tomato",      linewidth=1.5,
               linestyle="--", label=f"LS peak: {best_period * 24:.3f} h")
    ax.axvline(rotation_period * 24, color="darkorange",  linewidth=1.5,
               linestyle=":",  label=f"Rotation: {rotation_period * 24:.3f} h")
    ax.set_xlabel("Period (hours)", fontsize=12)
    ax.set_ylabel("Lomb-Scargle Power", fontsize=12)
    ax.set_title(f"Periodogram  —  {TARGET}", fontsize=12, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(fontsize=10)
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)

    # Right: flux folded on the estimated rotation period
    ax = axes[1]
    phase = ((t - t.min()) % rotation_period) / rotation_period
    ax.errorbar(
        phase, flux, yerr=err,
        fmt="o", markersize=3,
        color="steelblue", ecolor="lightsteelblue",
        elinewidth=1, capsize=2, linewidth=0
    )
    ax.set_xlabel(f"Phase  (period = {rotation_period * 24:.3f} h)", fontsize=12)
    ax.set_ylabel("Relative Flux  (rel_flux_T1)", fontsize=12)
    ax.set_title(f"Phase-folded light curve  —  {TARGET}", fontsize=12, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()

    return best_period, rotation_period