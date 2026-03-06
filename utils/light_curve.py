import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def sanity_check(df):
    # ── Quick sanity check ─────────────────────────────────────────────────────────

    required_cols = ["J.D.-2400000", "rel_flux_T1", "rel_flux_err_T1"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}\nFound: {list(df.columns)}")

    bjd  = df["J.D.-2400000"]
    flux = df["rel_flux_T1"]
    err  = df["rel_flux_err_T1"]

    print(f"J.D.-2400000    : {bjd.min():.6f}  →  {bjd.max():.6f}")
    print(f"rel_flux_T1: {flux.min():.6f}  →  {flux.max():.6f}")
    print(f"Median flux error: {err.median():.6f}")


def plot_light_curve(TARGET, NIGHT, bjd, flux, err):
    # ── Plot ───────────────────────────────────────────────────────────────────────

    # Offset the BJD axis to avoid long numbers on the x-axis tick labels
    bjd_offset = int(bjd.min())
    bjd_plot   = bjd - bjd_offset

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.errorbar(
        bjd_plot, flux, yerr=err,
        fmt="o",
        markersize=3,
        color="steelblue",
        ecolor="lightsteelblue",
        elinewidth=1,
        capsize=2,
        linewidth=0,
        label="rel\_flux\_T1"
    )

    ax.set_xlabel(f"J.D.-2400000  −  {bjd_offset}", fontsize=12)
    ax.set_ylabel("Relative Flux  (rel\_flux\_T1)", fontsize=12)
    ax.set_title(
        f"Light curve  —  {TARGET}  /  {NIGHT}",
        fontsize=13, fontweight="bold"
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(visible=True, which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    fig.tight_layout()
    plt.show()