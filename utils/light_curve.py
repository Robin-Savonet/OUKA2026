import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from math import floor, ceil
from astropy.timeseries import LombScargle
from collections import defaultdict

from utils.photometry import flux_to_mag


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

def load_night(night_dir, night_label):
    dat_path = os.path.join(night_dir, "pipelineout_datasubset.dat")

    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"No pipelineout_datasubset.dat found in:\n  {night_dir}")

    df = pd.read_csv(
        dat_path,
        sep="\t", comment="#",
        names=["index", "Label", "J.D.-2400000", "rel_flux_T1", "rel_flux_err_T1",
               "AIRMASS", "Source-Sky_T1", "Source_Error_T1"]
    )
    # No normalisation here — raw flux preserved for plot_light_curve_all_nights
    df["night"] = night_label
    return df

def _align_nights_by_phase(df, night_col, period, t_0, magnitude=False):
    """
    Instead of subtracting each night's own mean, find per-night offsets
    that minimise residuals relative to a reference night (the one with
    most data points), using phase-overlap regions.
    """
    nights = sorted(df[night_col].unique())
    phases = _compute_phase(df["J.D.-2400000"], t_0, period)
    df     = df.copy()
    df["phase"] = phases

    # Use the night with the most points as reference (offset = 0)
    ref_night = max(nights, key=lambda n: (df[night_col] == n).sum())
    offsets   = {ref_night: 0.0}

    ref_data = df[df[night_col] == ref_night].sort_values("phase")

    for night in nights:
        if night == ref_night:
            continue
        sub = df[df[night_col] == night].sort_values("phase")

        # Find phase overlap between this night and reference
        phase_min = max(sub["phase"].min(), ref_data["phase"].min())
        phase_max = min(sub["phase"].max(), ref_data["phase"].max())

        if phase_max - phase_min < 0.05:
            # No meaningful overlap — fall back to zero offset
            print(f"  WARNING: night {night} has little phase overlap with reference, offset set to 0")
            offsets[night] = 0.0
            continue

        # Interpolate reference onto this night's phase grid (overlap only)
        mask_night = (sub["phase"] >= phase_min) & (sub["phase"] <= phase_max)
        mask_ref   = (ref_data["phase"] >= phase_min) & (ref_data["phase"] <= phase_max)

        if mask_night.sum() < 2 or mask_ref.sum() < 2:
            offsets[night] = 0.0
            continue

        interp_ref = np.interp(
            sub.loc[mask_night, "phase"].values,
            ref_data["phase"].values,
            ref_data["rel_flux_T1"].values
        )
        offset = np.median(sub.loc[mask_night, "rel_flux_T1"].values - interp_ref)
        offsets[night] = offset
        print(f"  Night {night}: phase overlap [{phase_min:.2f}, {phase_max:.2f}], offset = {offset:+.4f}")

    # Apply offsets
    for night, offset in offsets.items():
        mask = df[night_col] == night
        df.loc[mask, "rel_flux_T1"] = df.loc[mask, "rel_flux_T1"] - offset

    df.drop(columns="phase", inplace=True)
    return df

def _load_mag_correction(night_dir, correction_file="mag_correction.dat"):
    """
    Load a mag_correction.dat file from a night folder if it exists.

    The file is expected to have columns:
        <index>   Date_________JDUT   mag_correction

    Returns a (jd_array, correction_array) tuple, or (None, None) if absent.
    """
    corr_path = os.path.join(night_dir, correction_file)
    if not os.path.isfile(corr_path):
        return None, None

    corr_df = pd.read_csv(
        corr_path,
        sep="\t", comment="#",
        names=["index", "Date_________JDUT", "mag_correction"],
        skiprows=1,   # skip the header line
    )
    # Drop duplicate JD entries (keep first), then sort
    corr_df = corr_df.drop_duplicates(subset="Date_________JDUT").sort_values("Date_________JDUT")
    return corr_df["Date_________JDUT"].values, corr_df["mag_correction"].values


def _apply_mag_correction(df, jd_corr, mag_corr):
    """
    Interpolate the magnitude correction onto the JD grid of df and add it
    to rel_flux_T1 (which holds magnitudes at this stage).

    Points outside the correction JD range are handled with boundary clamping
    (np.interp default). Swap for scipy interp1d if extrapolation is preferred.
    """
    interp_corr = np.interp(
        df["J.D.-2400000"].values,
        jd_corr,
        mag_corr,
    )
    df = df.copy()
    df["rel_flux_T1"] = df["rel_flux_T1"] + interp_corr
    return df


def plot_light_curve_all_nights(
        TARGET,
        target_dir='.',
        night_col="night",
        period=None,
        merge_nights=False,
        nb_plot_per_row=3,
        magnitude=False,
        filter_name=None,
        exptime=1.0,
        airmass_col="AIRMASS",
        use_phase_alignment=False,
        use_mag_correction=True,
        correction_file="mag_correction.dat",
        save_txt=True):
    """
    Automatically discovers all night folders under target_dir, loads and
    normalises each folder on the fly, and plots the light curve.

    Night folders must match YY_MM_DD or YY_MM_DD_X (where X is a lowercase
    letter suffix for multiple observations on the same night, e.g. 26_03_01_a).
    Folders with the same base date are merged into a single night panel after
    per-folder normalisation.

    If a night folder contains a mag_correction.dat file, the correction is
    interpolated onto each observation's JD and subtracted from the magnitude
    just before plotting (only active when magnitude=True).

    Parameters
    ----------
    TARGET          : str        — target name, used in titles
    target_dir      : str        — path to the folder containing night subfolders
    night_col       : str        — column name for the night label (default: 'night')
    period          : float|None — period in hours for phase folding, or None
    merge_nights    : bool       — if True and period given, overlay all nights on one plot
    nb_plot_per_row : int        — max number of panels per row (default: 3)
    magnitude       : bool       — if True, convert Source-Sky_T1 flux to calibrated magnitude
    filter_name     : str        — required if magnitude=True, one of 'B', 'V', 'R', 'I'
    exptime         : float      — exposure time in seconds used to convert ADU to flux
                                   (required if magnitude=True, default: 1.0)
    airmass_col     : str        — airmass column name (default: 'AIRMASS')
    use_mag_correction : bool   — if True (default), apply mag_correction.dat when found;
                                   set to False to skip all corrections even if files exist
    """
    if magnitude and filter_name is None:
        raise ValueError("filter_name must be provided when magnitude=True")

    # ── Discover and group night folders ──────────────────────────────────────
    night_pattern = re.compile(r"^\d{2}_\d{2}_\d{2}(_[a-z])?$")
    all_folders   = sorted([
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d)) and night_pattern.match(d)
    ])

    if not all_folders:
        raise FileNotFoundError(f"No night folders found in:\n  {target_dir}")

    # Group by base date — e.g. 26_03_01_a and 26_03_01_b → 26_03_01
    night_groups = defaultdict(list)
    for folder in all_folders:
        base = re.sub(r"_[a-z]$", "", folder)
        night_groups[base].append(folder)

    # ── Load all folders, collecting per-folder correction tables ─────────────
    # corrections: {base_date: list of (jd_array, corr_array)}
    corrections = defaultdict(list)
    dfs = []

    for base_date, folders in sorted(night_groups.items()):
        night_dfs = []
        for folder in folders:
            folder_path = os.path.join(target_dir, folder)
            df_folder   = load_night(folder_path, base_date)
            night_dfs.append(df_folder)
            print(f"  Loaded {folder}  ({len(df_folder)} rows)")

            # Try to load a magnitude correction for this sub-folder
            jd_corr, mag_corr = _load_mag_correction(folder_path, correction_file)
            if jd_corr is not None:
                corrections[base_date].append((jd_corr, mag_corr))
                print(f"    → {correction_file} found in {folder}  ({len(jd_corr)} points)")

        dfs.append(pd.concat(night_dfs, ignore_index=True))

    df     = pd.concat(dfs, ignore_index=True)
    nights = sorted(df[night_col].unique())

    # ── Magnitude conversion ───────────────────────────────────────────────────
    if magnitude:
        required = ["Source-Sky_T1", "Source_Error_T1", airmass_col]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found. Required for magnitude conversion.")

        raw_flux = df["Source-Sky_T1"].values / exptime
        raw_err  = df["Source_Error_T1"].values / exptime

        mag, mag_err = flux_to_mag(
            raw_flux, raw_err,
            filter_name,
            df[airmass_col].values,
        )
        df["rel_flux_T1"]     = mag
        df["rel_flux_err_T1"] = mag_err
        y_label  = f"Δm ({filter_name})  [mag]"
        invert_y = True

    else:
        # Normalise per night: (F − mean) / mean
        for night in nights:
            mask = df[night_col] == night
            mean = df.loc[mask, "rel_flux_T1"].mean()
            df.loc[mask, "rel_flux_T1"]     = (df.loc[mask, "rel_flux_T1"] - mean) / mean
            df.loc[mask, "rel_flux_err_T1"] =  df.loc[mask, "rel_flux_err_T1"] / abs(mean)

        y_label  = "(F − F̄) / F̄  [fractional flux]"
        invert_y = False

    # ── Apply magnitude corrections (magnitude mode only, right before plotting)
    # Interpolated by JD and added to the magnitude column, after
    # flux_to_mag so units are consistent. Nights without a correction file
    # are left untouched. Skipped entirely if use_mag_correction=False.
    if magnitude and use_mag_correction and corrections:
        for night, corr_list in corrections.items():
            mask     = df[night_col] == night
            night_df = df.loc[mask].copy()

            # Merge correction tables from multiple sub-folders for this night
            if len(corr_list) == 1:
                jd_all, corr_all = corr_list[0]
            else:
                jd_all   = np.concatenate([c[0] for c in corr_list])
                corr_all = np.concatenate([c[1] for c in corr_list])
                sort_idx = np.argsort(jd_all)
                jd_all, corr_all = jd_all[sort_idx], corr_all[sort_idx]
                _, unique_idx    = np.unique(jd_all, return_index=True)
                jd_all, corr_all = jd_all[unique_idx], corr_all[unique_idx]

            night_df = _apply_mag_correction(night_df, jd_all, corr_all)
            df.loc[mask, "rel_flux_T1"] = night_df["rel_flux_T1"].values
            print(f"  Applied mag correction to night {night}")

    # ── Global t_0: brightest point across all nights ─────────────────────────
    if period is not None:
        t_0 = df["J.D.-2400000"].iloc[
            np.argmin(df["rel_flux_T1"].values) if magnitude
            else np.argmax(df["rel_flux_T1"].values)
        ]

    # ── Zero-point alignment (replaces nightly mean subtraction) ──────────────
    if period is not None and use_phase_alignment:
        df = _align_nights_by_phase(df, night_col, period, t_0, magnitude)
    else:
        for night in nights:
            mask         = df[night_col] == night
            nightly_mean = df.loc[mask, "rel_flux_T1"].mean()
            df.loc[mask, "rel_flux_T1"] = df.loc[mask, "rel_flux_T1"] - nightly_mean

    # ── Save corrected magnitudes to .txt (all nights combined) ───────────────
    if magnitude and save_txt:
        out_path = f"{TARGET}_magnitudes.txt"
        out_data = df[["J.D.-2400000", "rel_flux_T1", "rel_flux_err_T1"]].sort_values("J.D.-2400000")
        np.savetxt(out_path, out_data.values, fmt="%.6f")
        print(f"  Saved {out_path}  ({len(out_data)} rows)")

    # ── Phase-folded single plot ───────────────────────────────────────────────
    if period is not None and merge_nights:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = plt.cm.tab10.colors
        for i, night in enumerate(nights):
            sub   = df[df[night_col] == night]
            phase = _compute_phase(sub["J.D.-2400000"], t_0, period)
            _errorbar(ax, phase, sub["rel_flux_T1"], sub["rel_flux_err_T1"],
                      color=colors[i % len(colors)], ecolor="lightgray", label=night)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(f"Phase  (period = {period} h)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"Phase-folded light curve — {TARGET} — All nights", fontsize=13, fontweight="bold")
        if invert_y:
            ax.invert_yaxis()
        _style_ax(ax)
        ax.legend(fontsize=10, title="Night")
        fig.tight_layout()
        plt.show()
        return

    # ── One subplot per night, multi-row layout ────────────────────────────────
    n          = len(nights)
    n_cols     = min(nb_plot_per_row, n)
    n_rows     = ceil(n / n_cols)
    bjd_offset = int(df["J.D.-2400000"].min())

    if n_rows == 1 and period is None:
        spans  = [df[df[night_col] == night]["J.D.-2400000"].max()
                - df[df[night_col] == night]["J.D.-2400000"].min()
                for night in nights]
        widths = [max(s, max(spans) * 0.05) for s in spans]
        gridspec_kw = {"width_ratios": widths}
    else:
        gridspec_kw = {}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5 * n_rows),
        sharey=True,
        gridspec_kw=gridspec_kw,
    )

    axes_flat = np.array(axes).flatten().tolist()

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax, night in zip(axes_flat, nights):
        sub      = df[df[night_col] == night]
        bjd      = sub["J.D.-2400000"]
        has_corr = night in corrections and magnitude and use_mag_correction

        if period is not None:
            x_data  = _compute_phase(bjd, t_0, period)
            x_label = f"Phase  (period = {period} h)"
            ax.set_xlim(-0.02, 1.02)
        else:
            x_data  = bjd - bjd_offset
            x_label = f"JD − {bjd_offset}"

        _errorbar(ax, x_data, sub["rel_flux_T1"], sub["rel_flux_err_T1"])
        ax.set_xlabel(x_label, fontsize=10)

        # Mark corrected nights in the subplot title
        title = f"{night}  ✓ corr" if has_corr else night
        ax.set_title(title, fontsize=10, fontweight="bold")
        _style_ax(ax)

        if axes_flat.index(ax) % n_cols == 0:
            ax.set_ylabel(y_label, fontsize=11)

    if invert_y:
        axes_flat[0].invert_yaxis()

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