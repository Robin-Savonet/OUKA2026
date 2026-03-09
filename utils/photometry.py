"""
utils/photometry.py

Photometric utility functions for asteroid colour computation.
Magnitude conversion follows:
    m_calib = -2.5 * log10(F) - A - Z - kappa * X

where:
    F      : relative flux (instrumental)
    A      : instrument constant (TRAPPIST North: A = -25)
    Z      : photometric zero point (filter-dependent, from Farnham Table VI)
    kappa  : atmospheric extinction coefficient (filter-dependent)
    X      : airmass
"""

import numpy as np


# ── Filter parameters from Farnham Table VI ────────────────────────────────────
# Keys match the folder names: filter-B, filter-V, filter-R, filter-I
# kappa : extinction coefficient (col 7)
# ZP    : photometric zero point  (col 8)

FILTER_PARAMS = {
    "B": {"kappa": 0.25,  "ZP": 2.297},
    "V": {"kappa": 0.14,  "ZP": 2.445},
    "R": {"kappa": 0.098, "ZP": 2.122},
    "I": {"kappa": 0.043, "ZP": 2.605},
}

# Instrument constant for TRAPPIST North
A = -25.0


def flux_to_mag(flux, flux_err, filter_name, airmass, A=A):
    """
    Convert instrumental relative flux to calibrated magnitude.

        m_calib = -2.5 * log10(F) - A - Z - kappa * X

    Parameters
    ----------
    flux        : float or array — relative flux F
    flux_err    : float or array — error on flux
    filter_name : str            — one of 'B', 'V', 'R', 'I'
    airmass     : float or array — airmass X at time of observation
    A           : float          — instrument constant (default: -25 for TRAPPIST North)

    Returns
    -------
    mag     : float or array — calibrated magnitude
    mag_err : float or array — propagated magnitude error
    """
    if filter_name not in FILTER_PARAMS:
        raise ValueError(f"Unknown filter '{filter_name}'. Must be one of {list(FILTER_PARAMS.keys())}")

    Z     = FILTER_PARAMS[filter_name]["ZP"]
    kappa = FILTER_PARAMS[filter_name]["kappa"]

    flux    = np.asarray(flux,     dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    airmass = np.asarray(airmass,  dtype=float)

    mag     = -2.5 * np.log10(flux) - A - Z - kappa * airmass
    # Error propagation: dm = (2.5 / ln10) * (dF / F)
    mag_err = (2.5 / np.log(10)) * (flux_err / flux)

    return mag, mag_err


def weighted_mean(values, errors):
    """
    Compute the inverse-variance weighted mean and its propagated error.

    Parameters
    ----------
    values : array-like — measured values
    errors : array-like — associated 1-sigma errors

    Returns
    -------
    mean     : float — weighted mean
    mean_err : float — error on the weighted mean
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    weights  = 1.0 / errors**2
    mean     = np.sum(weights * values) / np.sum(weights)
    mean_err = 1.0 / np.sqrt(np.sum(weights))

    return mean, mean_err


def colour_index(mag1, err1, mag2, err2):
    """
    Compute a colour index as mag1 - mag2 with propagated error.

    Parameters
    ----------
    mag1, err1 : float — magnitude and error of the first filter
    mag2, err2 : float — magnitude and error of the second filter

    Returns
    -------
    colour     : float — mag1 - mag2
    colour_err : float — propagated error
    """
    colour     = mag1 - mag2
    colour_err = np.sqrt(err1**2 + err2**2)
    return colour, colour_err