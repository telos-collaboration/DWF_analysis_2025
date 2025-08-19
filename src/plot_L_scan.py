#!/usr/bin/env python3

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

from . import plots
from .plot_Ls_scan import get_args, get_title
from .plot_mres_scans import bootstrap_curve_fit


def fit_form(x, m_inf, A, B):
    return m_inf + A * np.exp(B * x)


def plot(data, fit_result):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    fit_values, fit_errors = fit_result
    m_inf = ufloat(fit_values[0], fit_errors[0])

    ax.set_title(get_title(data))
    ax.set_xlabel(r"$m_{\mathrm{inf}} L$")
    ax.set_ylabel(r"$am_{\mathrm{PS}}$")
    ax.errorbar(
        data["Nx"] * m_inf.nominal_value,
        data["g0g5"],
        data["g0g5_err"],
        marker="o",
        ls="none",
        label="Data",
    )

    xlim = ax.get_xlim()
    x_range = np.linspace(*xlim, 1000)
    ax.plot(x_range, fit_form(x_range, *fit_values), linestyle="--", label="Fit form")
    ax.axhline(
        m_inf.nominal_value,
        linestyle=":",
        color="black",
        label=r"$am_{\mathrm{inf}}=" f"{m_inf:.02uSL}$",
    )
    ax.axhspan(
        m_inf.nominal_value - m_inf.std_dev,
        m_inf.nominal_value + m_inf.std_dev,
        color="black",
        alpha=0.2,
    )
    ax.set_xlim(xlim)

    ax.legend(loc="best")

    return fig


def fit(data):
    initial_m_inf = data.sort_values(by=["Nx"])["g0g5"].iloc[-1]
    initial_popt, _ = curve_fit(
        fit_form,
        data["Nx"] * initial_m_inf,
        data["g0g5"],
        sigma=data["g0g5_err"],
        p0=[initial_m_inf, 1, -1],
    )

    return bootstrap_curve_fit(
        fit_form,
        data["Nx"] * initial_popt[0],
        data["g0g5"],
        sigma=data["g0g5_err"],
        p0=initial_popt,
    )


def main():
    args = get_args()

    data = pd.read_csv(args.data, comment="#")
    fit_params = fit(data)
    plots.save_or_show(plot(data, fit_params), args.output_file)


if __name__ == "__main__":
    main()
