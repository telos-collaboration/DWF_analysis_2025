# python3 time_elapsed_autocorr_plaquette.py 1 hmc_51489.out hmc_51189.out hmc_51497.out hmc_51498.out hmc_51516.out hmc_48102.out hmc_51081.out hmc_51083.out hmc_51147.out hmc_50967.out hmc_51148.out hmc_51168.out hmc_49601.out hmc_51204.out hmc_51193.out hmc_51205.out hmc_51203.out hmc_49306.out hmc_49328.out hmc_49333.out hmc_49346.out hmc_49347.out hmc_49352.out hmc_49969.out
# TODO test and fix

from argparse import ArgumentParser, FileType

from collections import defaultdict
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from . import plots


def get_args():
    parser = ArgumentParser(description="Compute elapsed time for ensembles and plot")
    plots.add_styles_arg(parser)
    plots.add_output_arg(parser)
    parser.add_argument(
        "--data",
        required=True,
        help="CSV file containing spectrum, gradient flow, and HMC timing results",
    )
    args = parser.parse_args()

    plots.set_styles(args)
    return args


def fit_form_cost(x, C, A, B, chiral_w0):
    return C * (chiral_w0**A) / (x**B)


def fit_cost(data, chiral_w0s):
    flat_chiral_w0s = [chiral_w0s[beta] for beta in data["beta"]]
    return curve_fit(
        lambda x, C, A, B: fit_form_cost(x, C, A, B, flat_chiral_w0s),
        data["x"],
        data["y"],
        sigma=data["y_err"],
        p0=[250, 1, 3],
    )


def w0_fit_form(mass, A, B, C):
    return A + B * mass + C * mass**2


def get_chiral_w0(data, beta):
    subset = data[data["beta"] == beta].dropna(
        axis="index", subset=["g0g5", "w_0", "w_0_error"]
    )
    popt, pcov = curve_fit(
        w0_fit_form, subset["g0g5"], subset["w_0"], sigma=subset["w_0_error"]
    )
    return popt[0]


def add_derived_columns(data):
    data["x"] = data["g0g5"] / data["gi"]
    data["x_err"] = (
        data["x"]
        * ((data["g0g5_err"] / data["g0g5"]) ** 2 + (data["gi_err"] / data["gi"]) ** 2)
        ** 0.5
    )
    data["y"] = data["num_gpus"] * data["plaquette_autocorr"] * data["generation_time"]
    data["y_err"] = (
        data["y"]
        * (
            (data["plaquette_autocorr_error"] / data["plaquette_autocorr"]) ** 2
            + (data["generation_time_error"] / data["generation_time"]) ** 2
        )
        ** 0.5
    )


def plot(data, chiral_w0s, fit_params):
    markers = "os^v"
    fig, ax = plt.subplots(figsize=(4, 3))
    x_smooth = np.linspace(min(data["x"]) - 0.5, max(data["x"]) + 0.5, 1000)

    for colour_index, ((beta, chiral_w0), marker) in enumerate(
        zip(chiral_w0s.items(), markers)
    ):
        subset = data[data["beta"] == beta]

        # Plot data with error bars
        ax.errorbar(
            subset["x"],
            subset["y"],
            xerr=subset["x_err"],
            yerr=subset["y_err"],
            fmt=marker,
            color=f"C{colour_index}",
            label=f"$\\beta={beta}$",
        )

        # Plot fitted curve
        y_fit = fit_form_cost(x_smooth, *fit_params, chiral_w0)
        ax.plot(x_smooth, y_fit, color=f"C{colour_index}", linestyle="--")

    ax.set_xlabel(r"$m_{\rm PS}/m_{\rm V}$")
    ax.set_ylabel(
        r"$N_{\rm GPU}\cdot\tau_{\rm \langle P \rangle}\cdot \rm{time} \,\, [GPUsec]$"
    )
    ax.legend(loc="best")
    ax.set_yscale("log")  # Log scale for better visualization if needed
    ax.set_xscale("log")  # Log scale for better visualization if needed
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim(min(data["x"]) - 0.05, max(data["x"]) + 0.05)
    ax.set_ylim(2000, 50000)

    return fig


def main():
    args = get_args()
    data = pd.read_csv(args.data, comment="#")
    betas = sorted(set(data["beta"]))
    chiral_w0s = {beta: get_chiral_w0(data, beta) for beta in betas}
    add_derived_columns(data)
    fit_params, _ = fit_cost(data, chiral_w0s)
    plots.save_or_show(plot(data, chiral_w0s, fit_params), args.output_file)


if __name__ == "__main__":
    main()
