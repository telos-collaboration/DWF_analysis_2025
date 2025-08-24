#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

from . import plots


def get_title(data):
    betas = set(data["beta"])
    masses = set(data["mF"])

    elements = []
    if len(betas) == 1:
        (beta,) = betas
        elements.append(f"$\\beta = {beta}$")
    if len(masses) == 1:
        (mass,) = masses
        elements.append(f"$am_0 = {mass}$")

    return ", ".join(elements)


def plot(data):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.set_title(get_title(data))
    ax.set_xlabel("$L_s$")
    ax.set_ylabel(r"$am_{\mathrm{PS}}$")
    ax.errorbar(data["Ls"], data["g0g5"], data["g0g5_err"], marker="o", ls="none")

    inset = ax.inset_axes([0.52, 0.52, 0.45, 0.45])
    inset.errorbar(
        data["Ls"].iloc[-3:],
        data["g0g5"].iloc[-3:],
        data["g0g5_err"].iloc[-3:],
        marker="o",
        ls="none",
    )
    return fig


def main():
    args = plots.get_args("Plot scans of mPS against Ls")

    data = pd.read_csv(args.data, comment="#").sort_values(by=["Ls"])
    plots.save_or_show(plot(data), args.output_file)


if __name__ == "__main__":
    main()
