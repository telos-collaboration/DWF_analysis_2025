from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
import pandas as pd

from . import plots


def get_args():
    parser = ArgumentParser(
        description=(
            "Plot number of Dirac operator applications "
            "as a function of residual mass"
        )
    )
    plots.add_default_input_args(parser)
    plots.add_styles_arg(parser)
    plots.add_output_arg(parser)
    parser.add_argument("--title", default=None, help="Title for plot")
    args = parser.parse_args()

    plots.set_styles(args)
    return args


def get_colour(Ls):
    Ls_min = 0
    Ls_max = 24
    normalised_Ls = (Ls - Ls_min) / (Ls_max - Ls_min)
    return colormaps["gist_heat"](normalised_Ls)


def plot(data, title=None):
    fig, (dirac_ax, time_ax) = plt.subplot(nrows=2, figsize=(3.4, 4.5), sharex=True)

    if title:
        fig.suptitle(title)
    dirac_ax.set_ylabel("Dirac applications")
    time_ax.set_ylabel("Time [s]")
    time_ax.set_xlabel(r"$am_{\mathrm{res}}$")

    dirac_ax.grid(True, linestyle="--", alpha=0.7)
    time_ax.grid(True, linestyle="--", alpha=0.7)

    for Ls in sorted(set(data["Ls"])):
        subset = data[data["Ls"] == Ls]
        dirac_ax.errorbar(
            subset["mres"],
            subset["dslash_count"],
            xerr=subset["mres_err"],
            yerr=subset["dslash_count_error"],
            marker="o",
            ls="none",
            colour=get_colour(Ls),
            label=f"$L_s = {Ls}$",
        )
        time_ax.errorbar(
            subset["mres"],
            subset["generation_time"],
            xerr=subset["mres_err"],
            yerr=subset["generation_time_error"],
            marker="o",
            ls="none",
            colour=get_colour(Ls),
            label=f"$L_s = {Ls}$",
        )

    dirac_ax.set_ylim(0, None)
    dirac_ax.legend(loc="best")

    time_ax.set_ylim(0, None)
    time_ax.legend(loc="best")
    time_ax.set_xscale("log")

    return fig


def main():
    args = get_args()
    data = pd.read_csv(args.data, comment="#")
    plots.save_or_show(plot(data), args.output_file)


if __name__ == "__main__":
    main()
