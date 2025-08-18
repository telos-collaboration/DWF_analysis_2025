#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .plots import save_or_show
from .plot_mres_scans import (
    get_args,
    get_scanned_columns,
    format_axes,
    PAGE_WIDTH,
    format_title,
    create_subplots,
    subset_subplot,
)


def get_only(items):
    (only,) = list(set(items))
    return only


def plot(data, scanned_columns, height):
    fig, axes = create_subplots(2, figsize=(PAGE_WIDTH, height), sharey=True)
    (column,) = scanned_columns
    betas = sorted(set(data["beta"]))
    for ax, beta in zip(axes, betas):
        subset = data[data["beta"] == beta]
        common_params = {
            key: get_only(subset[key]) for key in ["M5", "beta", "a5", "alpha", "mF"]
        }
        common_params["Ls"] = 8
        subset_subplot(ax, subset, common_params, "Ls", None)
        format_axes(ax, column)
        ax.set_title(
            format_title(common_params, list(common_params.keys()), column, columns=5),
            multialignment="center",
        )

    axes[-1].set_ylabel("")

    return fig


def main():
    args = get_args()

    data = pd.read_csv(args.data, comment="#")
    _, scanned_columns = get_scanned_columns(data, ignore=["beta"])
    save_or_show(
        plot(data, scanned_columns, args.height),
        args.output_file,
    )


if __name__ == "__main__":
    main()
