#!/usr/bin/env python3

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plots


LABEL_TEXT = {
    "mF": "am_0",
    "M5": "am_5",
    "beta": r"\beta",
    "Ls": "L_s",
    "a5": "a_5/a",
    "alpha": r"\alpha",
}
PAGE_WIDTH = 7


def remove_float_columns(data):
    data = data.drop(columns=["name"])
    for column in data.columns:
        if data[column].isna().all():
            data = data.drop(columns=[column])
        for suffix in ["_error", "_err"]:
            if column.endswith(suffix):
                # Drop the error and the corresponding value column
                data = data.drop(columns=[column, column[: -len(suffix)]])
    return data


def get_scanned_columns(data, ignore=[]):
    """
    Given a Pandas DataFrame,
    identify which parameters are varied,
    and what the fixed value of parameters not being varied is.
    Returns a dict of the default parameters,
    and a list of parameter names that vary
    """

    defaults = {}
    scan_columns = []
    filtered_data = remove_float_columns(data)

    for column in filtered_data.columns:
        if column in ignore:
            continue
        counts = filtered_data[column].value_counts().reset_index()
        if len(counts) > 1:
            scan_columns.append(column)
        defaults[column] = counts[column][0]

    # Verify that non-scanned columns are fixed
    for column in scan_columns:
        if column in ignore:
            continue
        subset = filtered_data[filtered_data[column] != defaults[column]]
        for other_column in filtered_data.columns:
            if other_column == column or other_column in ignore:
                continue
            if set(subset[other_column]) != {defaults[other_column]}:
                raise ValueError("Scan doesn't seem consistent.")

    return defaults, scan_columns


def create_subplots(num_subplots, figsize=(7, 5), sharey=False):
    """
    Create an arbitrary number of subplots,
    in a consistent layout.
    """

    # Only one plot; shouldn't use the full width
    if num_subplots == 1:
        width, height = figsize
        fig, ax = plt.subplots(figsize=(width // 2 + 1, height))
        return fig, [ax]

    # Two plots; doesn't make sense to stack them, so use the full width
    if num_subplots == 2:
        return plt.subplots(ncols=2, figsize=figsize, sharey=sharey)

    # Even number of plots can be arranged in an N/2 x 2 grid
    if num_subplots % 2 == 0:
        fig, ax = plt.subplots(
            nrows=2, ncols=num_subplots // 2, figsize=figsize, sharey=sharey
        )
        return fig, ax.ravel()

    # Otherwise, we need to manually create the subplots to offset them
    fig = plt.figure(figsize=figsize)
    ax = []
    gridspec = fig.add_gridspec(2, num_subplots + 1)

    # Add first row of subplots, indexed by row_idx
    ax.append(fig.add_subplot(gridspec[0, 0:2]))
    sharey_ax = ax[0] if sharey else None
    for row_idx in range(1, num_subplots // 2 + 1):
        ax.append(
            fig.add_subplot(
                gridspec[0, row_idx * 2 : (row_idx + 1) * 2], sharey=sharey_ax
            )
        )

    # Add second row of subplots, offset horizontally by half a subplot
    for row_idx in range(num_subplots // 2):
        ax.append(
            fig.add_subplot(
                gridspec[1, row_idx * 2 + 1 : (row_idx + 1) * 2 + 1], sharey=sharey_ax
            )
        )

    return fig, ax


def format_axes(ax, label):
    ax.set_ylabel(r"$am_{\mathrm{res}}$")
    ax.set_xlabel(f"${LABEL_TEXT[label]}$")
    ax.grid(linestyle="--")
    if label in ["mF", "beta"]:
        ax.set_ylim(0, None)
    else:
        ax.set_yscale("log")


def balance_wrap(elements, elements_per_line):
    wrapped_elements = []
    for line_index in range(len(elements) // elements_per_line):
        num_left = len(elements) - line_index * elements_per_line
        start_idx = line_index * elements_per_line
        if num_left == elements_per_line + 1:
            wrapped_elements.append(
                elements[start_idx : start_idx + elements_per_line - 1]
            )
            wrapped_elements.append(elements[start_idx + elements_per_line - 1 :])
            break
        elif num_left <= 2 * elements_per_line:
            wrapped_elements.append(elements[start_idx : start_idx + elements_per_line])
            if num_left > elements_per_line:
                wrapped_elements.append(elements[start_idx + elements_per_line :])
        else:
            wrapped_elements.append(elements[start_idx : start_idx + elements_per_line])

    return wrapped_elements


def format_title(defaults, names_to_include, name_to_exclude, columns=3):
    """
    Pulls values for parameters listed in names_to_include,
    excluding name_to_exclude,
    from defaults,
    and formats them into a single string.
    """
    elements = [
        f"{LABEL_TEXT[name]} = {defaults[name]}"
        for name in names_to_include
        if name != name_to_exclude
    ]
    return "\n".join(
        f"${', '.join(line_elements)}$"
        for line_elements in balance_wrap(elements, columns)
    )


def get_subset(data, defaults, scan_column):
    main_subset = data[data[scan_column] != defaults[scan_column]]
    default_subset = data
    for column, value in defaults.items():
        default_subset = default_subset[default_subset[column] == value]

    return pd.concat([main_subset, default_subset]).sort_values(by=scan_column)


def add_fit_line(ax, subset, column, fit_form, colour="C0"):
    fit_forms = {
        "exponential": lambda x, A, B: A * np.exp(B * x),
        "linear": lambda x, A, B: A + B * x,
    }
    fit_value, fit_uncertainty = plots.bootstrap_curve_fit(
        fit_forms[fit_form],
        subset[column],
        subset["mres"],
        sigma=subset["mres_err"],
    )
    xlim = ax.get_xlim()
    x_range = np.linspace(*xlim, 1000)
    ax.plot(
        x_range,
        fit_forms[fit_form](x_range, *fit_value),
        ls="--",
        color=colour,
        label="Fit result",
    )
    ax.fill_between(
        x_range,
        fit_forms[fit_form](x_range, *(fit_value + 0.2 * fit_uncertainty)),
        fit_forms[fit_form](x_range, *(fit_value - 0.2 * fit_uncertainty)),
        color=colour,
        alpha=0.2,
    )
    ax.set_xlim(xlim)


def subset_subplot(ax, data, defaults, column, colour="C0"):
    subset = get_subset(data, defaults, column)
    scans_to_fit = {"Ls": "exponential", "beta": "linear", "mF": "linear"}
    if column in scans_to_fit:
        plot_style = {"linestyle": "none"}
    else:
        plot_style = {"linestyle": "--"}

    ax.errorbar(
        subset[column],
        subset["mres"],
        yerr=subset["mres_err"],
        label="Data",
        color=colour,
        marker=".",
        markerfacecolor="none",
        **plot_style,
    )
    if column in scans_to_fit:
        add_fit_line(ax, subset, column, scans_to_fit[column], colour=colour)
        ax.legend(loc="best")


def pad_edge(fig):
    """
    Stop labels crashing over the margin
    """
    engine = fig.get_layout_engine()
    engine.set(rect=(0.0, 0.0, 0.99, 1.0))


def plot(data, defaults, scanned_columns, height):
    fig, axes = create_subplots(len(scanned_columns), figsize=(PAGE_WIDTH, height))
    for colour_idx, (ax, column) in enumerate(zip(axes, scanned_columns)):
        subset_subplot(ax, data, defaults, column, f"C{colour_idx}")
        format_axes(ax, column)
        ax.set_title(
            format_title(defaults, scanned_columns, column), multialignment="center"
        )

    pad_edge(fig)
    return fig


def get_args():
    parser = ArgumentParser(description="Plot scans of mres against various parameters")
    plots.add_default_input_args(parser)
    plots.add_output_arg(parser)
    plots.add_styles_arg(parser)
    parser.add_argument(
        "--height",
        default=5,
        type=float,
        help="Height in inches of the plot to generate",
    )
    args = parser.parse_args()
    plots.set_styles(args)
    return args


def main():
    args = get_args()

    data = pd.read_csv(args.data, comment="#")
    default_values, scanned_columns = get_scanned_columns(data)
    plots.save_or_show(
        plot(data, default_values, scanned_columns, args.height),
        args.output_file,
    )


if __name__ == "__main__":
    main()
