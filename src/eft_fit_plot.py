#!/usr/bin/env python3

from argparse import ArgumentParser, FileType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import odr

from . import plots
from .eft_fit import fit_form_linear, fit_form_quadratic, homogenise_dfs


def get_args():
    parser = ArgumentParser(
        description="Fit Wilson and DWF data with appropriate EFTs and plot both data and fit result"
    )
    plots.add_default_input_args(parser)
    parser.add_argument(
        "--wilson_data",
        required=True,
        help="CSV containing spectrum and gradient flow results for Wilson fermions",
    )
    parser.add_argument(
        "--fit_result",
        default=None,
        help="CSV containing results of an EFT fit.",
    )
    plots.add_output_arg(parser)
    plots.add_styles_arg(parser)
    args = parser.parse_args()

    plots.set_styles(args)
    return args


def plot_series(ax, data, observable, filled, prop_registry, label=None):
    for beta in sorted(set(data["beta"])):
        colour, marker = prop_registry[beta]
        subset = data[data["beta"] == beta]
        formatted_label = None if label is None else label.format(beta=beta)
        ax.errorbar(
            subset["x"],
            subset[f"y_{observable}"],
            xerr=subset["x_err"],
            yerr=subset[f"y_{observable}_err"],
            color=colour,
            marker=marker,
            ls="none",
            markerfacecolor=colour if filled else "none",
            fillstyle="full" if filled else "none",
            label=formatted_label,
        )


class PropRegistry:
    def __init__(self):
        self.props = {}
        self.colours = [f"C{index}" for index in range(8)]
        self.markers = ["o", "s", "h", "^", "v", "D"]

    def __getitem__(self, key):
        if np.isnan(key):
            breakpoint()
            key = None
        if key not in self.props:
            if len(self.colours) == 0 or len(self.markers) == 0:
                breakpoint()
                raise ValueError("Not enough props!")
            self.props[key] = (self.colours.pop(0), self.markers.pop(0))

        return self.props[key]


def uncertainty_range(fit_form, x, fit_result, param_names):
    # Get parameters and uncertainties in broadcastable form
    (params,) = fit_result[param_names].values
    (statistical_uncertainties,) = fit_result[
        [f"{name}_err" for name in param_names]
    ].values
    (systematic_uncertainties,) = fit_result[
        [f"{name}_systematic_error" for name in param_names]
    ].values
    uncertainties = (statistical_uncertainties**2 + systematic_uncertainties**2) ** 0.5

    # Create an artificial set of bootstrap samples
    rng = plots.get_rng(fit_result)
    params_samples = rng.normal(
        params,
        statistical_uncertainties,
        (plots.NUM_BOOTSTRAP_SAMPLES, len(params)),
    )

    # Apply fit form to samples
    fit_samples = fit_form(x[:, np.newaxis], 0, *params_samples.swapaxes(0, 1))

    # Return 5% and 95% bounds
    return np.percentile(fit_samples, [5, 95], axis=1)


def add_fit_lines(ax, state, fit_results):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_range = np.linspace(*xlim, 1000)
    for colour, formulation in [("black", "Möbius DWF"), ("gold", "Wilson")]:
        fit_result = fit_results[
            (fit_results.state == state) & (fit_results.formulation == formulation)
        ].dropna(axis="columns")
        if len(fit_result) != 1:
            raise ValueError("Fit result not found")
        if "L1X" in fit_result.columns:
            fit_form = fit_form_quadratic
            param_names = ["w0X_squared", "L0X", "W0X", "L1X"]
        else:
            fit_form = fit_form_linear
            param_names = ["w0X_squared", "L0X", "W0X"]

        (params,) = fit_result[param_names].values

        ax.plot(
            x_range,
            fit_form(x_range, 0, *params),
            color=colour,
            label=f"{formulation} extrapolation ($a/w_0=0$)",
            linestyle="--",
        )
        ax.fill_between(
            x_range,
            *uncertainty_range(fit_form, x_range, fit_result, param_names),
            color=colour,
            alpha=0.2,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot(mobius_data, wilson_data, fit_results):
    fig, (v_ax, ps_ax) = plt.subplots(nrows=2, sharex=True, figsize=(6.5, 8.5))

    ps_ax.set_xlabel(r"$w_0^2 m_{\mathrm{PS}}^2$")
    v_ax.set_ylabel(r"$w_0^2 m_{\mathrm{V}}^2$")
    ps_ax.set_ylabel(r"$w_0^2 f_{\mathrm{PS}}^2$")

    prop_registry = PropRegistry()

    for data, fill_markers, label_substring in [
        (mobius_data, True, "Möbius DWF"),
        (wilson_data, False, "Wilson"),
    ]:
        for ax, observable, label in [
            (ps_ax, "fPS", None),
            (v_ax, "mV", f"{label_substring}, $\\beta = {{beta}}$"),
        ]:
            plot_series(ax, data, observable, fill_markers, prop_registry, label=label)

    ps_ax.set_xlim(0, None)
    ps_ax.set_ylim(0, None)

    for ax, state in [(v_ax, "mV"), (ps_ax, "fPS")]:
        add_fit_lines(ax, state, fit_results)

    v_ax.legend(loc="best")
    return fig


def main():
    args = get_args()
    mobius_data, wilson_data = homogenise_dfs(
        pd.read_csv(args.data, comment="#"),
        pd.read_csv(args.wilson_data, comment="#"),
    )
    fit_results = pd.read_csv(args.fit_result, comment="#")

    plots.save_or_show(
        plot(mobius_data, wilson_data, fit_results),
        args.output_file,
    )


if __name__ == "__main__":
    main()
