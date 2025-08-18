from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .extract_hdf5_files import fill_array
from . import plots

parser = ArgumentParser(description="Plot GMOR and vector-pseudoscalar mass ratio")
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
parser.add_argument(
    "--wilson_results",
    required=True,
    help="CSV file containing data from Wilson fermions",
)
plots.add_default_input_args(parser)
args = parser.parse_args()

plots.set_styles(args)

# Extract data into a Pandas DataFrame
df = pd.read_csv(args.data, comment="#")

# Get Wilson data
wilson_data = pd.read_csv(args.wilson_results, comment="#").dropna(
    axis="index", subset=["beta"]
)

beta_colour_marker = {}
colours = [f"C{index}" for index in range(6)]
markers = ["o", "s", "h", "^", "v", "D"]

# Define the beta values
betas = sorted(set(df["beta"]))

# Create a new figure for this plot
fig, (m_ax, w_ax) = plt.subplots(figsize=(5, 7), nrows=2, sharex=True)


def get_colour_marker(beta):
    if beta in beta_colour_marker:
        return beta_colour_marker[beta]
    colour = colours.pop(0)
    marker = markers.pop(0)

    beta_colour_marker[beta] = (colour, marker)
    return colour, marker


# Loop over sorted beta-N pairs
for beta in betas:
    colour, marker = get_colour_marker(beta)
    # Filter the DataFrame for each N
    df_filtered = df[df["beta"] == beta]
    # Extract the relevant columns
    m_PS = df_filtered["g0g5"].values
    m_PS_err = df_filtered["g0g5_err"].values
    w0 = df_filtered["w_0"].values
    w0_err = df_filtered["w_0_err"].values
    # Compute x = (m_PS * w0)^2 and propagate errors
    x = (m_PS * w0) ** 2
    err_x = 2 * (m_PS * w0) * np.sqrt((m_PS * w0_err) ** 2 + (w0 * m_PS_err) ** 2)
    # Compute y = 1 / w0 and propagate errors
    y = 1 / w0
    err_y = w0_err / (w0**2)

    # Plot the data points with error bars
    m_ax.errorbar(
        x,
        y,
        xerr=err_x,
        yerr=err_y,
        ls="none",
        marker=marker,
        color=colour,
        label=f"$\\beta = {beta:.1f}$",
    )

# Labeling the plot
m_ax.set_title("Möbius fermions")
m_ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
m_ax.set_ylabel(r"$a / w_0$")
m_ax.legend(loc="best")
m_ax.grid(True, linestyle="--", alpha=0.6)
m_ax.set_ylim(0, None)

for beta in sorted(set(wilson_data["beta"])):
    colour, marker = get_colour_marker(beta)
    subset = wilson_data[wilson_data["beta"] == beta]
    y = 1 / subset["w_0"]
    err_y = subset["w_0_error"] / subset["w_0"] ** 2
    x = subset["w_0"] ** 2 * subset["g5_mass"] ** 2
    err_x = (
        x
        * (
            2 * (subset["w_0_error"] / subset["w_0"]) ** 2
            + 2 * (subset["g5_mass_error"] / subset["g5_mass"]) ** 2
        )
        ** 0.25
    )
    w_ax.errorbar(
        x,
        y,
        xerr=err_x,
        yerr=err_y,
        ls="none",
        marker=marker,
        color=colour,
        label=f"$\\beta = {beta}$",
        markerfacecolor="none",
    )

w_ax.set_title("Wilson fermions")
w_ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
w_ax.set_ylabel(r"$a / w_0$")
w_ax.legend(loc="best")
w_ax.set_ylim(0, None)
w_ax.grid(True, linestyle="--", alpha=0.6)

# Save the plot
plots.save_or_show(fig, args.output_file)
