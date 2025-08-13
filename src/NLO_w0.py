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
plots.add_default_input_args(parser)
parser.add_argument(
    "--wilson_results",
    required=True,
    help="CSV file containing data from Wilson fermions",
)
args = parser.parse_args()

plots.set_styles(args)

# Extract data into a Pandas DataFrame
data = fill_array(
    args.plateau_results,
    args.wf_results,
    args.correlator_dir_template,
    args.wf_dir_template,
)
df = pd.DataFrame(data)

# Define color cycle
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

markers = "osh^v"

# Define the beta values
betas = [6.9, 7.2, 7.4, 6.7]

# Pair betas with N values and sort by beta
beta_N_pairs = sorted(zip(betas, range(1, 5)))

# Create a new figure for this plot
fig, (m_ax, w_ax) = plt.subplots(figsize=(5, 7), nrows=2, sharex=True)

# Loop over sorted beta-N pairs
for colour_idx, ((beta, N), marker) in enumerate(zip(beta_N_pairs, markers)):
    # Filter the DataFrame for each N
    df_filtered = df[df["N"] == N]
    # Extract the relevant columns
    m_PS = df_filtered["m_PS"].values
    m_PS_err = df_filtered["m_PS_err"].values
    w0 = df_filtered["w0"].values
    w0_err = df_filtered["w0_err"].values
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
        color=f"C{colour_idx}",
        label=f"$\\beta = {beta:.1f}$",
    )

# Labeling the plot
m_ax.set_title("Möbius fermions")
m_ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")
m_ax.set_ylabel(r"$a / w_0$")
m_ax.legend(loc="best")
m_ax.grid(True, linestyle="--", alpha=0.6)
m_ax.set_ylim(0, None)

# Get Wilson data
wilson_data = pd.read_csv(args.wilson_results, comment="#").dropna(
    axis="index", subset=["beta"]
)

for colour_idx, (beta, marker) in enumerate(
    zip(sorted(set(wilson_data["beta"])), markers)
):
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
        color=f"C{colour_idx}",
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
