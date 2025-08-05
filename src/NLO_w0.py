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

# Define colors for the four N values
colors = ["blue", "green", "red", "purple"]

# Define the beta values
betas = [6.9, 7.2, 7.4, 6.7]

# Pair betas with N values and sort by beta
beta_N_pairs = sorted(zip(betas, range(1, 5)))

# Create a new figure for this plot
fig, ax = plt.subplots(figsize=(8, 6))

# Loop over sorted beta-N pairs
for (beta, N), color in zip(beta_N_pairs, colors):
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
    ax.errorbar(
        x,
        y,
        xerr=err_x,
        yerr=err_y,
        fmt="o",
        color=color,
        label=f"$\\beta = {beta:.1f}$",
        capsize=2,
        elinewidth=1.5,
        markersize=6,
    )

# Labeling the plot
ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$", fontsize=14)
ax.set_ylabel(r"$a / w_0$", fontsize=14)
ax.legend(fontsize=12, loc="best")
ax.grid(True, linestyle="--", alpha=0.6)
# Save the plot
plots.save_or_show(fig, args.output_file)
