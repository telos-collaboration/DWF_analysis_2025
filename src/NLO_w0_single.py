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
    "--beta", type=int, required=True, help="Beta index of ensemble subset to use"
)
parser.add_argument(
    "--data",
    required=True,
    help="CSV file containing spectrum, gradient flow, and HMC timing results",
)
args = parser.parse_args()

plots.set_styles(args)

# Extract data into a Pandas DataFrame
df = pd.read_csv(args.data, comment="#")

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

# Filter the DataFrame for N = 1
df_filtered = df[df["beta"] == args.beta]
# Extract the relevant columns for all M
m_PS = df_filtered["g0g5"].values
m_PS_err = df_filtered["g0g5_err"].values
w0 = df_filtered["w_0"].values
w0_err = df_filtered["w_0_err"].values
# Compute x = m_PS * w0 and propagate errors
x = (m_PS * w0) ** 2  # Use x^2 instead of x
err_x = (
    2 * (m_PS * w0) * np.sqrt((m_PS * w0_err) ** 2 + (w0 * m_PS_err) ** 2)
)  # Propagate error for x^2
# Set y = w0 and its error
y = w0
err_y = w0_err


# Define the linear model function: y = a * x + b
def linear_model(x, a, b):
    return a * x + b


# Perform the fit using the entire dataset
params, covariance = curve_fit(linear_model, x, y, sigma=err_y, absolute_sigma=True)
# Extract the fitted parameters and their errors
a_fit, b_fit = params
a_err = np.sqrt(covariance[0, 0])
b_err = np.sqrt(covariance[1, 1])
# Plotting
fig, ax = plt.subplots(figsize=(7, 4))
# Plot the data points with error bars
ax.errorbar(
    x,
    y,
    xerr=err_x,
    yerr=err_y,
    fmt="o",
    color=CB_color_cycle[0],
    label="Data",
    capsize=1.5,
    elinewidth=1.5,
    markersize=6,
)
# Generate points for the fitted line
x_line = np.linspace(min(x), max(x), 1000)
y_fit = linear_model(x_line, a_fit, b_fit)
# Plot the fitted line
ax.plot(x_line, y_fit, label=f"Fit: $y = {a_fit:.2f}x + {b_fit:.2f}$", color="red")
# Calculate and plot error bands
y_fit_upper = linear_model(x_line, a_fit + a_err, b_fit + b_err)
y_fit_lower = linear_model(x_line, a_fit - a_err, b_fit - b_err)
ax.fill_between(
    x_line,
    y_fit_lower,
    y_fit_upper,
    color=CB_color_cycle[0],
    alpha=0.3,
    label="Confidence Band",
)
# Labeling the plot
ax.set_xlabel(r"$(m_{\rm PS} w_0)^2$")  # Update label for x^2
ax.set_ylabel(r"$w_0 / a$")
ax.legend(loc="best")
ax.grid(True, linestyle="--")
# Save the plot
plots.save_or_show(fig, args.output_file)
# Print the fitted parameters
print(f"Fit results: a = {a_fit:.3f} ± {a_err:.3f}, b = {b_fit:.3f} ± {b_err:.3f}")
