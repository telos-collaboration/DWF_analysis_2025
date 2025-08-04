import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from extract_hdf5_files import fill_array

# Paths to your CSV files
plateau_fits_path = "./CSVs/plateau_fits_results.csv"
WF_measurements_path = "./CSVs/WF_measurements.csv"
# Extract data into a Pandas DataFrame
data = fill_array(plateau_fits_path, WF_measurements_path)
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
# Style for plotting
plt.style.use("paperdraft.mplstyle")
for N in range(1, 5):
    # Filter the DataFrame for N = 1
    df_filtered = df[df["N"] == N]
    # Extract the relevant columns for all M
    m_PS = df_filtered["m_PS"].values
    m_PS_err = df_filtered["m_PS_err"].values
    w0 = df_filtered["w0"].values
    w0_err = df_filtered["w0_err"].values
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
    plt.figure(figsize=(7, 4))
    # Plot the data points with error bars
    plt.errorbar(
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
    x_line = np.linspace(min(x), max(x), 100)
    y_fit = linear_model(x_line, a_fit, b_fit)
    # Plot the fitted line
    plt.plot(x_line, y_fit, label=f"Fit: y = {a_fit:.2f}x + {b_fit:.2f}", color="red")
    # Calculate and plot error bands
    y_fit_upper = linear_model(x_line, a_fit + a_err, b_fit + b_err)
    y_fit_lower = linear_model(x_line, a_fit - a_err, b_fit - b_err)
    plt.fill_between(
        x_line,
        y_fit_lower,
        y_fit_upper,
        color=CB_color_cycle[0],
        alpha=0.3,
        label="Confidence Band",
    )
    # Labeling the plot
    plt.xlabel(r"$(m_{\rm PS} w_0)^2$", fontsize=14)  # Update label for x^2
    plt.ylabel(r"$w_0 / a$", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--")
    # Save the plot
    plt.savefig(f"./plots/NLO_mPS_w0_beta_{N}.pdf", dpi=130, bbox_inches="tight")
    # Print the fitted parameters
    print(f"Fit results: a = {a_fit:.3f} ± {a_err:.3f}, b = {b_fit:.3f} ± {b_err:.3f}")


# Define colors for the four N values
colors = ["blue", "green", "red", "purple"]

# Create a new figure for this plot
plt.figure(figsize=(8, 6))

# Define colors for the four N values
colors = ["blue", "green", "red", "purple"]

# Define the beta values
betas = [6.9, 7.2, 7.4, 6.7]

# Pair betas with N values and sort by beta
beta_N_pairs = sorted(zip(betas, range(1, 5)))

# Create a new figure for this plot
plt.figure(figsize=(8, 6))

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
    plt.errorbar(
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
plt.xlabel(r"$(m_{\rm PS} w_0)^2$", fontsize=14)
plt.ylabel(r"$a / w_0$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
# Save the plot
plt.savefig("./plots/chiral_aoverw0_vs_mPS.pdf", dpi=130, bbox_inches="tight")
