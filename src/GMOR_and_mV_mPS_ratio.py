import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from extract_hdf5_files import *
import matplotlib.pyplot as plt

# Paths to your CSV files
plateau_fits_path = "./CSVs/plateau_fits_results.csv"
WF_measurements_path = "./CSVs/WF_measurements.csv"
# Extract data into a Pandas DataFrame
data = fill_array(plateau_fits_path, WF_measurements_path)
df = pd.DataFrame(data)

# Define m0 values (these are the specific values you've mentioned earlier)
m0_values = np.array([0.10, 0.08, 0.06, 0.05, 0.04, 0.035])

# Define color cycle for the plots
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
plt.style.use("paperdraft.mplstyle")

# Loop over each value of N (1 to 4 in this case)
for N in range(1, 5):
    # Filter the DataFrame for N = N
    df_filtered = df[df["N"] == N]

    # Extract the relevant columns for m_PS, m_PS_err, w0, w0_err, fpi, fpi_err
    m_PS = df_filtered["m_PS"].values
    m_PS_err = df_filtered["m_PS_err"].values
    w0 = df_filtered["w0"].values
    w0_err = df_filtered["w0_err"].values
    fpi = df_filtered["fpi"].values
    fpi_err = df_filtered["fpi_err"].values

    # Compute x = m0 * w0 and propagate errors for squared (w0 * m_PS)^2
    x = (w0 * m_PS) ** 2  # Square of (w0 * m_PS)
    err_x = (
        2 * (w0 * m_PS) * np.sqrt((m_PS_err * w0) ** 2 + (w0_err * m_PS) ** 2)
    )  # Propagate error for squared values

    # Multiply x-axis by w0
    x_axis = m0_values * w0  # Modify the x-axis to be m0 * w0

    # Polynomial Fit (degree 2) for all points
    degree = 2
    initial_guess = np.ones(degree + 1)
    n_bootstrap = 1000

    bootstrap_coefs = np.zeros((n_bootstrap, degree + 1))

    for i in range(n_bootstrap):
        indices = np.random.choice(range(len(m0_values)), size=len(m_PS), replace=True)
        x_resample = m0_values[indices] * w0[indices]  # Use m0 * w0 for resampling
        y_resample = x[indices]  # Using (w0 * m_PS)^2
        err_resample = err_x[indices]

        try:
            popt, _ = curve_fit(
                lambda x, *params: np.polyval(params, x),
                x_resample,
                y_resample,
                sigma=err_resample,
                p0=initial_guess,
                absolute_sigma=True,
            )
            bootstrap_coefs[i] = popt
        except RuntimeError:
            continue

    # Filter outliers from bootstrap coefficients
    bootstrap_coefs = bootstrap_coefs[~np.all(bootstrap_coefs == 0, axis=1)]
    Q1 = np.percentile(bootstrap_coefs, 25, axis=0)
    Q3 = np.percentile(bootstrap_coefs, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_coefs = bootstrap_coefs[
        np.all(
            (bootstrap_coefs >= lower_bound) & (bootstrap_coefs <= upper_bound), axis=1
        )
    ]

    coef_mean = np.mean(filtered_coefs, axis=0)
    coef_std = np.std(filtered_coefs, axis=0)

    # Generate x values for the fit lines (using m0 * w0)
    x_fit = np.linspace(0, max(m0_values * w0), 500)
    y_fit = np.polyval(coef_mean, x_fit)
    y_fit_upper = np.polyval(coef_mean + 0.1 * coef_std, x_fit)
    y_fit_lower = np.polyval(coef_mean - 0.1 * coef_std, x_fit)

    # Fit last three points with linear function (a + b*x)
    def linear_fit(x, a, b):
        return a + b * x

    last_three_indices = np.array([3, 4, 5])  # Indices for m0 = 0.05, 0.04, 0.035
    m0_last_three = m0_values[last_three_indices]
    x_last_three = x[last_three_indices]
    err_x_last_three = err_x[last_three_indices]

    # Linear fit for the last three points
    popt_last_three, pcov_last_three = curve_fit(
        linear_fit,
        m0_last_three * w0[last_three_indices],
        x_last_three,
        sigma=err_x_last_three,
    )
    perr_last_three = np.sqrt(np.diag(pcov_last_three))

    # Generate the fit line for the last three points
    y_fit_last_three = linear_fit(x_fit, *popt_last_three)
    y_fit_last_three_upper = linear_fit(
        x_fit, *(popt_last_three + 0.1 * perr_last_three)
    )
    y_fit_last_three_lower = linear_fit(
        x_fit, *(popt_last_three - 0.1 * perr_last_three)
    )

    # Plot the results for w0 * m0 vs w0 * m_PS
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x_axis,
        x,
        yerr=err_x,
        fmt="o",
        color=CB_color_cycle[1],
        markersize=4,
        elinewidth=1.5,
        capsize=4,
        label="Data",
    )
    plt.plot(
        x_fit,
        y_fit,
        "--",
        linewidth=1.8,
        color=CB_color_cycle[0],
        label="Polynomial Fit (All Data)",
    )

    # Plot the linear fit for the last three points in green
    plt.plot(
        x_fit,
        y_fit_last_three,
        "--",
        linewidth=2,
        color="green",
        label="Linear Fit (Last Three Points)",
    )

    # Fill the confidence intervals for the full fit
    plt.fill_between(
        x_fit, y_fit_lower, y_fit_upper, color=CB_color_cycle[0], alpha=0.2
    )

    # Fill the confidence intervals for the linear fit of the last three points
    plt.fill_between(
        x_fit, y_fit_last_three_lower, y_fit_last_three_upper, color="green", alpha=0.2
    )

    # Final plot formatting
    plt.xlabel("$w_0 m_0$", fontsize=15)
    plt.ylabel("$(w_0 m_{\\rm PS})^2$", fontsize=15)
    plt.legend()
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(f"./plots/GMOR_w0m0_vs_w0m_PS_{N}.pdf", dpi=130, bbox_inches="tight")

    # Now for w0 * m0 vs w0 * f_pi, perform a similar approach
    x_fpi = w0 * fpi  # Square of (w0 * fpi)
    err_x_fpi = np.sqrt(
        (fpi_err * w0) ** 2 + (w0_err * fpi) ** 2
    )  # Propagate error for squared values

    # Linear fit for the smallest three points in the new plot (using the same method as before)
    popt_fpi_last_three, pcov_fpi_last_three = curve_fit(
        linear_fit,
        m0_last_three * w0[last_three_indices],
        x_fpi[last_three_indices],
        sigma=err_x_fpi[last_three_indices],
    )
    perr_fpi_last_three = np.sqrt(np.diag(pcov_fpi_last_three))

    # Generate the fit line for the last three points
    y_fit_fpi_last_three = linear_fit(x_fit, *popt_fpi_last_three)
    y_fit_fpi_last_three_upper = linear_fit(
        x_fit, *(popt_fpi_last_three + 0.1 * perr_fpi_last_three)
    )
    y_fit_fpi_last_three_lower = linear_fit(
        x_fit, *(popt_fpi_last_three - 0.1 * perr_fpi_last_three)
    )

    # Plot the results for w0 * m0 vs w0 * f_pi
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x_axis,
        x_fpi,
        yerr=err_x_fpi,
        fmt="o",
        color=CB_color_cycle[1],
        markersize=4,
        elinewidth=1.5,
        capsize=4,
        label="Data",
    )
    plt.plot(
        x_fit,
        y_fit_fpi_last_three,
        "--",
        linewidth=2,
        color="green",
        label="Linear Fit (Last Three Points)",
    )

    # Fill the confidence intervals for the linear fit of the last three points
    plt.fill_between(
        x_fit,
        y_fit_fpi_last_three_lower,
        y_fit_fpi_last_three_upper,
        color="green",
        alpha=0.2,
    )

    # Final plot formatting for w0 * m0 vs w0 * f_pi
    plt.xlabel("$w_0 m_0$", fontsize=15)
    plt.ylabel("$w_0 f_{\\rm PS}$", fontsize=15)
    plt.legend()
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(f"./plots/GMOR_w0m0_vs_w0fpi_{N}.pdf", dpi=130, bbox_inches="tight")

    # Now for w0 * m0 vs (w0^2 * m_PS * f_pi)^2, perform a similar approach
    x_PS_fpi = (w0**2) * m_PS * fpi  # (w0^2 * m_PS * f_pi)
    err_x_PS_fpi = np.sqrt(
        (2 * w0 * m_PS * fpi * w0_err) ** 2
        + (w0 * fpi * m_PS_err) ** 2
        + (w0 * m_PS * fpi_err) ** 2
    )  # Propagate error

    # Compute the squared values for (w0^2 * m_PS * f_pi)^2
    y_PS_fpi = x_PS_fpi**2
    err_y_PS_fpi = 2 * x_PS_fpi * err_x_PS_fpi  # Propagate error for the squared value

    # Linear fit for the smallest four points (using the same method as before)
    last_four_indices = np.array(
        [0, 1, 2, 3]
    )  # Indices for the smallest four values of m0
    m0_last_four = m0_values[last_four_indices]
    x_last_four = y_PS_fpi[last_four_indices]
    err_x_last_four = err_y_PS_fpi[last_four_indices]

    popt_PS_fpi_last_four, pcov_PS_fpi_last_four = curve_fit(
        linear_fit,
        m0_last_four * w0[last_four_indices],
        x_last_four,
        sigma=err_x_last_four,
    )
    perr_PS_fpi_last_four = np.sqrt(np.diag(pcov_PS_fpi_last_four))

    # Generate the fit line for the last four points
    y_fit_PS_fpi_last_four = linear_fit(x_fit, *popt_PS_fpi_last_four)
    y_fit_PS_fpi_last_four_upper = linear_fit(
        x_fit, *(popt_PS_fpi_last_four + 0.1 * perr_PS_fpi_last_four)
    )
    y_fit_PS_fpi_last_four_lower = linear_fit(
        x_fit, *(popt_PS_fpi_last_four - 0.1 * perr_PS_fpi_last_four)
    )

    # Plot the results for w0 * m0 vs (w0^2 * m_PS * f_pi)^2
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x_axis,
        y_PS_fpi,
        yerr=err_y_PS_fpi,
        fmt="o",
        color=CB_color_cycle[1],
        markersize=4,
        elinewidth=1.5,
        capsize=4,
        label="Data",
    )
    plt.plot(
        x_fit,
        y_fit_PS_fpi_last_four,
        "--",
        linewidth=2,
        color="green",
        label="Linear Fit (Last Four Points)",
    )

    # Fill the confidence intervals for the linear fit of the last four points
    plt.fill_between(
        x_fit,
        y_fit_PS_fpi_last_four_lower,
        y_fit_PS_fpi_last_four_upper,
        color="green",
        alpha=0.2,
    )

    # Final plot formatting for w0 * m0 vs (w0^2 * m_PS * f_pi)^2
    plt.xlabel("$w_0 m_0$", fontsize=15)
    plt.ylabel("$(w_0^2 m_{\\rm PS} f_{\\rm PS})^2$", fontsize=15)
    plt.legend()
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(
        f"./plots/GMOR_w0m0_vs_w0_mPS_fpi_{N}.pdf", dpi=130, bbox_inches="tight"
    )

    # Now for m0 vs m_V / m_PS, perform the required approach
    # Extract m_V (vector meson mass) and m_PS (pseudoscalar meson mass) from the filtered DataFrame
    m_V = df_filtered["m_V"].values
    m_PS = df_filtered["m_PS"].values

    # Define m0 vs m_V / m_PS fit function: a + b / x
    # Bootstrap resampling function

    def bootstrap_resample(x, y, yerr, n_resamples=1000):
        resampled_params = []
        for _ in range(n_resamples):
            # Resample the data (with replacement)
            indices = np.random.choice(len(x), size=len(x), replace=True)
            x_resampled = x[indices]
            y_resampled = y[indices]
            yerr_resampled = yerr[indices]

            # Fit the resampled data
            popt, _ = curve_fit(
                curve_fit_function, x_resampled, y_resampled, sigma=yerr_resampled
            )
            resampled_params.append(popt)

        return np.array(resampled_params)

    def curve_fit_function(x, a, b):
        return a + b / x

    # Propagate the error for m_V / m_PS
    m_V_m_PS = m_V / m_PS  # Compute the ratio m_V / m_PS
    err_m_V_m_PS = m_V_m_PS * np.sqrt(
        (m_PS_err / m_PS) ** 2 + (df_filtered["m_V_err"].values / m_V) ** 2
    )

    # Perform curve fitting on the original data
    popt_m0_vs_m_V_m_PS, pcov_m0_vs_m_V_m_PS = curve_fit(
        curve_fit_function, m0_values, m_V_m_PS, sigma=err_m_V_m_PS
    )

    # Generate the fit line
    x_fit_m0_vs_m_V_m_PS = np.linspace(min(m0_values), max(m0_values), 500)
    y_fit_m0_vs_m_V_m_PS = curve_fit_function(
        x_fit_m0_vs_m_V_m_PS, *popt_m0_vs_m_V_m_PS
    )

    # Perform bootstrap resampling
    bootstrap_params = bootstrap_resample(
        m0_values, m_V_m_PS, err_m_V_m_PS, n_resamples=1000
    )

    # Calculate the bootstrap confidence intervals for the parameters
    a_bootstrap = bootstrap_params[:, 0]
    b_bootstrap = bootstrap_params[:, 1]

    a_confidence_interval = np.percentile(
        a_bootstrap, [16, 50, 84]
    )  # 1-sigma confidence interval
    b_confidence_interval = np.percentile(b_bootstrap, [16, 50, 84])

    # Generate the bootstrap fit curves
    bootstrap_fit_curves = [
        curve_fit_function(x_fit_m0_vs_m_V_m_PS, a, b)
        for a, b in zip(a_bootstrap, b_bootstrap)
    ]
    bootstrap_fit_curves = np.array(bootstrap_fit_curves)

    # Calculate the upper and lower bounds for the error band
    y_lower = np.percentile(
        bootstrap_fit_curves, 16, axis=0
    )  # 16th percentile (lower bound)
    y_upper = np.percentile(
        bootstrap_fit_curves, 84, axis=0
    )  # 84th percentile (upper bound)

    # Plot the results for m0 vs m_V / m_PS with bootstrap error band
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        m0_values,
        m_V_m_PS,
        yerr=err_m_V_m_PS,
        fmt="o",
        color=CB_color_cycle[1],
        markersize=4,
        elinewidth=1.5,
        capsize=4,
        label="Data",
    )
    plt.plot(
        x_fit_m0_vs_m_V_m_PS,
        y_fit_m0_vs_m_V_m_PS,
        "--",
        linewidth=2,
        color=CB_color_cycle[0],
        label="Fit ($a + b/x$)",
    )
    plt.fill_between(
        x_fit_m0_vs_m_V_m_PS,
        y_lower,
        y_upper,
        color=CB_color_cycle[0],
        alpha=0.3,
        label="Bootstrap Error Band",
    )

    # Final plot formatting for m0 vs m_V / m_PS
    plt.xlabel("$am_0$", fontsize=15)
    plt.ylabel("$m_{\\rm V} / m_{\\rm PS}$", fontsize=15)
    plt.legend()
    plt.grid(linestyle="--")
    plt.tight_layout()
    plt.savefig(f"./plots/m0_vs_m_V_m_PS_{N}.pdf", dpi=130, bbox_inches="tight")
