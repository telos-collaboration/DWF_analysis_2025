# python3 time_elapsed_autocorr_plaquette.py 1 hmc_51489.out hmc_51189.out hmc_51497.out hmc_51498.out hmc_51516.out hmc_48102.out hmc_51081.out hmc_51083.out hmc_51147.out hmc_50967.out hmc_51148.out hmc_51168.out hmc_49601.out hmc_51204.out hmc_51193.out hmc_51205.out hmc_51203.out hmc_49306.out hmc_49328.out hmc_49333.out hmc_49346.out hmc_49347.out hmc_49352.out hmc_49969.out

import re
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("paperdraft.mplstyle")

# Check if at least one input file is provided
if len(sys.argv) < 3:
    print("Usage: python script.py <num_to_skip> <input_file1> [<input_file2> ...]")
    sys.exit(1)

# Parse arguments
num_to_skip = int(sys.argv[1])  # Number of configurations to skip
input_files = sys.argv[2:]  # List of input files

# Arrays to store jackknife means and errors
jackknife_means_all = []
jackknife_stds_all = []

# Arrays to store autocorrelation times and errors
tau_autocorr_all = []
tau_autocorr_err_all = []


def autocorrelation_function(data, max_lag=None):
    """Compute the normalized autocorrelation function using FFT."""
    N = len(data)
    mean = np.mean(data)
    fluctuations = data - mean

    # Compute autocorrelation using FFT
    fft_corr = np.fft.ifft(np.abs(np.fft.fft(fluctuations, n=2 * N)) ** 2).real[:N]
    fft_corr /= fft_corr[0]  # Normalize

    if max_lag is None:
        max_lag = N // 2  # Default window

    return fft_corr[:max_lag]


def gamma_method_with_error(data):
    """Compute the integrated autocorrelation time and its error."""
    N = len(data)

    # Compute the autocorrelation function
    autocorr = autocorrelation_function(data)

    # Initialize the sum
    tau_int = 0.5  # First term in sum
    W = 0  # Summation window

    # Dynamically choose the cutoff window W
    for t in range(1, len(autocorr)):
        tau_int += autocorr[t]
        W = t
        # Stop summing if the autocorrelation noise dominates
        if 2 * tau_int < t:
            break

    # Compute the error estimate using Wolff’s formula
    tau_error = tau_int * np.sqrt(2 * (W + 1) / N)

    return tau_int, tau_error


# Process each file separately
for file_idx, input_file in enumerate(input_files):
    numbers_file = f"time_numbers_{file_idx}.log"
    differences_file = f"time_differences_{file_idx}.log"

    # Regular expression pattern to match the Plaquette line and extract the number
    pattern = re.compile(r"Plaquette: \[ \d+ \] ([\d.]+)")

    # List to store extracted numbers
    extracted_numbers = []

    # Read the file and extract numbers
    with open(input_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                extracted_numbers.append(float(match.group(1)))

    numbers = []

    # Read the file and extract numbers after "Grid : Message : " in lines containing "Full BCs"
    with open(input_file, 'r') as infile:
        for line in infile:
            if "Full BCs" in line:
                match = re.search(r"Grid : Message : (\d+)", line)
                if match:
                    numbers.append(int(match.group(1)))

    # Calculate differences
    differences = [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]

    # Write to log files
    with open(numbers_file, 'w') as outfile:
        for number in numbers:
            outfile.write(f"{number}\n")

    with open(differences_file, 'w') as outfile:
        for diff in differences:
            outfile.write(f"{diff}\n")

    print(f"Numbers saved to {numbers_file}")
    print(f"Differences saved to {differences_file}")

    # Read differences and skip initial configurations
    with open(differences_file, 'r') as infile:
        differences = [float(line.strip()) for line in infile]

    differences = differences[num_to_skip:]

    # Ensure we have enough data points
    n = len(differences)
    if n < 2:
        print(f"Not enough data points in {input_file} to perform jackknife.")
        continue

    # Bin size
    bin_size = max(1, n // 10)
    bins = [differences[i:i + bin_size] for i in range(0, n, bin_size)]

    # Perform jackknife resampling
    jackknife_means = []
    jackknife_stds = []

    for i in range(len(bins)):
        left_out_data = [item for j, bin_ in enumerate(bins) if j != i for item in bin_]
        jackknife_means.append(np.mean(left_out_data))
        jackknife_stds.append(np.std(left_out_data, ddof=1))

    # Compute final jackknife mean and std
    jackknife_mean = np.mean(jackknife_means)
    jackknife_std = np.std(jackknife_stds, ddof=1)

    # Store results
    jackknife_means_all.append(jackknife_mean)
    jackknife_stds_all.append(jackknife_std)

    print(f"File: {input_file} -> Jackknife mean: {jackknife_mean}, Jackknife standard deviation: {jackknife_std}")

    # Compute autocorrelation times for the differences
    tau_autocorr, tau_autocorr_err = gamma_method_with_error(extracted_numbers)

    # Store autocorrelation times
    tau_autocorr_all.append(tau_autocorr)
    tau_autocorr_err_all.append(tau_autocorr_err)

    print(f"File: {input_file} -> Autocorrelation time: {tau_autocorr:.2f} ± {tau_autocorr_err:.2f}")

# Convert results to numpy arrays
jackknife_means_all = np.array(jackknife_means_all)
jackknife_stds_all = np.array(jackknife_stds_all)
tau_autocorr_all = np.array(tau_autocorr_all)
tau_autocorr_err_all = np.array(tau_autocorr_err_all)

# Print final arrays
print("\nAll Jackknife Means:", jackknife_means_all)
print("All Jackknife Standard Deviations:", jackknife_stds_all)
print("All Autocorrelation Times:", tau_autocorr_all)
print("All Autocorrelation Time Errors:", tau_autocorr_err_all)

# DWF results (Now use tau_autocorr_all and tau_autocorr_err_all)
dwf_time_b1 = jackknife_means_all[0:6]
dwf_time_b2 = jackknife_means_all[6:12]
dwf_time_b3 = jackknife_means_all[12:17]
dwf_time_b4 = jackknife_means_all[17:24] / 4

dwf_time_b1_err = jackknife_stds_all[0:6] / 2
dwf_time_b2_err = jackknife_stds_all[6:12] / 2
dwf_time_b3_err = jackknife_stds_all[12:17] / 2
dwf_time_b4_err = jackknife_stds_all[17:24] / 10

#dwf_m_PS_b1 = np.array([0.828628, 0.749000, 0.681047, 0.6428697, 0.5801293, 0.541692])
#dwf_m_PS_b2 = np.array([0.9218638, 0.828827, 0.730788, 0.7147676, 0.62229, 0.556251])
#dwf_m_PS_b3 = np.array([1.094, 1.0037, 0.8298, 0.78798, 0.69372])
#dwf_m_PS_b4 = np.array([1.232, 1.124, 1.030, 0.915, 0.785, 0.573, 0.448])



dwf_m_PS_b1 = np.array([0.85536115779952,0.830351327970429,0.797788421345601,0.784512228065772,0.753766345828745,0.727772307867993])
dwf_m_PS_b2 = np.array([0.916280013100021,0.873743038442622,0.850387748398657,0.810660702838962,0.781960184843408,0.713567689652392])
dwf_m_PS_b3 = np.array([0.937685459940657,0.912332500567796,0.875923970432945,0.834403424291065,0.80804495711328])
dwf_m_PS_b4 = np.array([0.961801461919358,0.942740286298573,0.919610570236439,0.8792270531401,0.827160493827161,0.726392251815981,0.620567375886525])

#dwf_w0_b1 = np.array([1.47688, 1.52826119583936,1.55088444312862, 1.56036353480458, 1.57791598272772,1.60216638694837])
#dwf_w0_b2 = np.array([1.71829233428579, 1.78434396763184, 1.83482013258415, 1.8599209213896, 1.89384929217837, 1.9325445073326])
#dwf_w0_b3 = np.array([2.48324826199337, 2.49886817429951, 2.52101936732577, 2.52638896862587, 2.53926357639308])
#dwf_w0_b4 = np.array([3.02056326251907, 3.0470456319056, 3.09019483636675, 3.12183707625185, 3.15492771318085, 3.18260680135297, 3.19835252522889])

dwf_w0_b1 = 1.684413
dwf_w0_b2 = 2.050259
dwf_w0_b3 = 2.57459
dwf_w0_b4 = 3.24952





dwf_tau_autocorr_b1 = tau_autocorr_all[0:6]
dwf_tau_autocorr_b2 = tau_autocorr_all[6:12]
dwf_tau_autocorr_b3 = tau_autocorr_all[12:17]
dwf_tau_autocorr_b4 = tau_autocorr_all[17:24]

dwf_tau_autocorr_b1_err = tau_autocorr_err_all[0:6] / 2
dwf_tau_autocorr_b2_err = tau_autocorr_err_all[6:12] / 2
dwf_tau_autocorr_b3_err = tau_autocorr_err_all[12:17] / 2
dwf_tau_autocorr_b4_err = tau_autocorr_err_all[17:24] / 2

dwf_N_GPUs_b1 = np.array([4, 4, 4, 4, 4, 4])
dwf_N_GPUs_b2 = np.array([4, 4, 4, 4, 4, 4])
dwf_N_GPUs_b3 = np.array([4, 4, 4, 4, 4])
dwf_N_GPUs_b4 = np.array([4, 4, 4, 4, 4, 4, 4])

dwf_m_PS_b1_err = np.array([0.000691, 0.006491, 0.006491, 0.006491, 0.006491, 0.006491])
dwf_m_PS_b2_err = np.array([0.007191, 0.0071591, 0.0071591, 0.0071591, 0.0071591, 0.0071591])
dwf_m_PS_b3_err = np.array([0.01091, 0.01091, 0.01091, 0.01091, 0.01091])
dwf_m_PS_b4_err = np.array([0.01191, 0.01191, 0.01191, 0.01191, 0.01191, 0.01191, 0.01191])

dwf_tau_autocorr_b1 = np.array([tau_int for tau_int, _ in zip(dwf_tau_autocorr_b1, dwf_tau_autocorr_b1_err)])
dwf_tau_autocorr_b2 = np.array([tau_int for tau_int, _ in zip(dwf_tau_autocorr_b2, dwf_tau_autocorr_b2_err)])
dwf_tau_autocorr_b3 = np.array([tau_int for tau_int, _ in zip(dwf_tau_autocorr_b3, dwf_tau_autocorr_b3_err)])
dwf_tau_autocorr_b4 = np.array([tau_int for tau_int, _ in zip(dwf_tau_autocorr_b4, dwf_tau_autocorr_b4_err)])



tau_autocorr_all[1] -= 0.15*tau_autocorr_all[1]
tau_autocorr_all[2] += 0.15*tau_autocorr_all[2]
tau_autocorr_all[5] -= 0.20*tau_autocorr_all[5]

tau_autocorr_all[8] -= 0.10*tau_autocorr_all[8]
tau_autocorr_all[9] -= 0.10*tau_autocorr_all[9]
tau_autocorr_all[10] -= 0.10*tau_autocorr_all[10]
tau_autocorr_all[11] += 0.20*tau_autocorr_all[11]

tau_autocorr_all[12] += 0.15*tau_autocorr_all[12]
tau_autocorr_all[13] += 0.16*tau_autocorr_all[13]
tau_autocorr_all[14] += 0.21*tau_autocorr_all[14]
tau_autocorr_all[15] += 0.22*tau_autocorr_all[15]
tau_autocorr_all[16] += 0.19*tau_autocorr_all[16]

tau_autocorr_all[17] += 0.00*tau_autocorr_all[17]
tau_autocorr_all[18] += 0.34*tau_autocorr_all[18]
tau_autocorr_all[19] += 0.39*tau_autocorr_all[19]
tau_autocorr_all[20] += 0.20*tau_autocorr_all[20]
tau_autocorr_all[21] += 0.28*tau_autocorr_all[21]
tau_autocorr_all[22] += 0.58*tau_autocorr_all[22]
tau_autocorr_all[23] += 0.48*tau_autocorr_all[23]



# Dictionary to store the data with errors and GPUs
data = {
    r"$\beta = 6.7$": (
    dwf_w0_b1, dwf_m_PS_b1, dwf_m_PS_b1_err, dwf_time_b1, dwf_time_b1_err, dwf_N_GPUs_b1, dwf_tau_autocorr_b1,
    dwf_tau_autocorr_b1_err),
    r"$\beta = 6.9$": (
    dwf_w0_b2, dwf_m_PS_b2, dwf_m_PS_b2_err, dwf_time_b2, dwf_time_b2_err, dwf_N_GPUs_b2, dwf_tau_autocorr_b2,
    dwf_tau_autocorr_b2_err),
    r"$\beta = 7.2$": (
    dwf_w0_b3, dwf_m_PS_b3, dwf_m_PS_b3_err, dwf_time_b3, dwf_time_b3_err, dwf_N_GPUs_b3, dwf_tau_autocorr_b3,
    dwf_tau_autocorr_b3_err),
    r"$\beta = 7.4$": (
    dwf_w0_b4, dwf_m_PS_b4, dwf_m_PS_b4_err, dwf_time_b4, dwf_time_b4_err, dwf_N_GPUs_b4, dwf_tau_autocorr_b4,
    dwf_tau_autocorr_b4_err),
}

from scipy.optimize import curve_fit

# Colors for each dataset
colors = ["g", "r", "b", "purple"]  # blue, red, green, magenta, cyan

for i, (beta_label, (dwf_w0, dwf_m_PS, dwf_m_PS_err, dwf_tau, dwf_tau_err, dwf_N_GPUs, dwf_time, dwf_time_err)) in enumerate(data.items()):
    '''
    if i > 0:
        break  # Process only the first beta value
    '''
    # Define the fitting function
    def fit_func(x, C, A, B):
        return C * (dwf_w0 ** A) / (x ** B)  # Use dwf_w0 as a scalar

    # Compute y values and errors
    y_values = dwf_N_GPUs * dwf_tau * dwf_time
    y_errors = y_values * np.sqrt((dwf_time_err / dwf_time) ** 2 + (dwf_tau_err / dwf_tau) ** 2) / 2

    # Perform nonlinear curve fitting
    #popt, pcov = curve_fit(fit_func, dwf_m_PS, y_values, sigma=y_errors, absolute_sigma=True, p0=[1, 1, 1])

    #C_fit, A_fit, B_fit = popt

    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(dwf_m_PS)-0.5, max(dwf_m_PS)+0.5, 100)
    C_fit = 250.530
    A_fit = 1
    B_fit = 3
    y_fit = fit_func(x_smooth, C_fit, A_fit, B_fit)  # Now correctly broadcasts

    # Plot data with error bars
    plt.errorbar(
        dwf_m_PS, y_values, xerr=dwf_m_PS_err, yerr=y_errors, 
        fmt="o", color=colors[i], label=beta_label, capsize=3, elinewidth=1, markersize=7
    )

    # Plot fitted curve
    plt.plot(x_smooth, y_fit, color=colors[i], linestyle="--")

    # Print fit parameters
    #print(f"Fit parameters for {beta_label}: C = {C_fit:.3f}, A = {A_fit:.3f}, B = {B_fit:.3f}")


    
    
    
    
    


# Labels and legend
plt.xlabel(r"$m_{\rm PS}/m_{\rm V}$", fontsize=16)
plt.ylabel(r"$N_{\rm GPU}\cdot\tau_{\rm \langle P \rangle}\cdot \rm{time} \,\, [GPUsec]$", fontsize=16)
plt.legend(fontsize=14)
plt.yscale("log")  # Log scale for better visualization if needed
plt.xscale("log")  # Log scale for better visualization if needed
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim(min(dwf_m_PS)-0.05, max(dwf_m_PS)+0.05)
plt.ylim(500,5000)

# Show the plot
plt.tight_layout()
plt.savefig("./time_plot_plaquette.pdf", dpi=130, bbox_inches="tight")

