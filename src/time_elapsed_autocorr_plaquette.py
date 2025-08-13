# python3 time_elapsed_autocorr_plaquette.py 1 hmc_51489.out hmc_51189.out hmc_51497.out hmc_51498.out hmc_51516.out hmc_48102.out hmc_51081.out hmc_51083.out hmc_51147.out hmc_50967.out hmc_51148.out hmc_51168.out hmc_49601.out hmc_51204.out hmc_51193.out hmc_51205.out hmc_51203.out hmc_49306.out hmc_49328.out hmc_49333.out hmc_49346.out hmc_49347.out hmc_49352.out hmc_49969.out
# TODO test and fix

from argparse import ArgumentParser, FileType

from collections import defaultdict
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from . import plots

parser = ArgumentParser(description="Compute elapsed time for ensembles and plot")
parser.add_argument("datafiles", nargs="+", metavar="datafile", help="HMC log filename")
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
parser.add_argument(
    "--skip",
    type=int,
    default=0,
    help="Number of trajectories to skip in analysis for thermalisation",
)
parser.add_argument(
    "--metadata_csv",
    required=True,
    help="CSV file containing metadata describing ensembles",
)
parser.add_argument(
    "--spectrum_csv", required=True, help="CSV file containing spectrum results"
)
parser.add_argument(
    "--wflow_csv", required=True, help="CSV file containing gradient flow observables"
)
args = parser.parse_args()

plots.set_styles(args)


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


def check_metadata(line, metadata):
    split_line = line.split()
    key = split_line[0].strip(":")
    if "Full Dimensions" in line:
        metadata["Nx"] = int(split_line[10].lstrip("["))
        metadata["Ny"] = int(split_line[11])
        metadata["Nz"] = int(split_line[12])
        metadata["Nt"] = int(split_line[13].rstrip("]"))
        return
    if line.startswith("SharedMemoryMpi:  World communicator of size"):
        metadata["num_gpus"] = int(split_line[5])
        return

    if key in ["M5", "Mass", "b", "c", "beta"]:
        metadata[key] = float(split_line[1])
    elif key in ["Ls", "starttraj"]:
        metadata[key] = int(split_line[1])


data = []

# Process each file separately
for file_idx, input_file in enumerate(args.datafiles):
    metadata = {}
    # Regular expression pattern to match the Plaquette line and extract the number
    pattern = re.compile(r"Plaquette: \[ \d+ \] ([\d.]+)")

    # List to store extracted numbers
    plaquette_values = []
    cumulative_dslash_counts = []

    # Read the file and extract numbers
    with open(input_file, "r") as f:
        for line in f:
            if not line.startswith("Grid : Message"):
                if not line.startswith("Grid"):
                    check_metadata(line, metadata)
                continue

            if "Full Dimensions" in line:
                check_metadata(line, metadata)

            match = pattern.search(line)
            if match:
                plaquette_values.append(float(match.group(1)))

            if "Full BCs" in line:
                match = re.search(r"Grid : Message : (\d+)", line)
                if match:
                    cumulative_dslash_counts.append(int(match.group(1)))

    # Calculate differences
    cumulative_dslash_counts_array = np.asarray(cumulative_dslash_counts)[args.skip :]
    dslash_counts = (
        cumulative_dslash_counts_array[1:] - cumulative_dslash_counts_array[:-1]
    )

    # Ensure we have enough data points
    if len(dslash_counts) < 2:
        print(f"Not enough data points in {input_file} to perform jackknife.")
        continue

    # Bin size
    bin_size = max(1, len(dslash_counts) // 10)
    num_bins = min(len(dslash_counts), 10)
    dslash_bins = dslash_counts[: bin_size * num_bins].reshape((num_bins, bin_size))

    # Perform jackknife resampling
    jackknife_means = []
    jackknife_stds = []

    for bin_index in range(num_bins):
        filtered_dslash_bins = np.vstack(
            [dslash_bins[:bin_index], dslash_bins[: bin_index + 1]]
        )
        jackknife_means.append(np.mean(filtered_dslash_bins))
        jackknife_stds.append(np.std(filtered_dslash_bins, ddof=1))

    # Compute final jackknife mean and std
    dslash_count = np.mean(jackknife_means)
    dslash_count_error = np.std(jackknife_stds, ddof=1)

    print(f"File: {input_file} -> Dslash count: {dslash_count} ± {dslash_count_error}")

    # Compute autocorrelation times for the differences
    tau_autocorr, tau_autocorr_error = gamma_method_with_error(plaquette_values)

    data.append(
        {
            **metadata,
            "dslash_count": dslash_count,
            "dslash_count_error": dslash_count_error,
            "tau_autocorr": tau_autocorr,
            "tau_autocorr_err": tau_autocorr_error,
        }
    )

    print(
        f"File: {input_file} -> Autocorrelation time: {tau_autocorr:.2f} ± {tau_autocorr_error:.2f}"
    )


metadata = pd.read_csv(args.metadata_csv)
wflow_data = pd.read_csv(args.wflow_csv, comment="#")
spectrum_data = pd.read_csv(args.spectrum_csv, comment="#")


def get_ensemble_label(datum, metadata):
    result = metadata.query(f"beta == {metadata['beta']} & mF == {metadata['Mass']}")
    if len(result) != 1:
        raise ValueError(f"Unique match not found for {datum}")

    return f"ens{result['beta_index'][0]}_m{result['mass_index'][0]}"


def get_ensemble_data(ensemble_key, data):
    if "name" not in data.keys():
        raise ValueError(f"'name' column not found in {data}")

    result = data[data["name"].str.endswith(ensemble_key)]
    if len(result) != 1:
        raise ValueError(f"Unique datum not found for {ensemble_key} in {data}")

    return result


def get_chiral_w0(dataset):
    def w0_fit_form(mass, A, B, C):
        return A + B * mass + C * mass**2

    masses = [datum["Mass"] for datum in dataset]
    w0 = [datum["w_0"] for datum in dataset]
    w0_error = [datum["w_0"] for datum in dataset]
    popt, pcov = curve_fit(w0_fit_form, masses, w0, sigma=w0_error)
    return popt[0]


betas = sorted(set(datum["beta"] for datum in data))
grouped_data = defaultdict(list)

for datum in data:
    ensemble_label = get_ensemble_label(datum, metadata)
    for extra_data in spectrum_data, wflow_data:
        datum.update(
            get_ensemble_data(ensemble_label, extra_data).to_dict(orient="records")[0]
        )
    grouped_data[datum["beta"]].append(datum)

markers = "os^v"
chiral_w0s = {beta: get_chiral_w0(dataset) for beta, dataset in grouped_data.items()}


def fit_func(x, C, A, B, chiral_w0):
    return C * (chiral_w0**A) / (x**B)


def fit_cost(grouped_data, chiral_w0s):
    x_values = [
        datum["g0g5"] / datum["gi"]
        for dataset in grouped_data.values()
        for datum in dataset
    ]
    y_values = [
        datum["num_gpus"] * datum["tau_autocorr"] * datum["dslash_count"]
        for dataset in grouped_data.values()
        for datum in dataset
    ]
    y_error_factors = [
        (
            (datum["tau_autocorr_error"] / datum["tau_autocorr"]) ** 2
            + (datum["dslash_count_error"] / datum["dslash_count"]) ** 2
        )
        ** 0.5
        for dataset in grouped_data.values()
        for datum in dataset
    ]
    y_errors = [
        y_value * y_error_factor
        for y_value, y_error_factor in zip(y_values, y_error_factors)
    ]
    flat_chiral_w0s = [
        chiral_w0
        for chiral_w0, dataset in zip(chiral_w0s.values(), grouped_data.values())
        for _ in dataset
    ]
    return curve_fit(
        lambda x, C, A, B: fit_func(x, C, A, B, flat_chiral_w0s),
        x_values,
        y_values,
        sigma=y_errors,
        p0=[250, 1, 3],
    )


(C_fit, A_fit, B_fit), _ = fit_cost(grouped_data, chiral_w0s)
all_g5_masses = [
    datum["g5g5"] for dataset in grouped_data.values() for datum in dataset
]
fig, ax = plt.subplots()


for colour_index, ((beta, dataset), marker) in enumerate(
    zip(grouped_data.items(), markers)
):
    chiral_w0 = get_chiral_w0(dataset)
    x_values = [datum["g0g5"] / datum["gi"] for datum in dataset]
    x_errors = [
        x_value
        * (
            (datum["g0g5_err"] / datum["g0g5"]) ** 2
            + (datum["gi_err"] / datum["gi"]) ** 2
        )
        ** 0.5
        for x_value, datum in zip(x_values, dataset)
    ]

    # Compute y values and errors
    y_values = [
        datum["num_gpus"] * datum["tau_autocorr"] * datum["dslash_count"]
        for datum in dataset
    ]
    y_errors = [
        y_value
        * (
            (datum["tau_autocorr_error"] / datum["tau_autocorr"]) ** 2
            + (datum["dslash_count_error"] / datum["dslash_count"]) ** 2
        )
        for y_value, datum in zip(y_values, dataset)
    ]

    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(all_g5_masses) - 0.5, max(all_g5_masses) + 0.5, 1000)
    y_fit = fit_func(
        x_smooth, C_fit, A_fit, B_fit, chiral_w0
    )  # Now correctly broadcasts

    # Plot data with error bars
    ax.errorbar(
        x_values,
        y_values,
        xerr=x_errors,
        yerr=y_errors,
        fmt=marker,
        color=f"C{colour_index}",
        label=r"$\\beta={beta}$",
    )

    # Plot fitted curve
    ax.plot(x_smooth, y_fit, color=f"C{colour_index}", linestyle="--")

    # Print fit parameters
    # print(f"Fit parameters for {beta_label}: C = {C_fit:.3f}, A = {A_fit:.3f}, B = {B_fit:.3f}")

# Labels and legend
ax.set_xlabel(r"$m_{\rm PS}/m_{\rm V}$")
ax.set_ylabel(
    r"$N_{\rm GPU}\cdot\tau_{\rm \langle P \rangle}\cdot \rm{time} \,\, [GPUsec]$"
)
ax.legend(loc="best")
ax.set_yscale("log")  # Log scale for better visualization if needed
ax.set_xscale("log")  # Log scale for better visualization if needed
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.set_xlim(min(all_g5_masses) - 0.05, max(all_g5_masses) + 0.05)
ax.set_ylim(500, 5000)

# Show the plot
plots.save_or_show(fig, args.output_file)
