#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
from collections import defaultdict
from itertools import pairwise
import re

import numpy as np
import pandas as pd

from .provenance import get_basic_metadata, text_metadata


def get_args():
    parser = ArgumentParser(
        description="Extract elapsed time and Dslash counts from HMC logs"
    )
    parser.add_argument(
        "datafiles", nargs="+", metavar="datafile", help="HMC log filename"
    )
    parser.add_argument(
        "--output_file",
        type=FileType("w"),
        default="-",
        help="Where to place output (default: stdout)",
    )
    parser.add_argument(
        "--tag", default=None, help="Tag to label ensemble in resulting CSV"
    )
    return parser.parse_args()


def check_metadata(line, metadata):
    split_line = line.split()
    if len(split_line) == 0:
        return

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


def read_file(filename):
    metadata = {}
    # Regular expression pattern to match the Plaquette line and extract the number
    pattern = re.compile(r"Plaquette: \[ (\d+) \] ([\d.]+)")

    # List to store extracted numbers
    trajectory_indices = []
    plaquette_values = []
    cumulative_dslash_counts = []
    generation_times = []

    # Read the file and extract numbers
    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("Grid : Message"):
                if not line.startswith("Grid"):
                    check_metadata(line, metadata)

                if (
                    line.startswith("Grid : HMC")
                    and "Total time for trajectory (s)" in line
                ):
                    generation_times.append(float(line.split()[-1]))
                continue

            if "Full Dimensions" in line:
                check_metadata(line, metadata)
                continue

            plaquette_match = pattern.search(line)
            if plaquette_match:
                trajectory_indices.append(int(plaquette_match.group(1)))
                plaquette_values.append(float(plaquette_match.group(2)))
                continue

            if "Full BCs" in line:
                cumulative_dslash_counts.append(int(line.split()[-1]))
                continue

    return {
        "metadata": metadata,
        "trajectory_indices": np.asarray(trajectory_indices),
        "plaquettes": np.asarray(plaquette_values),
        "dslash_counts": np.diff(cumulative_dslash_counts, prepend=0),
        "generation_times": np.asarray(generation_times),
    }


def check_lengths_consistent(datum):
    lengths = [len(values) for key, values in datum.items() if key != "metadata"]
    if len(set(lengths)) != 1:
        breakpoint()
        raise ValueError("Different lengths obtained")


def check_metadata_consistency(data):
    metadata = [tuple(sorted(datum["metadata"].items())) for datum in data]
    if len(set(metadata)) > 1:
        message = "Inconsistent metadata:\n" + "\n".join(
            f" - {single_metadata}" for single_metadata in set(metadata)
        )
        raise ValueError(message)


def check_consistency(data):
    for datum in data:
        check_lengths_consistent(datum)
    check_metadata_consistency(data)


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


def concatenate(data):
    check_consistency(data)
    sorted_data = sorted(data, key=lambda datum: datum["trajectory_indices"][0])
    for datum, next_datum in pairwise(sorted_data):
        if datum["trajectory_indices"][-1] != 1 + next_datum["trajectory_indices"][0]:
            raise ValueError("Non-consecutive ensembles detected")

    combined_data = defaultdict(list)
    for datum in data:
        for key, value in datum.items():
            if key != "metadata":
                combined_data[key].append(value)

    result = {key: np.concatenate(value) for key, value in combined_data.items()}
    result["metadata"] = data[0]["metadata"]
    return result


def append_tuple(data, key, values):
    value, error = values
    data[key] = value
    data[f"{key}_error"] = error


def jackknife(data):
    if len(data) < 2:
        raise ValueError("Insufficient data to perform jackknife.")

    # Bin data
    bin_size = max(1, len(data) // 10)
    num_bins = min(len(data), 10)
    bins = data[: bin_size * num_bins].reshape((num_bins, bin_size))

    # Construct jackknife samples
    jackknife_means = []
    jackknife_stds = []

    for bin_index in range(num_bins):
        filtered_bins = np.vstack([bins[:bin_index], bins[: bin_index + 1]])
        jackknife_means.append(np.mean(filtered_bins))
        jackknife_stds.append(np.std(filtered_bins, ddof=1))

    # Compute final jackknife mean and std
    value = np.mean(jackknife_means)
    error = np.std(jackknife_stds, ddof=1)

    return value, error


def compute_means(data):
    result = {**data["metadata"]}

    append_tuple(
        result, "plaquette_autocorr", gamma_method_with_error(data["plaquettes"])
    )
    append_tuple(result, "plaquette", jackknife(data["plaquettes"]))
    append_tuple(result, "dslash_count", jackknife(data["dslash_counts"]))
    append_tuple(result, "generation_time", jackknife(data["generation_times"]))
    result["min_trajectory_index"] = min(data["trajectory_indices"])
    result["max_trajectory_index"] = max(data["trajectory_indices"])
    return result


def main():
    args = get_args()
    data = [
        datum
        for filename in args.datafiles
        if len((datum := read_file(filename))["trajectory_indices"]) > 0
    ]
    results = compute_means(concatenate(data))
    results["name"] = args.tag
    print(text_metadata(get_basic_metadata()), file=args.output_file)
    print(pd.DataFrame([results]).to_csv(index=False), file=args.output_file)


if __name__ == "__main__":
    main()
