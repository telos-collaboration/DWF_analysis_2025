from argparse import ArgumentParser, FileType
import re
import os
import numpy as np
from scipy.optimize import curve_fit


def compute_jackknife(data, bin_size, beta):
    """
    Compute the jackknife error given a list of data and bin size.
    """
    n = len(data)
    if n == 0:
        return 0.0
    jackknife_means = np.array(
        [
            np.mean(np.concatenate((data[:i], data[i + bin_size :])))
            for i in range(0, n, bin_size)
        ]
    )
    mean = np.mean(data)
    variance = (
        (len(jackknife_means) - 1)
        * np.sum((jackknife_means - mean) ** 2)
        / len(jackknife_means)
    )
    return np.sqrt(variance)


def process_input_files(
    input_file_pattern, output_file_plaq, evolution_time, data_all, beta, step
):
    print(f"Processing input file: {input_file_pattern}")

    # Read in the input file
    with open(input_file_pattern, "r") as input_file:
        input_lines = input_file.readlines()

    # Define the regex pattern and extract the relevant data once
    pattern2 = re.compile(r"\[WilsonFlow\] Energy density \(plaq\) : (\d{2,4}) ")
    data_dict = {i: [] for i in range(1, evolution_time + 1)}

    print("Extracting and organizing data...")
    for line in input_lines:
        match = pattern2.search(line)
        if match:
            number = int(match.group(1))
            if 1 <= number <= evolution_time:
                try:
                    value = float(line.strip().split()[-1])
                    data_dict[number].append(value)
                except ValueError:
                    continue

    # Compute jackknife errors and write to output file
    results = []
    for i in range(1, evolution_time + 1):
        data = data_dict[i]
        data_all.append(data)
        if data:
            average = np.mean(data)
            bin_size = max(1, len(data) // 10)
            error = compute_jackknife(np.array(data), bin_size, beta)
            first_column = step + (i - 1) * step
            results.append(f"{first_column:.3f}\t{average:.4f}\t0\t{error:.4f}\n")
            print(
                f"Processed data for step {i}: Average={average:.4f}, Error={error:.4f}"
            )

    with open(output_file_plaq, "w") as output_file:
        output_file.writelines(results)
    print(f"Results written to {output_file_plaq}")


def compute_derivatives(data1, data2, step):
    # Adjust to the smaller length of the two arrays to avoid broadcasting errors
    min_length = min(len(data1), len(data2))
    data1, data2 = np.array(data1[:min_length]), np.array(data2[:min_length])
    return (data2 - data1) / step  # assuming time step of 'step'


def find_closest_to_target(file_path, target=0.2815):
    """
    Find the value in the first column corresponding to the second column value
    closest to the target.
    """
    closest_value = None
    closest_row = None
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                first_column = float(parts[0])
                second_column = float(parts[1])
                if closest_value is None or abs(second_column - target) < abs(
                    closest_value - target
                ):
                    closest_value = second_column
                    closest_row = (first_column, second_column)
            except ValueError:
                continue
    return closest_row


def process_top_charges(directory, top_charge_number):
    input_files = [
        f for f in os.listdir(directory) if f.endswith(".out") and "wflow." in f
    ]
    missing_files = []

    for file in input_files:
        file_path = os.path.join(directory, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"The following files are missing: {', '.join(missing_files)}")

    unique_numbers = set()  # Keep track of unique numbers seen so far
    numbers_in_order = []  # Keep track of the order in which numbers appear in the input files

    for file in input_files:
        file_path = os.path.join(directory, file)
        if not os.path.exists(file_path):
            continue  # Skip this file if it doesn't exist

        with open(file_path, "r") as f_in:
            for line in f_in:
                if f"Top. charge           : {top_charge_number}" in line:
                    number = float(line.split()[-1])
                    if number not in unique_numbers:
                        unique_numbers.add(number)
                        numbers_in_order.append(number)

    return numbers_in_order


def process_files(input_files, output_file, output_file_with_index, top_charge_number):
    unique_numbers = set()  # Keep track of unique numbers seen so far
    numbers_in_order = []  # Keep track of the order in which numbers appear in the input files

    for file in input_files:
        with open(file, "r") as f_in:
            for line in f_in:
                if f"Top. charge           : {top_charge_number}" in line:
                    number = float(line.split()[-1])
                    if number not in unique_numbers:
                        unique_numbers.add(number)
                        numbers_in_order.append(number)

    with open(output_file, "w") as f_out, open(
        output_file_with_index, "w"
    ) as f_out_index:
        for i, number in enumerate(numbers_in_order):
            f_out.write(f"{number}\n")
            f_out_index.write(f"{i + 1}\t{number}\n")


def compute_autocorrelation_and_fit(data, tmax=8):
    if len(data) < tmax:
        print("Insufficient data points for autocorrelation and fit.")
        return None, None

    mean = np.mean(data)
    N = len(data)
    autocorr = []

    for lag in range(1, tmax):
        gamma = (1 / (N - lag)) * np.sum((data[:-lag] - mean) * (data[lag:] - mean))
        autocorr.append(gamma)

    # Normalize the autocorrelation
    autocorr = np.array(autocorr)
    if autocorr[0] == 0 or np.allclose(autocorr, 0):
        print("Autocorrelation is zero or close to zero. Skipping fit.")
        return None, None

    autocorr /= autocorr[0]

    # Exponential fit function
    def exponential_sum(t, A, B):
        return A * np.exp(-t / B)

    x = np.arange(1, tmax)
    y = autocorr[: tmax - 1]

    # Perform the curve fitting
    try:
        params, covariance = curve_fit(exponential_sum, x, y, maxfev=10000)
        A, B = params
        B_error = np.sqrt(np.diag(covariance))[1]  # Error in B
        return B, B_error
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}. Returning None.")
        return None, None


parser = ArgumentParser(
    description="Extracts scales from gradient flow histories for all ensembles and outputs resulting data to CSV"
)
parser.add_argument(
    "--wf_dir",
    required=True,
    help="Top-level directory containing gradient flow histories",
)
parser.add_argument(
    "--csv_file", type=FileType("w"), default="-", help="Output CSV file"
)
args = parser.parse_args()


# Automatically detect the range for N and M
ens_dirs = [
    d for d in os.listdir(args.wf_dir) if os.path.isdir(os.path.join(args.wf_dir, d))
]
N_range = sorted(
    set(int(d.split("_")[0][3:]) for d in ens_dirs if d.startswith("ens") and "_m" in d)
)
M_range = sorted(
    set(int(d.split("_")[1][1:]) for d in ens_dirs if d.startswith("ens") and "_m" in d)
)

# Max Wilson Flow time
evolution_time = 1800

csv_results = []

avg_Q_array = []
tau_Q_array = []

step = 0.005

for N in N_range:
    for M in M_range:
        if N == 1:
            beta = 6.9
            step = 0.005
        elif N == 2:
            beta = 7.2
            step = 0.005
            if M == 7 or M == 8:
                step = 0.01
        elif N == 3:
            beta = 7.4
            step = 0.005
            if M == 8:
                step = 0.01
        elif N == 4:
            step = 0.005
            beta = 6.7

        directory = f"{args.wf_dir}/ens{N}_m{M}"
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue

        input_files = [
            f for f in os.listdir(directory) if f.endswith(".out") and "wflow" in f
        ]
        if not input_files:
            print(f"No input files found in directory: {directory}")
            continue

        output_file_name = f"{directory}/WF_b68_am-08_l8.txt"
        data_all = []

        # Process input files for the current ensemble
        process_input_files(
            f"{directory}/{input_files[0]}",
            output_file_name,
            evolution_time,
            data_all,
            beta,
            step,
        )

        # Create new file names with '_2' suffix
        output_file_name_2 = output_file_name.replace(".txt", "_2.txt")

        # Compute jackknife derivatives
        print("Computing derivatives...")
        derivatives_all = []
        for i in range(1, len(data_all)):
            # Skip empty lists to prevent broadcasting errors
            if not data_all[i - 1] or not data_all[i]:
                print(f"Skipping empty data sets at interval {i}")
                continue

            derivatives = compute_derivatives(data_all[i - 1], data_all[i], step)
            if len(derivatives) == 0:
                continue
            print("beta: ", beta)
            bin_size = max(1, len(derivatives) // 10)
            error = compute_jackknife(derivatives, bin_size, beta)
            mean_derivative = np.mean(derivatives)
            derivatives_all.append(
                (
                    step + (i - 1) * step,
                    (step + i * step) * mean_derivative,
                    0,
                    (step + (i - 1) * step) * error,
                )
            )
            print(
                f"Processed derivatives for interval {i}: Mean={mean_derivative:.4f}, Error={error:.4f}"
            )

        # Write the results to the new file
        with open(output_file_name_2, "w") as file_2:
            print(f"Writing derivatives to {output_file_name_2}...")
            file_2.writelines(
                f"{line[0]:.3f}\t{line[1]}\t{line[2]}\t{line[3]}\n"
                for line in derivatives_all
            )

        print(f"New file created with '_2' suffix: {output_file_name_2}")

        # Find the closest value to 0.200
        closest = find_closest_to_target(output_file_name_2)
        if closest:
            csv_results.append((directory, closest[0], 0.01))
        # print(csv_results)
        # Process top charge numbers for the current ensemble
        top_charge_numbers = process_top_charges(directory, 100)
        print(f"Top charge numbers for ensemble ens{N}_m{M}: {top_charge_numbers}")
        bin_size2 = max(1, len(top_charge_numbers) // 10)
        avg_Q = np.mean(top_charge_numbers)
        error_avg_Q = compute_jackknife(top_charge_numbers, bin_size2, beta)
        avg_Q_array.append([avg_Q, error_avg_Q])
        # Now process files for top charges with index
        input_files = [
            os.path.join(directory, f) for f in input_files
        ]  # Ensure full paths
        process_files(
            input_files,
            f"{directory}/top_charges_b68-am08.txt",
            f"{directory}/top_charges_b68-am08_with_index.txt",
            1800,
        )
        tau, tau_error = compute_autocorrelation_and_fit(top_charge_numbers)
        if tau is not None and tau_error is not None:
            tau_Q_array.append([tau, tau_error])
        else:
            tau_Q_array.append([0.0, 0.0])  # Default or placeholder values


# Write the results to a CSV file
args.csv_file.write("directory,w_0,w_0_error,<Q>,<Q>_err,tau_Q,err_tau_Q\n")
for idx, row in enumerate(csv_results):
    # Multiply WF value by (1/0.02) and cast it to integer
    wf_value = row[1]
    args.csv_file.write(
        f"{row[0]},{wf_value},{row[2]:.2f},{avg_Q_array[idx][0]},{avg_Q_array[idx][1]},{tau_Q_array[idx][0]},{tau_Q_array[idx][1]}\n"
    )

print(f"WF measurements saved to {args.csv_file.name}")
