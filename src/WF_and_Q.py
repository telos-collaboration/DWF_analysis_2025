from argparse import ArgumentParser, FileType
import re
import os
import numpy as np
from scipy.optimize import curve_fit


def compute_jackknife(data, bin_size):
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
    input_file_pattern, output_file_plaq, evolution_time, data_all, step
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
            error = compute_jackknife(np.array(data), bin_size)
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


def process_top_charges(input_filename, top_charge_number):
    unique_numbers = set()  # Keep track of unique numbers seen so far
    numbers_in_order = []  # Keep track of the order in which numbers appear in the input files

    with open(input_filename, "r") as f_in:
        for line in f_in:
            if f"Top. charge           : {top_charge_number}" in line:
                number = float(line.split()[-1])
                if number not in unique_numbers:
                    unique_numbers.add(number)
                    numbers_in_order.append(number)

    return numbers_in_order


def process_files(
    input_filename, output_file, output_file_with_index, top_charge_number
):
    unique_numbers = set()  # Keep track of unique numbers seen so far
    numbers_in_order = []  # Keep track of the order in which numbers appear in the input files

    with open(input_filename, "r") as f_in:
        for line in f_in:
            if f"Top. charge           : {top_charge_number}" in line:
                number = float(line.split()[-1])
                if number not in unique_numbers:
                    unique_numbers.add(number)
                    numbers_in_order.append(number)

    for i, number in enumerate(numbers_in_order):
        output_file.write(f"{number}\n")
        output_file_with_index.write(f"{i + 1}\t{number}\n")


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
    "flow_filename",
    help="Log of gradient flow histories for the specified ensemble",
)
parser.add_argument(
    "--tag", default="", help="Tag to distinguish ensemble in combined CSVs"
)
parser.add_argument(
    "--step_length",
    type=float,
    required=True,
    help="Step size between adjacent obsrvable measurements",
)
parser.add_argument(
    "--output_file_t2E", default="/dev/stdout", help="Where to print history of t2E(t)"
)
parser.add_argument(
    "--output_file_W",
    type=FileType("w"),
    default="-",
    help="Where to print history of W(t)",
)
parser.add_argument(
    "--output_file_Q_history",
    type=FileType("w"),
    default="-",
    help="Where to print history of topological charge",
)
parser.add_argument(
    "--output_file_Q_histogram",
    type=FileType("w"),
    default="-",
    help="Where to print histogram of topological charge",
)
parser.add_argument(
    "--output_file_summary",
    type=FileType("w"),
    default="-",
    help="Where to output a CSV of w0, Q, etc.",
)
args = parser.parse_args()


# Max Wilson Flow time
evolution_time = 1800

data_all = []

# Process input files for the current ensemble
process_input_files(
    args.flow_filename,
    args.output_file_t2E,
    evolution_time,
    data_all,
    args.step_length,
)

# Compute jackknife derivatives
print("Computing derivatives...")
derivatives_all = []
for i in range(1, len(data_all)):
    # Skip empty lists to prevent broadcasting errors
    if not data_all[i - 1] or not data_all[i]:
        print(f"Skipping empty data sets at interval {i}")
        continue

    derivatives = compute_derivatives(data_all[i - 1], data_all[i], args.step_length)
    if len(derivatives) == 0:
        continue
    bin_size = max(1, len(derivatives) // 10)
    error = compute_jackknife(
        derivatives,
        bin_size,
    )
    mean_derivative = np.mean(derivatives)
    derivatives_all.append(
        (
            args.step_length + (i - 1) * args.step_length,
            (args.step_length + i * args.step_length) * mean_derivative,
            0,
            (args.step_length + (i - 1) * args.step_length) * error,
        )
    )
    print(
        f"Processed derivatives for interval {i}: Mean={mean_derivative:.4f}, Error={error:.4f}"
    )

# Write the results to the new file
print(f"Writing derivatives to {args.output_file_W.name}...")
args.output_file_W.writelines(
    f"{line[0]:.3f}\t{line[1]}\t{line[2]}\t{line[3]}\n" for line in derivatives_all
)

# Find the closest value to threshold value
closest = find_closest_to_target(args.output_file_W.name)
if closest:
    wf_value = closest[0]
else:
    wf_value = None

# Process top charge numbers for the current ensemble
top_charge_numbers = process_top_charges(args.flow_filename, 100)
print(f"Top charge numbers for ensemble {args.flow_filename}: {top_charge_numbers}")
bin_size2 = max(1, len(top_charge_numbers) // 10)
avg_Q = np.mean(top_charge_numbers)
error_avg_Q = compute_jackknife(top_charge_numbers, bin_size2)
# Now process files for top charges with index
process_files(
    args.flow_filename,
    args.output_file_Q_histogram,
    args.output_file_Q_history,
    1800,
)
tau, tau_error = compute_autocorrelation_and_fit(top_charge_numbers)
if tau is None or tau_error is None:
    # Default or placeholder values
    tau, tau_error = 0.0, 0.0

# Write the results to a CSV file
args.output_file_summary.write("directory,w_0,w_0_error,<Q>,<Q>_err,tau_Q,err_tau_Q\n")
# Multiply WF value by (1/0.02) and cast it to integer
args.output_file_summary.write(
    f"{args.tag},{wf_value},0.01,{avg_Q},{error_avg_Q},{tau},{tau_error}\n"
)

print(f"WF measurements saved to {args.output_file_summary.name}")
