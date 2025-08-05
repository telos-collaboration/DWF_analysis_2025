from argparse import ArgumentParser, FileType

import os
import csv
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .pj5q import read_pj5q
from .pcac import read_pcac
from .g5g5 import (
    read_g5g5,
    read_gi,
    read_g0gi,
    read_g0g5,
    read_g5gi,
    read_id,
    read_g0g5_g5,
)
from . import corrutils as cu


parser = ArgumentParser(
    description="Fits correlation functions for all ensembles and outputs resulting spectrum to CSV"
)
parser.add_argument(
    "--correlator_dir",
    required=True,
    help="Top-level directory containing correlation functions",
)
parser.add_argument(
    "--csv_file", type=FileType("w"), default="-", help="Output CSV file"
)
args = parser.parse_args()

# Configuration
header = [
    "name",
    "g5g5",
    "g5g5_err",
    "g0g5",
    "g0g5_err",
    "gi",
    "gi_err",
    "g0gi",
    "g0gi_err",
    "g5gi",
    "g5gi_err",
    "g0g5gi",
    "g0g5gi_err",
    "id",
    "id_err",
    "effmass",
    "effmass_err",
    "fpi",
    "fpi_err",
    "Z_A",
    "err_Z_A",
]


# Extract (N, M) from subdirectory names
def extract_N_M(subdir_name):
    match = re.search(r"ens(\d+)_m(\d+)", subdir_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float("inf"), float("inf")


# Extract correlators from XML files
def extract_correlators(file_path, gamma_snk, gamma_src):
    import xml.etree.ElementTree as ET

    tree = ET.parse(file_path)
    root = tree.getroot()
    correlators = []
    for elem in root.findall(".//elem"):
        snk_elem = elem.find("gamma_snk")
        src_elem = elem.find("gamma_src")
        if snk_elem is not None and src_elem is not None:
            snk, src = snk_elem.text, src_elem.text
            if snk == gamma_snk and src == gamma_src:
                corr_elements = elem.find("corr").findall("elem")
                correlators = [
                    float(corr.text.strip("()").split(",")[0]) for corr in corr_elements
                ]
                break
    return correlators


def fold_correlators2(correlators, T, V):
    folded = []
    if correlators[0] < 0:
        correlators[0] = -correlators[0] / V
    folded.append(correlators[0])
    for i in range(1, int(T / 2)):
        if correlators[i] * correlators[T - i] < 0:
            correlators[i] = -correlators[i]
        t = (correlators[i] + correlators[T - i]) / 2 / V
        folded.append(t)
    folded.append(correlators[int(T / 2)] / V)
    return folded


# Fitting function
def fitting_function(t, a, b, m, T):
    return a * b / (2 * m) * (np.exp(-t * m) + np.exp(-(T - t) * m))


# Perform fit with jackknife resampling
def perform_fit(correlators_G5G5, correlators_GT5, ti1, tf1, ti2, tf2, T):
    t_values_G5G5 = np.arange(ti1, tf1)
    t_values_GT5 = np.arange(ti2, tf2)
    n = len(correlators_G5G5)
    fit_params = []
    for i in range(n):
        indices = np.delete(np.arange(n), i)
        ydata_G5G5 = np.mean(correlators_G5G5[indices], axis=0)[ti1:tf1]
        ydata_GT5 = np.mean(correlators_GT5[indices], axis=0)[ti2:tf2]
        popt, _ = curve_fit(
            lambda t, b, m: fitting_function(t, 1, b, m, T), t_values_G5G5, ydata_G5G5
        )
        b, m = popt
        popt, _ = curve_fit(
            lambda t, a: fitting_function(t, a, np.sqrt(b), m, T),
            t_values_GT5,
            ydata_GT5,
        )
        a = popt[0]
        fit_params.append((a, b, m))
    return np.array(fit_params)


# Compute jackknife errors
def jackknife_fit_error(fit_params):
    n = len(fit_params)
    mean_fit_params = np.mean(fit_params, axis=0)
    errors = np.sqrt((n - 1) * np.mean((fit_params - mean_fit_params) ** 2, axis=0))
    return mean_fit_params, errors


# Main script
writer = csv.writer(args.csv_file)
if args.csv_file.tell() == 0:
    writer.writerow(header)

subdirectories = [
    sub
    for sub in os.listdir(args.correlator_dir)
    if os.path.isdir(os.path.join(args.correlator_dir, sub))
]
sorted_subdirectories = sorted(subdirectories, key=lambda x: extract_N_M(x))

for subdirectory in sorted_subdirectories:
    subdir_path = os.path.join(args.correlator_dir, subdirectory)
    print(f"Processing: {subdir_path}")

    # Read the correlators for each channel
    g5g5_correlators_array = read_g5g5(subdir_path)
    g0g5_correlators_array = read_g0g5(subdir_path)
    gi_correlators_array = read_gi(subdir_path)
    g0gi_correlators_array = read_g0gi(subdir_path)
    g5gi_correlators_array = read_g5gi(subdir_path)
    g0g5gi_correlators_array = read_g5gi(subdir_path)
    id_correlators_array = read_id(subdir_path)

    time_extent = g5g5_correlators_array.shape[1]

    # Fold the correlators
    g5g5_correlators_array = cu.fold_correlators(g5g5_correlators_array.T)
    g0g5_correlators_array = cu.fold_correlators(g0g5_correlators_array.T)
    gi_correlators_array = cu.fold_correlators(gi_correlators_array.T)
    g0gi_correlators_array = cu.fold_correlators(g0gi_correlators_array.T)
    g5gi_correlators_array = cu.fold_correlators(g5gi_correlators_array.T)
    g0g5gi_correlators_array = cu.fold_correlators(g0g5gi_correlators_array.T)
    id_correlators_array = cu.fold_correlators(id_correlators_array.T)

    print("time_extent: ", time_extent)
    # Compute the effective masses
    eff_mass_g5g5, err_eff_mass_g5g5, eff_mass_extended_g5g5 = cu.effective_mass(
        g5g5_correlators_array, time_extent
    )
    eff_mass_g0g5, err_eff_mass_g0g5, eff_mass_extended_g0g5 = cu.effective_mass(
        g0g5_correlators_array, time_extent
    )
    eff_mass_gi, err_eff_mass_gi, eff_mass_extended_gi = cu.effective_mass(
        gi_correlators_array, time_extent
    )
    eff_mass_g0gi, err_eff_mass_g0gi, eff_mass_extended_g0gi = cu.effective_mass(
        g0gi_correlators_array, time_extent
    )
    eff_mass_g5gi, err_eff_mass_g5gi, eff_mass_extended_g5gi = cu.effective_mass(
        g5gi_correlators_array, time_extent
    )
    eff_mass_g0g5gi, err_eff_mass_g0g5gi, eff_mass_extended_g0g5gi = cu.effective_mass(
        g0g5gi_correlators_array, time_extent
    )
    eff_mass_id, err_eff_mass_id, eff_mass_extended_id = cu.effective_mass(
        id_correlators_array, time_extent
    )

    # Plateau fitting
    factor = 1
    ti, tf = (
        int(time_extent / 2) - 3,
        int(time_extent / 2) - 1,
    )  # Plateau time range
    covariance_matrix_g5g5 = np.cov(eff_mass_extended_g5g5.T)
    mean_plateau_g5g5, error_plateau_g5g5, _ = cu.perform_correlated_fit(
        eff_mass_g5g5, covariance_matrix_g5g5, ti, tf
    )
    error_plateau_g5g5 = factor * error_plateau_g5g5

    covariance_matrix_g0g5 = np.cov(eff_mass_extended_g0g5.T)
    mean_plateau_g0g5, error_plateau_g0g5, _ = cu.perform_correlated_fit(
        eff_mass_g0g5, covariance_matrix_g0g5, ti, tf
    )
    error_plateau_g0g5 = factor * error_plateau_g0g5

    covariance_matrix_gi = np.cov(eff_mass_extended_gi.T)
    mean_plateau_gi, error_plateau_gi, _ = cu.perform_correlated_fit(
        eff_mass_gi, covariance_matrix_gi, ti, tf
    )
    error_plateau_gi = factor * error_plateau_gi

    covariance_matrix_g0gi = np.cov(eff_mass_extended_g0gi.T)
    mean_plateau_g0gi, error_plateau_g0gi, _ = cu.perform_correlated_fit(
        eff_mass_g0gi, covariance_matrix_g0gi, ti, tf
    )
    error_plateau_g0gi = factor * error_plateau_g0gi

    covariance_matrix_g5gi = np.cov(eff_mass_extended_g5gi.T)
    mean_plateau_g5gi, error_plateau_g5gi, _ = cu.perform_correlated_fit(
        eff_mass_g5gi, covariance_matrix_g5gi, ti, tf
    )
    error_plateau_g5gi = factor * error_plateau_g5gi

    covariance_matrix_g0g5gi = np.cov(eff_mass_extended_g0g5gi.T)
    mean_plateau_g0g5gi, error_plateau_g0g5gi, _ = cu.perform_correlated_fit(
        eff_mass_g0g5gi, covariance_matrix_g0g5gi, ti, tf
    )
    error_plateau_g0g5gi = factor * error_plateau_g0g5gi

    covariance_matrix_id = np.cov(eff_mass_extended_id.T)
    mean_plateau_id, error_plateau_id, _ = cu.perform_correlated_fit(
        eff_mass_id, covariance_matrix_id, ti, tf
    )
    error_plateau_id = factor * error_plateau_id

    # Compute the effective mass of the ratio
    PJ5q_correlators_array = read_pj5q(subdir_path)
    PJ5q_correlators_array = cu.fold_correlators(PJ5q_correlators_array.T)
    PJ5q_avgs = np.array(
        [
            np.mean(PJ5q_correlators_array[i, 0:])
            for i in range(PJ5q_correlators_array.shape[0])
        ]
    )

    mres, err_mres = cu.jackknife_effective_mass_block(
        PJ5q_correlators_array, g5g5_correlators_array
    )

    tmp = PJ5q_avgs / np.mean(g5g5_correlators_array, axis=1)
    mres_array = np.nan_to_num(tmp)  # Replace NaNs with 0

    covariance_matrix = np.cov((PJ5q_correlators_array.T / g5g5_correlators_array.T).T)
    mean_plateau, error_plateau, _ = cu.perform_correlated_fit(
        mres, covariance_matrix, ti, tf
    )
    error_plateau = factor * error_plateau

    # Spatial Volume declaration:
    if time_extent == 32:
        L = 24
    elif time_extent == 48:
        L = 48
    elif time_extent == 64:
        L = 56

    V = L**3

    # Additional XML-based fits
    gamma_pairs = [("Gamma5", "Gamma5"), ("GammaTGamma5", "Gamma5")]
    correlators_data = {pair: [] for pair in gamma_pairs}
    file_pattern = os.path.join(subdir_path, "pt_ll.[0-9]*.xml")
    files = glob.glob(file_pattern)
    for file_path in files:
        for gamma_snk, gamma_src in gamma_pairs:
            correlators = extract_correlators(file_path, gamma_snk, gamma_src)
            folded = fold_correlators2(correlators, time_extent, V)
            correlators_data[(gamma_snk, gamma_src)].append(folded)

    # Perform jackknife fits
    correlators_G5G5 = np.array(correlators_data[("Gamma5", "Gamma5")])
    correlators_GT5 = np.array(correlators_data[("GammaTGamma5", "Gamma5")])
    fit_params = perform_fit(
        correlators_G5G5, correlators_GT5, ti, tf, ti, tf, time_extent
    )
    mean_fit, errors_fit = jackknife_fit_error(fit_params)

    fpi = mean_fit[0] / mean_fit[2]
    error_fpi = np.sqrt(
        (errors_fit[0] / mean_fit[2]) ** 2
        + (mean_fit[0] * errors_fit[2] / mean_fit[2] ** 2) ** 2
    )

    # Find Z_A
    therm = 0

    PCAC_correlators_array = read_pcac(subdir_path)
    tmp = PCAC_correlators_array.T
    PCAC_correlators_array = cu.fold_correlators_ZA(tmp)
    PCAC_avgs = np.array(
        [
            np.mean(PCAC_correlators_array[i, therm:])
            for i in range(PCAC_correlators_array.shape[0])
        ]
    )

    g5g5_correlators_array = read_g0g5_g5(subdir_path)
    tmp = g5g5_correlators_array.T
    g5g5_correlators_array = cu.fold_correlators(tmp)
    g5g5_avgs = np.array(
        [
            np.mean(g5g5_correlators_array[i, therm:])
            for i in range(g5g5_correlators_array.shape[0])
        ]
    )
    ZA, err_ZA, ZA_extended = cu.jackknife_effective_mass_block_ZA(
        PCAC_correlators_array, g5g5_correlators_array
    )
    err_ZA /= V
    tmp = g5g5_avgs
    mres_array = tmp

    for i in range(len(PCAC_avgs)):
        if np.isnan(tmp[i]):
            mres_array[i] = 0.0
        else:
            mres_array[i] = tmp[i]

    Nt = mres_array.shape[0]  # Number of time slices

    eff_mass, err_eff_mass, eff_mass_extended = cu.effective_mass(
        g5g5_correlators_array, time_extent
    )

    Lt_corr = np.mean(g5g5_correlators_array, axis=1)
    err_Lt = np.std(g5g5_correlators_array, axis=1)

    factor = 1 / 100
    ti = int(time_extent / 2) - 6
    tf = int(time_extent / 2) - 2
    plateau = True

    if plateau:
        covariance_matrix = np.cov(ZA_extended.T)
        mean_plateau_ZA, error_plateau_ZA, chi_square = cu.perform_correlated_fit(
            ZA, covariance_matrix, ti, tf
        )
        error_plateau_ZA = factor * error_plateau_ZA

    # Write the results for this subdirectory
    results = [
        subdir_path,
        mean_plateau_g5g5,
        error_plateau_g5g5,
        mean_plateau_g0g5,
        error_plateau_g0g5,
        mean_plateau_gi,
        error_plateau_gi,
        mean_plateau_g0gi,
        error_plateau_g0gi,
        mean_plateau_g5gi,
        error_plateau_g5gi,
        mean_plateau_g0g5gi,
        error_plateau_g0g5gi,
        mean_plateau_id,
        error_plateau_id,
        mean_plateau,
        error_plateau,
        fpi,
        error_fpi,
        mean_plateau_ZA,
        error_plateau_ZA,
    ]
    writer.writerow(results)

    print(f"Results for {subdirectory} have been saved.")

print(f"All results have been saved to {args.csv_file.name}.")
