#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
from itertools import product

import numpy as np
import pandas as pd
from scipy import odr

from . import plots
from .provenance import get_basic_metadata, text_metadata


def get_args():
    parser = ArgumentParser(
        description="Fit Wilson and DWF data with appropriate EFTs and plot both data and fit result"
    )
    parser.add_argument(
        "--mobius_data",
        required=True,
        help="CSV containing spectrum and gradient flow results for Möbius domain wall fermions",
    )
    parser.add_argument(
        "--wilson_data",
        required=True,
        help="CSV containing spectrum and gradient flow results for Wilson fermions",
    )
    parser.add_argument(
        "--output_file",
        type=FileType("w"),
        default="-",
        help="Where to output fit results (default stdout)",
    )
    return parser.parse_args()


def fit_form_linear(w0m_squared, one_over_w0, w0X_squared, L0X, W0X):
    # Eq. 60, 61, 63 of draft
    return w0X_squared * (1 + L0X * w0m_squared) + W0X * one_over_w0


def fit_form_quadratic(w0m_squared, one_over_w0, w0X_squared, L0X, W0X, L1X):
    # Eq. 62 of draft
    return (
        w0X_squared * (1 + L0X * w0m_squared + L1X * w0m_squared**2) + W0X * one_over_w0
    )


def fit_form_cubic(w0m_squared, one_over_w0, w0X_squared, L0X, W0X, L1X, L2X):
    # Extension for evaluating systematics
    return (
        w0X_squared
        * (1 + L0X * w0m_squared + L1X * w0m_squared**2 + L2X * w0m_squared**3)
        + W0X * one_over_w0
    )


def fit_single(data, state_column, fit_order):
    fit_forms = {
        1: fit_form_linear,
        2: fit_form_quadratic,
        3: fit_form_cubic,
    }
    # w0X_squared and W0X in addition to polynomial coefficients in w0m_squared
    n_params = fit_order + 2
    combined_x = np.vstack([data["x"].values, data["one_over_w0"].values])
    combined_dx = np.vstack([data["x_err"].values, data["one_over_w0_err"].values])
    odr_data = odr.RealData(
        combined_x,
        data[f"y_{state_column}"].values,
        combined_dx,
        data[f"y_{state_column}_err"].values,
    )
    model = odr.Model(
        lambda params, data: fit_forms[fit_order](data[0], data[1], *params)
    )
    odr_obj = odr.ODR(odr_data, model, beta0=[1.0] * n_params)
    return odr_obj.run()


def bootstrap_fit(data, state_column, fit_order):
    rng = plots.get_rng(data.values.copy(order="C"))
    fit_samples = np.asarray(
        [
            fit_single(
                data.loc[rng.choice(data.index, len(data))],
                state_column,
                fit_order,
            ).beta
            for _ in range(plots.NUM_BOOTSTRAP_SAMPLES)
        ]
    )
    return fit_samples.mean(axis=0), fit_samples.std(axis=0)


def get_systematics(data, state_column, fit_order):
    n_params = fit_order + 2
    results = [
        fit_single(
            data[data["x"] <= data["x"].max() * test_upper_bound],
            state_column,
            test_fit_order,
        ).beta[:n_params]
        for test_upper_bound, test_fit_order in product(
            [1, 2 / 3, 1 / 2], [fit_order, fit_order + 1]
        )
    ]
    return np.max(results, axis=0) - np.min(results, axis=0)


def bootstrap_fit_with_systematics(data, state_column, fit_order):
    central_value, statistical_error = bootstrap_fit(data, state_column, fit_order)
    systematic_error = get_systematics(data, state_column, fit_order)
    return central_value, statistical_error, systematic_error


def fit_eft(data, fit_type):
    mV_result = bootstrap_fit_with_systematics(data, "mV", 1)
    if fit_type == "mobius":
        fPS_result = bootstrap_fit_with_systematics(data, "fPS", 2)
    else:
        fPS_result = bootstrap_fit_with_systematics(data, "fPS", 1)

    return {"mV": mV_result, "fPS": fPS_result}


def write_fit_results(mobius_fit, wilson_fit, output_file):
    results = []
    for formulation, fit_results in [
        ("Möbius DWF", mobius_fit),
        ("Wilson", wilson_fit),
    ]:
        for state in "mV", "fPS":
            result = {"state": state, "formulation": formulation}

            for key, value, statistical_error, systematic_error in zip(
                ["w0X_squared", "L0X", "W0X", "L1X"], *fit_results[state]
            ):
                result[key] = value
                result[f"{key}_err"] = statistical_error
                result[f"{key}_systematic_error"] = systematic_error
            results.append(result)

    print(text_metadata(get_basic_metadata(), comment_char="#"), file=output_file)
    print(pd.DataFrame(results).to_csv(index=False), file=output_file)


def square_scale_w0(w0, w0_err, value, error):
    result_value = w0**2 * value**2
    result_err = result_value * ((w0_err / w0) ** 2 + (error / value) ** 2) ** 0.5
    return result_value, result_err


def homogenise_df(data, columns):
    w0 = data[columns["w0"]]
    w0_err = data[columns["w0_err"]]
    one_over_w0 = 1 / w0
    one_over_w0_err = w0_err / w0**2
    one_over_w0_squared = 1 / w0**2
    one_over_w0_squared_err = w0_err / w0**3
    mPS = data[columns["mPS"]]
    mPS_err = data[columns["mPS_err"]]
    fPS = data[columns["fPS"]]
    fPS_err = data[columns["fPS_err"]]
    mV = data[columns["mV"]]
    mV_err = data[columns["mV_err"]]

    x_value, x_err = square_scale_w0(w0, w0_err, mPS, mPS_err)
    y_fPS_value, y_fPS_err = square_scale_w0(w0, w0_err, fPS, fPS_err)
    y_mV_value, y_mV_err = square_scale_w0(w0, w0_err, mV, mV_err)

    return pd.DataFrame(
        {
            "beta": data[columns["beta"]],
            "w0": w0,
            "w0_err": w0_err,
            "one_over_w0": one_over_w0,
            "one_over_w0_err": one_over_w0_err,
            "one_over_w0_squared": one_over_w0_squared,
            "one_over_w0_squared_err": one_over_w0_squared_err,
            "x": x_value,
            "x_err": x_err,
            "y_fPS": y_fPS_value,
            "y_fPS_err": y_fPS_err,
            "y_mV": y_mV_value,
            "y_mV_err": y_mV_err,
        }
    ).dropna()


def homogenise_dfs(mobius_df, wilson_df):
    return (
        homogenise_df(
            mobius_df,
            {
                "beta": "beta",
                "w0": "w_0",
                "w0_err": "w_0_error",
                "fPS": "fpi",
                "fPS_err": "fpi_err",
                "mPS": "g0g5",
                "mPS_err": "g0g5_err",
                "mV": "gi",
                "mV_err": "gi_err",
            },
        ),
        homogenise_df(
            wilson_df,
            {
                "beta": "beta",
                "w0": "w_0",
                "w0_err": "w_0_error",
                "mPS": "g5_mass",
                "mPS_err": "g5_mass_error",
                "fPS": "g5_decay_const",
                "fPS_err": "g5_decay_const_error",
                "mV": "gk_mass",
                "mV_err": "gk_mass_error",
            },
        ),
    )


def main():
    args = get_args()
    mobius_data, wilson_data = homogenise_dfs(
        pd.read_csv(args.mobius_data, comment="#"),
        pd.read_csv(args.wilson_data, comment="#"),
    )

    mobius_fit = fit_eft(mobius_data, "mobius")
    wilson_fit = fit_eft(wilson_data, "wilson")
    write_fit_results(mobius_fit, wilson_fit, args.output_file)


if __name__ == "__main__":
    main()
