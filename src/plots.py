#!/usr/bin/env python3

from argparse import ArgumentParser
import hashlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


NUM_BOOTSTRAP_SAMPLES = 200


def get_args(description):
    parser = ArgumentParser(description=description)
    add_default_input_args(parser)
    add_output_arg(parser)
    add_styles_arg(parser)
    args = parser.parse_args()

    set_styles(args)
    return args


def add_styles_arg(parser):
    parser.add_argument(
        "--plot_styles",
        help="Matplotlib style file to use",
        default="styles/paperdraft.mplstyle",
    )


def add_default_input_args(parser):
    parser.add_argument(
        "--data",
        required=True,
        help="CSV file containing spectrum, gradient flow, and HMC timing results",
    )


def add_output_arg(parser, key=None, description=None):
    parser.add_argument(
        f"--output_file_{key}" if key else "--output_file",
        help="Where to place resulting plot"
        + (f" of {description}" if description else ""),
        required=True,
    )


def get_rng(data):
    filename_hash = hashlib.md5(np.asarray(data)).digest()
    seed = abs(int.from_bytes(filename_hash, "big"))
    return np.random.default_rng(seed)


def bootstrap_curve_fit(fit_form, x, y, sigma, p0=None):
    popt, _ = curve_fit(fit_form, x, y, sigma=sigma, p0=p0)
    popt_samples = []
    rng = get_rng([x, y, sigma])
    max_attempts = NUM_BOOTSTRAP_SAMPLES * 5
    for attempt_count in range(max_attempts):
        sample = rng.choice(x.index, size=len(x))
        try:
            popt_sample, _ = curve_fit(
                fit_form,
                x[sample],
                y[sample],
                sigma=sigma[sample],
                p0=popt,
            )
        except RuntimeError:
            continue
        popt_samples.append(popt_sample)
        if len(popt_samples) >= NUM_BOOTSTRAP_SAMPLES:
            break
    else:
        raise RuntimeError("Unable to collect enough bootstrap samples")

    if attempt_count > NUM_BOOTSTRAP_SAMPLES:
        print(
            "Bootstrapped with "
            f"{NUM_BOOTSTRAP_SAMPLES / attempt_count * 100}% efficiency"
        )

    return popt, np.std(popt_samples, axis=0)


def set_styles(args):
    plt.style.use(args.plot_styles)


def save_or_show(fig, filename):
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close(fig)
