#!/usr/bin/env python3

import matplotlib.pyplot as plt


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


def set_styles(args):
    plt.style.use(args.plot_styles)


def save_or_show(fig, filename):
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close(fig)
