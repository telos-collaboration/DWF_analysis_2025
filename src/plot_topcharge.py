from argparse import ArgumentParser

from flow_analysis.readers import read_flows_grid
from flow_analysis.measurements.scales import measure_w0
from flow_analysis.measurements.Q import Q_mean, flat_bin_Qs, Q_fit
from flow_analysis.fit_forms import gaussian

import matplotlib.pyplot as plt
import numpy as np

from . import plots

parser = ArgumentParser(description="Plot topological charge history and histogram")
parser.add_argument("datafile", help="Filename to read and plot")
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
parser.add_argument(
    "--ensemble_label", default=None, help="Label to use in plot legend"
)
parser.add_argument(
    "--W0", type=float, default=0.28125, help="Reference scale for W0 computation"
)
args = parser.parse_args()

plots.set_styles(args)

# Load data from first file
flows = read_flows_grid(args.datafile, check_consistency=False)
w0_squared = measure_w0(flows, args.W0, "plaq").nominal_value ** 2

# Create first plot
fig, (history_ax, histogram_ax) = plt.subplots(
    figsize=(3, 2), ncols=2, width_ratios=(2, 1), sharey=True
)

history_ax.plot(
    flows.trajectories, flows.Q_history(w0_squared), label=args.ensemble_label
)
history_ax.axhline(y=0, color="black", linestyle="-", linewidth=0.6)

history_ax.text(
    0.02,
    0.98,
    f"$\\langle Q_L \\rangle$ = {Q_mean(flows, w0_squared)}",
    transform=history_ax.transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
)
history_ax.set_xlabel("Trajectories")
history_ax.set_ylabel("$Q_L(w^2_0)$")
history_ax.tick_params(axis="both", which="major")
history_ax.legend(loc="lower left", frameon=False)

top_charge_range, top_charge_counts = flat_bin_Qs(flows, w0_squared)
histogram_ax.step(top_charge_counts, top_charge_range, label="Histogram")

amplitude, fitted_top_charge, sigma = Q_fit(flows, w0_squared, with_amplitude=True)
smooth_top_charge_range = np.linspace(
    min(top_charge_range) - 0.5, max(top_charge_range) + 0.5, 1000
)
histogram_ax.plot(
    gaussian(
        smooth_top_charge_range,
        amplitude.nominal_value,
        fitted_top_charge.nominal_value,
        sigma.nominal_value,
    ),
    smooth_top_charge_range,
    label="Fit",
)

plots.save_or_show(fig, args.output_file)
