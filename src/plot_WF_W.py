from argparse import ArgumentParser

from flow_analysis.readers import read_flows_grid
from flow_analysis.measurements.scales import compute_wt_t

import matplotlib.pyplot as plt

from . import plots

parser = ArgumentParser(description="Plot the scale W(t) for a gradient flow history")
parser.add_argument("datafile", help="Filename to read and plot")
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
parser.add_argument(
    "--ensemble_label", default=None, help="Label to use in plot legend"
)
args = parser.parse_args()

plots.set_styles(args)

# Load data from files
flows = read_flows_grid(args.datafile, check_consistency=False)
w_value, w_error = compute_wt_t(flows, operator="plaq")

# Set up plot
fig, ax = plt.subplots(figsize=(4.5, 3.0))

# Plot lines
ax.plot(
    flows.times[1:-1], w_value, label=args.ensemble_label, color="C0", linewidth=3.5
)

ax.set_xlabel("$t/a^2$")
ax.set_ylabel(r"${\cal W}(t)$")
ax.set_title(args.ensemble_label)

# Add legend
ax.legend(loc="best")

# Fill curves between lines
ax.fill_between(
    flows.times[1:-1],
    w_value + w_error,
    w_value - w_error,
    color="C0",
    alpha=0.35,
)

plots.save_or_show(fig, args.output_file)
