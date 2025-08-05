from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from . import plots

parser = ArgumentParser(description="Plot a gradient flow history")
parser.add_argument("datafile", help="Filename to read and plot")
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
args = parser.parse_args()

plots.set_styles(args)

# Load data from files
data1 = np.loadtxt(args.datafile)

# Set up plot
fig, ax = plt.subplots(figsize=(4.5, 3.0))

# Define line styles
line_style1 = "purple"
line_style2 = "#29BCC1"
line_style3 = "#4581A9"
line_style4 = "orange"

# Plot lines
ax.plot(
    data1[:, 0], data1[:, 1], label="$\\beta = 6.7$", color=line_style1, linewidth=3.5
)

ax.set_xlabel("$t/a^2$")
ax.set_ylabel(r"${\cal E}(t)$")

# Add legend
ax.legend(loc="best")

# Fill curves between lines
ax.fill_between(
    data1[:, 0],
    data1[:, 1] - data1[:, 3],
    data1[:, 1] + data1[:, 3],
    color=line_style1,
    alpha=0.35,
)

plots.save_or_show(fig, args.output_file)
