from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from . import plots

parser = ArgumentParser(description="Plot GMOR and vector-pseudoscalar mass ratio")
parser.add_argument(
    "--Q_history",
    help="File containing topological charges as a function of trajectory index",
)
parser.add_argument(
    "--Q_histogram", help="File containing histogram of topological charges"
)
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
args = parser.parse_args()

plots.set_styles(args)


# Load data from first file
data1 = np.loadtxt(args.Q_history)
x = data1[:, 0]
y = data1[:, 1]

# Create first plot
fig, (ax1, ax2) = plt.subplots(
    figsize=(3, 2), ncols=2, width_ratios=(2, 1), sharey=True
)
ax1.plot(x, y, label=r"$\beta = 6.9, \, am_0 = 0.05$")
ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.6)

ax1.text(
    0.02,
    0.98,
    r"$\langle Q_L \rangle$ = -0.00084(81)",
    transform=ax1.transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
)
print(np.mean(y), np.std(y))
print(np.mean(y), np.std(y))
ax1.set_xlabel("Trajectories")
ax1.set_ylabel("$Q_L(w^2_0)$")
ax1.tick_params(axis="both", which="major")
ax1.set_ylim([-1.2 * np.max(np.abs(y)), 1.2 * np.max(np.abs(y))])
ax1.legend(loc="lower left", frameon=False)

# Load data from second file
data2 = np.loadtxt(args.Q_histogram)
# Normalize the data for histogram
data2_norm = data2

# Create second plot
bin_range = (np.min(data2_norm), np.max(data2_norm))
n, bins, patches = ax2.hist(
    data2_norm,
    bins=8,
    range=bin_range,
    orientation="horizontal",
    density=True,
    linewidth=0.5,
    color="white",
    edgecolor="darkblue",
)
for patch in patches:
    patch.set_linestyle("-")
    patch.set_linewidth(1)
ax2.tick_params(axis="both", which="major")
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticklabels([])
ax2.tick_params(axis="x", which="major", bottom=False, labelbottom=False)

# Fit histogram with normal distribution
mu, std = norm.fit(data2_norm)
x_fit = np.linspace(bin_range[0], bin_range[1], 100)
p_fit = norm.pdf(x_fit, mu, std)
ax2.plot(p_fit, x_fit, "r-", linewidth=2)

# Set y limits for the fit line
ymin, ymax = ax2.get_ylim()
y_fit = np.linspace(ymin, ymax, 100)
x_fit_range = norm.pdf(y_fit, mu, std)
ax2.plot(x_fit_range, y_fit, color="orange", linestyle="-", linewidth=2)

# Calculate the reduced chi-square
chi_square = np.sum(
    ((data2_norm - norm.pdf(data2_norm, mu, std)) / np.std(data2_norm)) ** 2
)
dof = len(data2_norm) - 3  # Degrees of freedom
reduced_chi_square = chi_square / dof

# Print the value of the reduced chi-square
print("Reduced Chi-square:", reduced_chi_square)

# Save the figure in PDF format with dpi=300 and specified size
plots.save_or_show(fig, args.output_file)
