import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

# Activating text rendering by LaTeX
plt.style.use("paperdraft.mplstyle")

# Check if the flag '--dp' is present
use_dp_flag = '--dp' in sys.argv

# Define the data file paths based on the flag
data1_path = './top_charges_b68-am08_with_index.txt'
data2_path = './top_charges_b68-am08.txt'
if use_dp_flag:
    data1_path = './top_charges_b68-am08_with_index.txt'
    data2_path = './top_charges_b68-am08.txt'

# Load data from first file
data1 = np.loadtxt(data1_path)
x = data1[:, 0]
y = data1[:, 1]

# Create first plot
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(x, y, label='$\\beta = 6.9, \, am_0 = 0.05$')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.6)
'''
ax1.text(0.02, 0.98, r'$\langle Q_L \rangle$ = {:.3f}({:.0f})'.format(
    np.mean(y), 1000 * np.std(y) / np.sqrt(10)),
    transform=ax1.transAxes, fontsize=12, verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
'''
ax1.text(0.02, 0.98, r'$\langle Q_L \rangle$ = -0.00084(81)',
    transform=ax1.transAxes, fontsize=12, verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
print(np.mean(y), np.std(y))
print(np.mean(y), np.std(y))
ax1.set_xlabel('Trajectories', fontsize=12)
ax1.set_ylabel('$Q_L(w^2_0)$', fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_ylim([-1.2 * np.max(np.abs(y)), 1.2 * np.max(np.abs(y))])
ax1.legend(loc='lower left', fontsize=12, frameon=False)

# Load data from second file
data2 = np.loadtxt(data2_path)
# Normalize the data for histogram
data2_norm = data2

# Create second plot
ax2 = fig.add_axes([0.65, 0.1, 0.3, 0.8])
bin_range = (np.min(data2_norm), np.max(data2_norm))
n, bins, patches = ax2.hist(data2_norm, bins=8, range=bin_range,
                            orientation='horizontal', density=True,
                            linewidth=0.5, color='white', edgecolor='darkblue')
for patch in patches:
    patch.set_linestyle('-')
    patch.set_linewidth(1)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticklabels([])
ax2.tick_params(axis='x', which='major', bottom=False, labelbottom=False)

# Fit histogram with normal distribution
mu, std = norm.fit(data2_norm)
x_fit = np.linspace(bin_range[0], bin_range[1], 100)
p_fit = norm.pdf(x_fit, mu, std)
ax2.plot(p_fit, x_fit, 'r-', linewidth=2)

# Set y limits for the fit line
ymin, ymax = ax2.get_ylim()
y_fit = np.linspace(ymin, ymax, 100)
x_fit_range = norm.pdf(y_fit, mu, std)
ax2.plot(x_fit_range, y_fit, color='orange', linestyle='-', linewidth=2)

# Calculate the reduced chi-square
chi_square = np.sum(((data2_norm - norm.pdf(data2_norm, mu, std)) / np.std(data2_norm)) ** 2)
dof = len(data2_norm) - 3  # Degrees of freedom
reduced_chi_square = chi_square / dof

# Print the value of the reduced chi-square
print("Reduced Chi-square:", reduced_chi_square)

# Adjust subplots and show the plot
fig.subplots_adjust(left=0.10, right=0.6, bottom=0.1, top=0.9)

# Save the figure in PDF format with dpi=300 and specified size
plt.savefig('./topological_charge_b68.pdf', dpi=300, bbox_inches='tight')
