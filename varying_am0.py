import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the exponential function to fit
def lin_func(x, a, b):
    return a + (b * x)

plt.style.use("paperdraft.mplstyle")

# Load the data from the file
data = np.loadtxt('varying_am0.txt')

# Extract columns: x, y, and err_y
x = data[:, 0]
y = data[:, 1]
err_y = data[:, 2]

# Perform the fit
popt, pcov = curve_fit(lin_func, x, y, sigma=err_y, absolute_sigma=True)

# Extract the parameters
a_fit, b_fit = popt

# Generate x values for the fit line
x_fit = np.linspace(min(x), max(x), 500)
y_fit = lin_func(x_fit, a_fit, b_fit)
x_fit2 = np.linspace(min(x) - 0.05, max(x) + 0.05, 500)


# Create the plot
plt.figure(figsize=(6, 4))
plt.errorbar(x, y, yerr=err_y, fmt='o', markersize=9.9, elinewidth=2.2, capsize=5, label='Data', markerfacecolor='none')
plt.plot(x_fit, y_fit, '--',color="#1f77b4", label=f'Fit: $y = {a_fit:.2e} \cdot e^{{{b_fit:.2e} x}}$')
#plt.yscale('log')
plt.ylim(0.33,0.35)

# Add labels and title
plt.xlabel('$am_0$', fontsize=15)
plt.ylabel('$am_{\\rm res}$', fontsize=15)
#plt.legend()
#plt.grid(linestyle='--')

# Show the plot
#plt.show()
plt.savefig('./varying_am0.pdf', dpi=130, bbox_inches='tight')

