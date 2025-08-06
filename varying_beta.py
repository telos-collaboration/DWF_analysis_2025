import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the linear function to fit
def lin_func(x, a, b):
    return a + (b * x)

plt.style.use("paperdraft.mplstyle")

# Load the data from the file
data = np.loadtxt('varying_beta.txt')

# Extract columns: x, y, and err_y
x = data[:, 0]
y = data[:, 1]
err_y = data[:, 2]

# Perform the fit
popt, pcov = curve_fit(lin_func, x, y, sigma=err_y, absolute_sigma=True)

# Extract the parameters and their standard deviations
a_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
a_err, b_err = perr
a_err = 10**5 * a_err
b_err = 10**4 * b_err
# Format the parameters with their errors
a_str = f'{a_fit:.5f}({(a_err):.0f})'
b_str = f'{b_fit:.4f}({b_err:.0f})'

# Generate x values for the fit line
x_fit = np.linspace(6.5, 7.4, 500)
y_fit = lin_func(x_fit, a_fit, b_fit)

# Calculate the confidence interval (1-sigma band)
y_fit_upper = lin_func(x_fit, a_fit + a_err, b_fit + b_err)
y_fit_lower = lin_func(x_fit, a_fit - a_err, b_fit - b_err)

# Create the plot
plt.figure(figsize=(6, 4))
plt.errorbar(x, y, yerr=err_y, fmt='o', markersize=9.9, elinewidth=2.2, capsize=5, label='Data', markerfacecolor='none')
#plt.plot(x_fit, y_fit, '--', color="#1f77b4", label=f'Fit: $y = {a_str} + {b_str} x$')
plt.plot(x_fit, y_fit, '--', color="#1f77b4")

print(lin_func(7.2, a_fit, b_fit))


# Plot the error band
#plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color="#1f77b4", alpha=0.3, label='1-sigma error band')

#plt.ylim(0.001250, 0.001550)
plt.ylim(0.0005, 0.0035)

# Add labels and title
plt.xlabel('$\\beta$', fontsize=20)
plt.ylabel('$am_{\\rm res}$', fontsize=20)
plt.legend()
plt.title('$L_s = 8, \, am_5 = 1.8, \, a_5 = 1.0, \, \\alpha = 2.0, \, am_0 = 0.06$', fontsize=18)
# Show the plot
#plt.show()
plt.savefig('./varying_beta.pdf', dpi=130, bbox_inches='tight')
