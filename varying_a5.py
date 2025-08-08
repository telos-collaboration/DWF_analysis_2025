import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the exponential function to fit
def lin_func(x, a, b):
    return a + (b * x)

plt.style.use("paperdraft.mplstyle")

# Load the data from the file
data = np.loadtxt('varying_a5.txt')

# Extract columns: x, y, and err_y
x = data[:, 0]
y = data[:, 1]
err_y = data[:, 2]


# Create the plot
plt.figure(figsize=(6, 4))
plt.errorbar(x, y, yerr=err_y, fmt='o', color='orange', linestyle='--', linewidth=1.0, markersize=4.9, elinewidth=2.2, capsize=5, label='Data')
#plt.plot(x_fit, y_fit, '--',color="#1f77b4", label=f'Fit: $y = {a_fit:.2e} \cdot e^{{{b_fit:.2e} x}}$')
#plt.yscale('log')
#plt.ylim(0.33,0.35)

plt.yscale('log')
# Add labels and title
plt.xlabel('$a_5 / a$', fontsize=20)
plt.ylabel('$am_{\\rm res}$', fontsize=20)
#plt.legend()
plt.grid(linestyle='--')

plt.title('$L_s = 8, \, \\beta = 6.8, \, am_5 = 1.8, \, \\alpha = 2.0, \, am_0 = 0.06$', fontsize=18)

# Show the plot
#plt.show()
plt.savefig('./varying_a5.pdf', dpi=130, bbox_inches='tight')

