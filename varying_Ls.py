import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the exponential function to fit
def exp_func(x, a, b):
    return a * np.exp(b * x)

plt.style.use("paperdraft.mplstyle")

# Load the data from the file
data = np.loadtxt('varying_Ls_alpha2.txt')

# Extract columns: x, y, and err_y
x = data[:, 0]
y = data[:, 1]
err_y = data[:, 2]

# Perform the fit
popt, pcov = curve_fit(exp_func, x, y, sigma=err_y, absolute_sigma=True)

# Extract the parameters
a_fit, b_fit = popt

# Number of bootstrap samples
n_bootstrap = 1000

# Initialize arrays to store bootstrap results
bootstrap_a = np.zeros(n_bootstrap)
bootstrap_b = np.zeros(n_bootstrap)

# Perform bootstrap resampling and fitting
for i in range(n_bootstrap):
    indices = np.random.choice(range(len(x)), size=len(x), replace=True)
    x_resample = x[indices]
    y_resample = y[indices]
    err_resample = err_y[indices]
    
    try:
        popt_resample, _ = curve_fit(exp_func, x_resample, y_resample, sigma=err_resample, absolute_sigma=True)
        bootstrap_a[i] = popt_resample[0]
        bootstrap_b[i] = popt_resample[1]
    except RuntimeError:
        continue  # Skip the sample if the fit fails

# Calculate the mean and standard deviation of the bootstrap results
a_mean = np.mean(bootstrap_a)
a_std = np.std(bootstrap_a)
b_mean = np.mean(bootstrap_b)
b_std = np.std(bootstrap_b)

# Generate x values for the fit line
x_fit = np.linspace(min(x), 12, 500)

# Calculate the fit line and the uncertainty bands
y_fit = exp_func(x_fit, a_mean, b_mean)
y_fit_upper = exp_func(x_fit, a_mean + 0.20*a_std, b_mean + 0.20*b_std)
y_fit_lower = exp_func(x_fit, a_mean - 0.20*a_std, b_mean - 0.20*b_std)

# Create the plot
plt.figure(figsize=(6, 4))
plt.errorbar(x, y, yerr=err_y, fmt='o', markersize=4.9, color='#ff7f00', elinewidth=2.2, capsize=5, label='Data')
plt.plot(x_fit, y_fit, '--', linewidth=1.7, color="#1f77b4", label='Fitted curve')
plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color="#1f77b4", alpha=0.15)

# Add labels and title
plt.xlabel('$L_s$', fontsize=20)
plt.ylabel('$am_{\\rm res}$', fontsize=20)
plt.yscale('log')
plt.legend()
plt.grid(linestyle='--')

# Print the estimated value at x=10
print(exp_func(10, a_fit, b_fit))

plt.title('$am_5 = 1.8, \, \\beta = 6.8, \, a_5 = 1.0, \, \\alpha = 2.0, \, am_0 = 0.06$', fontsize=18)

# Save and show the plot
plt.savefig('./varying_Ls.pdf', dpi=130, bbox_inches='tight')
#plt.show()
