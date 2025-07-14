import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use("paperdraft.mplstyle")
plt.figure(figsize=(7, 4.5))

# Define the function to fit
def fit_function(L, m_inf, A):
    return m_inf * (1 + A * np.exp(-m_inf * L) / (m_inf * L)**(3/2))

# Read data from file
input_file = 'finite_V.txt'  # Change this to the path of your input file
data = np.loadtxt(input_file, skiprows=1)

L = data[:, 0]
m_L = data[:, 1]
err_m_L = data[:, 2]

# Perform the curve fit
initial_guess = [1.0, 1.0]  # Initial guess for m_inf and A
params, covariance = curve_fit(fit_function, L, m_L, sigma=err_m_L, absolute_sigma=True, p0=initial_guess)

m_inf, A = params
m_inf_err = np.sqrt(covariance[0, 0])

# Print fitted parameters
print(f"Fitted parameters:\nm_inf = {m_inf} ± {m_inf_err}\nA = {A}")

# Calculate m_inf * L
m_inf_L = m_inf * L

# Generate x values for the fit curve
x = np.linspace(0.9 * min(L), 1.1 * max(L), 1000)

# Plot m_inf * L vs m(L)
plt.errorbar(m_inf_L, m_L, yerr=err_m_L, fmt='o', markersize=4.8, elinewidth=1.8)
plt.plot(x * m_inf, fit_function(x, m_inf, A), label='Fitted line', linestyle='--', linewidth=1.8)
plt.axhline(y=m_inf, label=f'$am_{{\\rm inf}} = {m_inf:.4f} ({m_inf_err*10000:.0f})$', linewidth=0.8, color='black', linestyle='--')

plt.xlim(m_inf * 0.9 * min(L), m_inf * 1.1 * max(L))

plt.xlabel('$m_{\\rm inf} L$', fontsize=14)
plt.ylabel('$am_{\pi}$', fontsize=14)
plt.legend()
plt.title('Finite volume fit, $am_0 = 0.0075$', fontsize=14)
#plt.show()

plt.savefig('finite_V_extr_am0p0075.pdf', dpi=130, bbox_inches='tight')
