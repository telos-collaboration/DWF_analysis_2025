import numpy as np
import matplotlib.pyplot as plt

# Activating text rendering by LaTeX
plt.style.use("paperdraft.mplstyle")


# Function to load data from file
def load_data(filename):
    data = np.loadtxt(filename)
    x, err_x, y, err_y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return x, err_x, y, err_y


# Function to calculate 1/y and propagate the error
def inverse_and_propagate_error(y, err_y):
    inv_y = 1 / y
    err_inv_y = err_y / (y**2)  # Error propagation for 1/y
    return inv_y, err_inv_y


# Function to calculate x^2 and propagate its error
def square_and_propagate_error(x, err_x):
    x_squared = x**2
    err_x_squared = 1 * x * err_x  # Error propagation for x^2
    return x_squared, err_x_squared


# Filenames for the datasets
filenames = [
    "NLO_w0_b6p7.txt",
    "NLO_w0_b6p9.txt",
    "NLO_w0_b7p2.txt",
    "NLO_w0_b7p4.txt",
    "NLO_w0_b7p5.txt",
]
# colors = ['green', 'red', 'blue', 'purple', 'black']
colors = ["b", "g", "orange", "purple", "red"]
labels = [
    "$\\beta = 6.9$",
    "$\\beta = 7.05$",
    "$\\beta = 7.2$",
    "$\\beta = 7.4$",
    "$\\beta = 7.5$",
]

# Initialize plot
plt.figure(figsize=(6, 4))

# Loop over datasets and plot
for filename, color, label in zip(filenames, colors, labels):
    # Load data
    x, err_x, y, err_y = load_data(filename)

    # Calculate x^2 and its propagated error
    x_squared, err_x_squared = square_and_propagate_error(x, err_x)

    # Calculate 1/y and propagated errors
    inv_y, err_inv_y = (y, err_y)

    # Plot with error bars (for x^2 instead of x)
    plt.errorbar(
        x_squared,
        inv_y,
        yerr=err_inv_y,
        xerr=err_x_squared,
        fmt="o",
        color=color,
        label=label,
        capsize=1,
        markersize=5.7,
        markerfacecolor="none",
    )

# Customize plot
plt.xlabel("$w^2_0 m^2_{\\rm PS}$")  # Update x-axis label for x^2
plt.ylabel("$a/w_0$")
# plt.ylim(0.20, 1.65)
# plt.xlim(0.00, 0.85)
# plt.title('Plot of x^2 vs 1/y with Error Propagation')
plt.legend()
# plt.grid(True, linestyle='--')

# Show plot
# plt.show()
plt.title("Wilson fermions")
# Save the figure in PDF format with dpi=300 and specified size
plt.savefig("./chiral_aoverw0_vs_mPS.pdf", dpi=300, bbox_inches="tight")
