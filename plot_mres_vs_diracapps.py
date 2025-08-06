import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches


# Read the data from the file
data = np.loadtxt("dirac_apps_vs_Ls.txt")

plt.style.use("paperdraft.mplstyle")

cb_colors = [cm.RdBu(i) for i in np.linspace(-0.02, 0.45, 10)]

# Extract columns: x, err_x, y, err_y
x = data[:, 0]
err_x = data[:, 1]
y = data[:, 2]
err_y = data[:, 3]

# Plot the data
plt.figure(figsize=(7, 5))

Ls = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]



for i in range(10):
    #print(cb_colors[i])
    # Plot with error bars
    plt.errorbar(x[i], y[i], xerr=err_x[i], yerr=err_y[i], fmt='o',color=cb_colors[i], capsize=4, label=f"$L_s = {Ls[i]}$", markersize=4.6, elinewidth=2.2)

# Add labels, grid, and legend
plt.xlabel("$am_{\\rm res}$", fontsize=18)
plt.ylabel("Dirac apps.", fontsize=18)
plt.title("Lattice $32 \\times 24^3, \, \\beta = 6.8, \, am_0 = 0.06$", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=15, ncol=2)
plt.ylim(0,150000)
plt.xscale('log')
# Show the plot
plt.savefig('./dirac_apps_vs_mres.pdf', dpi=300, bbox_inches='tight')
#plt.show()

