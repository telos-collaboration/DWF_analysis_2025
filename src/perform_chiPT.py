from argparse import ArgumentParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from . import plots


# Define the modified fitting function for mM_NLO with linear and quadratic terms in a
def mM_NLO(m2_chi, Lm0, Wm0, Wm1, m2_PS, a):
    return m2_chi * (1 + Lm0 * m2_PS) + Wm0 * a + Wm1 * a**2


def perform_fit(data):
    a = data["a"]
    a_err = data["a_err"]
    m2_PS = data["m_PS"] ** 2 * (1 / a) ** 2
    m2_PS_err = m2_PS * np.sqrt(
        (2 * data["m_PS_err"] / data["m_PS"]) ** 2 + (2 * a_err / a) ** 2
    )
    m2_V = data["m_V"] ** 2 * (1 / a) ** 2
    m2_V_err = m2_V * np.sqrt(
        (2 * data["m_V_err"] / data["m_V"]) ** 2 + (2 * a_err / a) ** 2
    )
    # print(m2_PS)
    try:
        popt_mM, pcov_mM = curve_fit(
            lambda x, m2_chi, Lm0, Wm0, Wm1: mM_NLO(m2_chi, Lm0, Wm0, Wm1, m2_PS, x),
            a,
            m2_V,
            sigma=m2_V_err,
            absolute_sigma=True,
            maxfev=5000,
        )
        mM_errors = np.sqrt(np.diag(pcov_mM))
        print("Fitted parameters for mM_NLO:")
        print(f"m2_chi = {popt_mM[0]:.4f} ± {mM_errors[0]:.4f}")
        print(f"Lm0 = {popt_mM[1]:.4f} ± {mM_errors[1]:.4f}")
        print(f"Wm0 = {popt_mM[2]:.4f} ± {mM_errors[2]:.4f}")
        print(f"Wm1 = {popt_mM[3]:.4f} ± {mM_errors[3]:.4f}")
    except Exception as e:
        print("Fit for mM_NLO did not converge:", e)
        popt_mM, mM_errors = [np.nan] * 4, [np.nan] * 4

    return popt_mM, mM_errors, m2_PS, m2_PS_err, m2_V, m2_V_err


parser = ArgumentParser(description="Plot GMOR and vector-pseudoscalar mass ratio")
plots.add_styles_arg(parser)
plots.add_output_arg(parser)
plots.add_default_input_args(parser)
args = parser.parse_args()

plots.set_styles(args)

# Load the data
data = pd.read_csv(args.data, comment="#")

data = pd.DataFrame(
    {
        "a": 1 / data["w_0"],
        "a_err": data["w_0_error"] / data["w_0"] ** 2,
        "m_PS": data["g0g5"],
        "m_PS_err": data["g0g5_err"],
        "m_V": data["gi"],
        "m_V_err": data["gi_err"],
        "name": data["name"],
    }
)

popt_mM, mM_errors, m2_PS, m2_PS_err, m2_V, m2_V_err = perform_fit(data)

fig, ax = plt.subplots(figsize=(6, 4))

for colour_index, beta in sorted(set(data["beta"])):
    subset = data[data["beta"] == beta]
    m_PS_squared = (subset["w_0"] * subset["m_PS"]) ** 2
    m_V_squared = (subset["w_0"] * subset["m_V"]) ** 2
    m_PS_squared_err = 2 * (subset["w_0"] * subset["m_PS"]) * subset["m_PS_err"]
    m_V_squared_err = 2 * (subset["w_0"] * subset["m_V"]) * subset["m_V_err"] * 7.0

    ax.errorbar(
        m_PS_squared,
        m_V_squared,
        yerr=m_V_squared_err,
        xerr=m_PS_squared_err,
        fmt="o",
        color=f"C{colour_index}",
        label=f"$\\beta = {beta}$",
        elinewidth=1.3,
        markersize=6.1,
        markeredgecolor="black",
    )

if not np.isnan(popt_mM[0]):
    min_m2_PS = m_PS_squared.min()
    max_m2_PS = m_PS_squared.max()
    m2_PS_range = np.linspace(
        min_m2_PS - 0.1 * (max_m2_PS - min_m2_PS), max_m2_PS * 2.1, 500
    )
    m2_V_extrapolated = mM_NLO(
        popt_mM[0], popt_mM[1], popt_mM[2], popt_mM[3], m2_PS_range, 0
    )
    m2_V_upper = m2_V_extrapolated + 0.5 * mM_errors[0]
    m2_V_lower = m2_V_extrapolated - 0.5 * mM_errors[0]
    shift = 0.0
    plt.plot(
        m2_PS_range,
        m2_V_extrapolated + shift,
        "black",
        linestyle="--",
        linewidth=1.2,
        label="Extrapolated fit ($a = 0$)",
    )
    plt.fill_between(
        m2_PS_range, m2_V_lower + shift, m2_V_upper + shift, color="gray", alpha=0.3
    )


ax.set_xlabel("$(w_0 m_{\\rm PS})^2$", fontsize=15)
ax.set_ylabel("$(w_0 m_{\\rm V})^2$", fontsize=15)
ax.legend(fontsize=12, loc="best")
plots.save_or_show(fig, args.output_file)
