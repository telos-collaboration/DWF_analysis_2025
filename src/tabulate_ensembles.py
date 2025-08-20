#!/usr/bin/env python3

import pandas as pd

from .tables import table_main, basic_formatter, format_float_column


def format_columns(data):
    return pd.DataFrame(
        {
            "Ensemble": data["name"],
            r"$\beta$": format_float_column(data, "beta", 2),
            "$L_s$": data["Ls"],
            "$N_t$": data["Nt"],
            "$N_s$": data["Nx"],
            "$am_0": format_float_column(data, "mF", 2),
            "$w_0/a$": basic_formatter(data, "w_0", "w_0_error"),
            r"$\langle Q_L(w_0^2)\rangle$": basic_formatter(data, "<Q>", "<Q>_err"),
            r"$\tau_{\mathrm{int}}^Q$": basic_formatter(data, "tau_Q", "tau_Q_err"),
            r"$am_{\mathrm{res}}$": basic_formatter(data, "mres", "mres_err"),
        }
    )


if __name__ == "__main__":
    table_main(format_columns, "basic ensemble and Wilson flow")
