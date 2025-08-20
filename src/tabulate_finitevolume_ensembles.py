#!/usr/bin/env python3

import pandas as pd

from .tables import table_main, basic_formatter, format_float_column


def format_columns(data):
    return pd.DataFrame(
        {
            r"$\beta$": format_float_column(data, "beta"),
            "$L_s$": data["Ls"],
            "$N_t$": data["Nt"],
            "$N_s$": data["Nx"],
            "$am_0": format_float_column(data, "mF"),
            r"$am_{\mathrm{PS}}$": basic_formatter(data, "g0g5", "g0g5_err"),
        }
    )


if __name__ == "__main__":
    table_main(format_columns, "basic ensemble and Wilson flow")
