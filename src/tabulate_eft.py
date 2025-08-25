#!/usr/bin/env python3

from format_multiple_errors import format_multiple_errors
import pandas as pd

from .tables import table_main, basic_formatter, format_float_column


def format_observable(column):
    return [f"${key[0]}_{{\\mathrm{{{key[1:]}}}}}$" for key in column]


def expand_observable(obs_name):
    return obs_name, f"{obs_name}_err", f"{obs_name}_systematic_error"


def missing_formatter(data, *columns):
    return [
        "---"
        if pd.isna(value)
        else "${}$".format(
            format_multiple_errors(value, *errors, abbreviate=True, latex=True)
        )
        for value, *errors in data[list(columns)].itertuples(index=False)
    ]


def format_columns(data):
    breakpoint()
    return pd.DataFrame(
        {
            "F": data["formulation"],
            "$X$": format_observable(data["state"]),
            r"$(w_0 X^{\chi, \mathrm{F}})^2$": basic_formatter(
                data, *expand_observable("w0X_squared")
            ),
            r"$L_{X}^{0,\mathrm{F}}$": basic_formatter(data, *expand_observable("L0X")),
            r"$L_{X}^{1,\mathrm{F}}$": missing_formatter(
                data, *expand_observable("L1X")
            ),
            r"$W_{X}^{0,\mathrm{F}}$": basic_formatter(data, *expand_observable("W0X")),
            r"$\chi^2/N_{\textrm{d.o.f.}}$": format_float_column(data, "chisquare", 2),
        }
    )


if __name__ == "__main__":
    table_main(format_columns, "EFT fit results")
