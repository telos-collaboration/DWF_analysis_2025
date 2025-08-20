import pandas as pd

from .tables import table_main, basic_formatter


def format_columns(data):
    return pd.DataFrame(
        {
            "Ensemble": data["name"],
            r"$am_{\mathrm{PS}}$": basic_formatter(data, "g0g5", "g0g5_err"),
            r"$am_{\mathrm{V}}$": basic_formatter(data, "gi", "gi_err"),
            r"$af_{\mathrm{PS}}$": basic_formatter(data, "fpi", "fpi_err"),
            r"$Z_A$": basic_formatter(data, "Z_A", "Z_A_err"),
        }
    )


if __name__ == "__main__":
    table_main(format_columns, "spectrum")
