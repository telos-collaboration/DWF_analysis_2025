from argparse import ArgumentParser, FileType
from functools import partial

import pandas as pd
from format_multiple_errors import format_column_errors
from .provenance import get_basic_metadata, text_metadata


def get_args():
    parser = ArgumentParser(description="Out spectrum data as LaTeX table")
    parser.add_argument("datafile", help="CSV file containing spectrum data")
    parser.add_argument(
        "--output_file",
        help="File in which to place output; defaults to stdout",
        type=FileType("w"),
        default="-",
    )
    return parser.parse_args()


def output_data(dataframe, output_file):
    print(text_metadata(get_basic_metadata(), comment_char="%"), file=output_file)
    print(
        dataframe.style.hide(axis=0).to_latex(hrules=True, column_format="|c|c|c|c|c|"),
        file=output_file,
    )


def format_columns(data):
    formatter = lambda value, error: format_column_errors(
        value, error, df=data, abbreviate=True, latex=True
    )
    return pd.DataFrame(
        {
            "Ensemble": data["name"]
            .str.replace(".*ens", "B", regex=True)
            .str.replace("_m", "M"),
            r"$am_{\mathrm{PS}}$": formatter("g0g5", "g0g5_err"),
            r"$am_{\mathrm{V}}$": formatter("gi", "gi_err"),
            r"$af_{\mathrm{PS}}$": formatter("fpi", "fpi_err"),
            r"$Z_A$": formatter("Z_A", "Z_A_err"),
        }
    )


def main():
    args = get_args()
    data = pd.read_csv(args.datafile, comment="#")
    output_data(format_columns(data), args.output_file)


if __name__ == "__main__":
    main()
