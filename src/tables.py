#!/usr/bin/env python3

from argparse import ArgumentParser, FileType

from format_multiple_errors import format_column_errors
import pandas as pd

from .provenance import get_basic_metadata, text_metadata


def format_float_column(data, value_column, decimal_places=None):
    if decimal_places is None:
        template = "{value}"
    else:
        template = f"{{value:.0{decimal_places}}}"
    return pd.Series([template.format(value=value) for value in data[value_column]])


def basic_formatter(data, value_column, error_column):
    return format_column_errors(
        value_column, error_column, df=data, abbreviate=True, latex=True
    )


def get_args(description=None):
    if description is None:
        description = ""
    else:
        description = f"{description} "

    parser = ArgumentParser(description=f"Out {description}data as LaTeX table")
    parser.add_argument("datafile", help=f"CSV file containing {description}data")
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
        dataframe.style.hide(axis=0).to_latex(
            hrules=True,
            column_format="|c" * len(dataframe.columns) + "|",
        ),
        file=output_file,
    )


def table_main(format_columns, description=None):
    args = get_args(description)
    data = pd.read_csv(args.datafile, comment="#")
    output_data(format_columns(data), args.output_file)
