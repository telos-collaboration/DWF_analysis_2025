from argparse import ArgumentParser, FileType
from collections import defaultdict

import pandas as pd

from .provenance import get_basic_metadata, text_metadata


def get_args():
    parser = ArgumentParser(description="Concatenate compatible CSV files into one")
    parser.add_argument(
        "csv_files",
        metavar="csv_file",
        nargs="+",
        help="Filenames of CSV files to concatenate",
    )
    parser.add_argument(
        "--output_file",
        type=FileType("w"),
        default="-",
        help="Where to place resulting concatenated CSV",
    )
    return parser.parse_args()


def concatenate(data):
    grouped_data = defaultdict(list)
    for datum in data:
        grouped_data[tuple(datum.columns)].append(datum)

    disjoint_data = [
        pd.concat(dataset).set_index("name") for dataset in grouped_data.values()
    ]
    if len(disjoint_data) == 0:
        return pd.DataFrame()
    if len(disjoint_data) == 1:
        return disjoint_data[0]
    return disjoint_data[0].join(disjoint_data[1:])


def main():
    args = get_args()
    data = [pd.read_csv(filename, comment="#") for filename in args.csv_files]
    concatenated_data = concatenate(data)
    print(text_metadata(get_basic_metadata()), file=args.output_file)
    print(concatenated_data.to_csv(), file=args.output_file)


if __name__ == "__main__":
    main()
