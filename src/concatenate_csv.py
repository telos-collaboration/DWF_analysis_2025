from argparse import ArgumentParser, FileType

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


def main():
    args = get_args()
    data = [pd.read_csv(filename, comment="#") for filename in args.csv_files]
    concatenated_data = pd.concat(data, ignore_index=True)
    print(text_metadata(get_basic_metadata()), file=args.output_file)
    print(concatenated_data.to_csv(index=False), file=args.output_file)


if __name__ == "__main__":
    main()
