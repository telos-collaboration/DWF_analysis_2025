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
    parser.add_argument(
        "--metadata_file",
        default=None,
        help="Global metadata file to annotate each row with",
    )
    return parser.parse_args()


def strip_consistent_columns(left_df, right_df):
    left_df = left_df.copy()
    common_columns = list(set(left_df.columns).intersection(right_df.columns))
    for index, row in right_df.iterrows():
        if index in left_df.index:
            if not (left_df.loc[index][common_columns] == row[common_columns]).all():
                raise ValueError("Inconsistent metadata.")
        else:
            left_df.loc[index] = row[common_columns]

    return left_df, right_df.drop(columns=common_columns)


def join_consistently(data):
    joined_df = data[0]
    for right_df in data[1:]:
        left_df_consistent, right_df_stripped = strip_consistent_columns(
            joined_df, right_df
        )
        joined_df = left_df_consistent.join(right_df_stripped)

    return joined_df


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
    return join_consistently(disjoint_data)


def annotate(data, metadata_file):
    if metadata_file is None:
        return data

    metadata = pd.read_csv(metadata_file, comment="#")
    return data.merge(metadata, on="name")


def main():
    args = get_args()
    data = [pd.read_csv(filename, comment="#") for filename in args.csv_files]
    concatenated_data = concatenate(data)
    annotated_data = annotate(concatenated_data, args.metadata_file)
    print(text_metadata(get_basic_metadata()), file=args.output_file)
    print(annotated_data.to_csv(), file=args.output_file)


if __name__ == "__main__":
    main()
