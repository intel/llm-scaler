import argparse
import os
import pandas as pd


def process_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    config_sort_columns = ["Date", "Version", "Model", "Tag", "Batch Size"]
    available_config_columns = [c for c in config_sort_columns if c in df.columns]
    if available_config_columns:
        df = df.sort_values(by=available_config_columns, ascending=True)

    base = os.path.splitext(csv_file)[0]
    order_file = base + "_ordered.csv"
    df.to_csv(order_file, index=False)

    html_file = base + ".html"
    df.to_html(html_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process one or more CSV benchmark files.")
    parser.add_argument("csv_files", nargs="+", help="CSV files to process.")
    args = parser.parse_args()
    for csv_file in args.csv_files:
        process_csv(csv_file)
