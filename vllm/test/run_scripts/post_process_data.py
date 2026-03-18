import os
import pandas as pd
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: post_process_data.py <csv_file>")
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
df = df.sort_values(by=df.columns[:5].tolist(), ascending=True)

base = os.path.splitext(csv_file)[0]
order_file = base + "_ordered.csv"
df.to_csv(order_file, index=False)

html_file = base + ".html"
df.to_html(html_file, index=False)
