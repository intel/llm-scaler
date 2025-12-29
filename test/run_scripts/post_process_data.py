import pandas as pd
import sys

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
df = df.sort_values(by=df.columns[:5].tolist(), ascending=True)

order_file = csv_file.split('.')[0] + "_ordered.csv"
df.to_csv(order_file, index=False)

html_file = csv_file.split('.')[0] + ".html"
df.to_html(html_file, index=False)