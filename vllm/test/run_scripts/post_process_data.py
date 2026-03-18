import pandas as pd
import sys

# TODO: No argument count validation. If no argument is passed, sys.argv[1] raises
# an IndexError. Add: if len(sys.argv) < 2: sys.exit("Usage: post_process_data.py <csv_file>")
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
df = df.sort_values(by=df.columns[:5].tolist(), ascending=True)

# TODO: csv_file.split('.')[0] only keeps the portion before the first dot, which
# truncates filenames like "result_2024_01_01.csv" to "result_2024_01_01" but breaks
# for names like "my.report.csv" → "my". Use os.path.splitext(csv_file)[0] instead.
order_file = csv_file.split('.')[0] + "_ordered.csv"
df.to_csv(order_file, index=False)

html_file = csv_file.split('.')[0] + ".html"
df.to_html(html_file, index=False)