import os
import csv
from datetime import datetime
import argparse
import re
from typing import List
from script_config import ANALYSIS_PATH

# Capture signed ints/floats, including scientific notation.
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _parse_num(token: str):
    if "." in token or "e" in token.lower():
        return float(token)
    return int(token)


def get_num(line: str, index: int = 0) -> float:
    if index == 0:
        match = NUMBER_PATTERN.search(line)
        return _parse_num(match.group(0)) if match else 0

    for pos, token in enumerate(NUMBER_PATTERN.findall(line)):
        if pos == index:
            return _parse_num(token)
    return 0


def parse_model_and_tag(text: str):
    pattern = r"^(.+?)(?:[@:+])(.*)$"
    m = re.match(pattern, text.strip())
    if not m:
        return text, ""

    model, tag = m.group(1).strip(), m.group(2).strip()
    tag = tag if tag else ""
    return model, tag

def parse_batch_size(text: str):
    matches = re.findall(r"\d+", text)
    return matches[0] if matches else ""

def extract_config_info(path: str, add_config_header) -> List[str]:
    if not add_config_header:
        return []
    
    parts = path.strip().split(os.sep)
    for i in range(len(parts)):
        part = parts[i]
        m = re.match(r"^(\d{4}_\d{2}_\d{2})-(.+)$", part)
        if not m:
            continue
        date, version = m.group(1), m.group(2)
        if i+2 >=len(parts):
            return ["", "", "", "", ""]

        model_and_tag = parts[i + 1]
        model,tag = parse_model_and_tag(model_and_tag)
        batch_size = parse_batch_size(parts[i+2])
        return [date, version, model, tag, batch_size]
    return ["", "", "", "", ""]

def process_file(raw_data: str, add_config_header: bool, output: str):
    results = []
    current_group = []

    if not os.path.exists(raw_data):
        raise FileNotFoundError(f"Input file not found: {raw_data}")
    with open(raw_data, encoding="UTF-8") as f:
        for dataline in f:
            dataline = dataline.strip()
            if dataline.startswith('Successful requests:'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Benchmark duration (s):'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Total input tokens:'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Total generated tokens:'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Request throughput (req/s):'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Output token throughput (tok/s):'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Total Token throughput (tok/s):'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Mean TTFT (ms):'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Mean TPOT (ms):'):
                current_group.append(get_num(dataline))
            elif dataline.startswith('Mean ITL (ms):'):
                current_group.append(get_num(dataline))
                results.append(current_group)
                current_group = []

    config_info = extract_config_info(path=raw_data,add_config_header=add_config_header)
    directory = '%s/%s/' % (output, datetime.now().strftime("%Y_%m_%d"))
    os.makedirs(directory, exist_ok=True)
    filename = directory + "result"

    config_headers = [
        "Date", "Version", "Model", "Tag", "Batch Size"
    ]

    result_headers = [
        "Successful Requests", 
        "Benchmark Duration (s)", "Total Input Tokens", 
        "Total Generated Tokens", "Request Throughput (req/s)", "Output Token Throughput (tok/s)", 
        "Total Token Throughput (tok/s)", "Mean TTFT (ms)", "Mean TPOT (ms)", "Mean ITL (ms)"
    ]

    headers = config_headers+result_headers if add_config_header else result_headers

    with open(f'{filename}_unrounded.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(headers)
        for result in results:
            row = config_info + list(map(str, result)) if add_config_header else list(map(str, result))
            writer.writerow(row)

    with open(f'{filename}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(headers)
        for result in results:
            rounded_result = [f'{value:.2f}' if isinstance(value, float) else str(value) for value in result]
            row = config_info + rounded_result if add_config_header else rounded_result
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Scaler Benchmark Process Data Config")
    parser.add_argument("--raw_data", nargs="+", required=True)
    parser.add_argument("--add_config_header", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--format", type=str, required=False)
    parser.add_argument("--output", type=str, required=False)
    args = parser.parse_args()
    add_config_header = args.add_config_header
    output = args.output
    if not output:
        output = ANALYSIS_PATH
    for raw_data in args.raw_data:
        process_file(raw_data=raw_data, add_config_header=add_config_header, output=output)
