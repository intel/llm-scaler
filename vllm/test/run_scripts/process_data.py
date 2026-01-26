import os
from datetime import datetime
import unicodedata  # 处理 ASCII 码的包
import argparse
import re
from typing import List
from script_config import ANALYSIS_PATH

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def get_num(line: str, index: int = 0) -> float:
    candidates = [float(i) if '.' in i else int(i) for i in line.split() if is_number(i)]
    return candidates[index] if index < len(candidates) else 0


def parse_model_and_tag(text: str):
    pattern = r"^(.+?)(?:[@:+])(.*)$"
    m = re.match(pattern, text.strip())
    if not m:
        return text, ""

    model, tag = m.group(1).strip(), m.group(2).strip()
    tag = tag if tag != "" else None
    return model, tag

def parse_batch_size(text: str):
    return re.findall(r"\d+", text)[0]

def extract_config_info(path: str, add_config_header) -> str:
    if not add_config_header:
        return ",,,,,"
    
    parts = path.strip().split(os.sep)
    for i in range(len(parts)):
        part = parts[i]
        m = re.match(r"^(\d{4}_\d{2}_\d{2})-(.+)$", part)
        if not m:
            continue
        date, version = m.group(1), m.group(2)
        if i+2 >=len(parts):
            return ',,,,,'

        model_and_tag = parts[i + 1]
        model,tag = parse_model_and_tag(model_and_tag)
        batch_size = parse_batch_size(parts[i+2])
        return ", ".join([date, version, model, tag, batch_size])+", "
    return ',,,,,'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LLM Scaler Benchmark Process Data Config")
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--add_config_header", type=bool, required=False)
    parser.add_argument("--format", type=str, required=False)
    parser.add_argument("--output", type=str, required=False)
    args = parser.parse_args()
    raw_data = args.raw_data
    add_config_header = args.add_config_header
    format = args.format
    output = args.output

    results = []
    current_group = []

    with open(raw_data, encoding="UTF-8") as f:
        for dataline in f:
            dataline = dataline.strip()
            # if dataline.startswith('Running benchmark with batch size'):
            #     current_group = [get_num(dataline)]
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

    config_info = extract_config_info(path=raw_data,add_config_header=add_config_header)
    if not output:
        output = ANALYSIS_PATH
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

    if add_config_header:
        headers = config_headers+result_headers
    else:
        headers = result_headers
    

    with open(f'{filename}_unrounded.csv', 'a') as f:
        if f.tell() == 0:
            f.write(", ".join(headers) + '\n')
        for result in results:
            f.write(config_info + ", ".join(map(str, result)) + '\n')

    with open(f'{filename}.csv', 'a') as f:
        if f.tell() == 0:
            f.write(", ".join(headers) + '\n')
        for result in results:
            f.write(config_info + ", ".join(f'{value:.2f}' if isinstance(value, float) else str(value) for value in result) + '\n')


