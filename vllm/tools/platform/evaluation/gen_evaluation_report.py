import re
import csv
import sys

# -------------------------------
# Parse P2P bandwidth, keep only 128MB and 256MB
# -------------------------------
def parse_p2p_bandwidth(lines):
    TARGETS = {
        "Unidirectional Write": r"Bandwidth Write : Device\( 0 \)->Device\( 1 \)",
        "Unidirectional Read": r"Bandwidth Read : Device\( 0 \)<-Device\( 1 \)",
        "Bidirectional Write": r"Bandwidth Write : Device\( 0 \)<->Device\( 1 \)",
        "Bidirectional Read": r"Bandwidth Read : Device\( 0 \)<->Device\( 1 \)",
    }
    keep_sizes = ["128 MB", "256 MB"]
    results = []

    i = 0
    while i < len(lines):
        line = lines[i]
        for name, pattern in TARGETS.items():
            if re.search(pattern, line):
                i += 1
                while i < len(lines) and "BW [GBPS]" in lines[i]:
                    match = re.search(r"([\d]+ ?[KM]?B):\s+([\d\.]+)", lines[i])
                    if match:
                        size = match.group(1).strip()
                        if size in keep_sizes:
                            bw = float(match.group(2))
                            results.append(["p2p", name, size, bw])
                    i += 1
                break
        i += 1

    return results

# -------------------------------
# Parse GPU memory bandwidth (H2D, D2H, D2D)
# -------------------------------
def parse_gpu_memory_bandwidth(lines):
    results = []
    h2d = d2h = d2d_float8 = d2d_float16 = None
    for i, line in enumerate(lines):
        if "GPU Copy Host to Shared Memory" in line:
            match = re.search(r"([\d\.]+) GB/s", line)
            if match: h2d = float(match.group(1))
        elif "GPU Copy Shared Memory to Host" in line:
            match = re.search(r"([\d\.]+) GB/s", line)
            if match: d2h = float(match.group(1))
        elif "Global memory bandwidth" in line:
            j = i + 1
            while j < len(lines) and lines[j].strip() != "":
                if "float8" in lines[j]:
                    match = re.search(r"float8\s*:\s*([\d\.]+) GB/s", lines[j])
                    if match: d2d_float8 = float(match.group(1))
                elif "float16" in lines[j]:
                    match = re.search(r"float16\s*:\s*([\d\.]+) GB/s", lines[j])
                    if match: d2d_float16 = float(match.group(1))
                j += 1
    if h2d is not None: results.append(["GPU memory bandwidth", "H2D", "", h2d])
    if d2h is not None: results.append(["GPU memory bandwidth", "D2H", "", d2h])
    if d2d_float8 is not None: results.append(["GPU memory bandwidth", "D2D", "float8", d2d_float8])
    if d2d_float16 is not None: results.append(["GPU memory bandwidth", "D2D", "float16", d2d_float16])
    return results

# -------------------------------
# Parse GEMM int8 performance
# -------------------------------
def parse_gemm_int8(lines):
    in_int8 = False
    for line in lines:
        if "matrix multiplication" in line and "int8 precision" in line:
            in_int8 = True
        elif in_int8 and "Average performance" in line:
            match = re.search(r"Average performance:\s*([\d\.]+)TF", line)
            if match:
                return [["gemm", "int8", "", float(match.group(1))]]
    return []

# -------------------------------
# Parse oneCCL benchmarks (allreduce/allgather/alltoall)
# Only extract busbw at 128MB
# -------------------------------
def parse_ccl_busbw(lines, target_bytes=134217728):
    results = []
    current_test = None
    pattern_test = re.compile(r"benchmarking:\s*(allreduce|allgather|alltoall)", re.I)
    for line in lines:
        clean_line = line.lstrip("# ").strip()
        m = pattern_test.match(clean_line)
        if m:
            current_test = m.group(1).lower()
            continue
        if current_test and re.match(r"^\d", clean_line):
            cols = clean_line.split()
            if len(cols) >= 9:
                bytes_val = int(cols[0])
                busbw_val = float(cols[8])
                if bytes_val == target_bytes:
                    results.append(["1ccl", current_test, "128MB", busbw_val])
                    current_test = None
    return results

# -------------------------------
# Load reference values
# -------------------------------
def load_reference(reference_file):
    reference = {}
    with open(reference_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 4:
                key = (row[0], row[1], row[2])
                reference[key] = row[3]
    return reference

# -------------------------------
# Main function
# -------------------------------
def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_log> <reference_csv> <output_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    reference_file = sys.argv[2]
    output_file = sys.argv[3]

    with open(input_file, "r") as f:
        lines = f.readlines()

    all_results = []
    all_results.extend(parse_p2p_bandwidth(lines))
    all_results.extend(parse_gpu_memory_bandwidth(lines))
    all_results.extend(parse_gemm_int8(lines))
    all_results.extend(parse_ccl_busbw(lines))

    reference = load_reference(reference_file)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Subcategory", "Data/Packet Size", "Measured (GB/s)", "Reference (GB/s)"])
        for row in all_results:
            key = (row[0], row[1], row[2])
            ref_val = reference.get(key, "")
            writer.writerow(row + [ref_val])

    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    main()
