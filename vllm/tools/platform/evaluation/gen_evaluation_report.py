import csv
import re
import sys

# -------------------------------
# Compile patterns once
# -------------------------------
P2P_SECTION_PATTERNS = {
    "Unidirectional Write": re.compile(r"Bandwidth Write : Device\( 0 \)->Device\( 1 \)"),
    "Unidirectional Read": re.compile(r"Bandwidth Read : Device\( 0 \)<-Device\( 1 \)"),
    "Bidirectional Write": re.compile(r"Bandwidth Write : Device\( 0 \)<->Device\( 1 \)"),
    "Bidirectional Read": re.compile(r"Bandwidth Read : Device\( 0 \)<->Device\( 1 \)"),
}
P2P_ROW_PATTERN = re.compile(r"^\s*([\d]+\s*[KM]?B)\s*:\s*([\d\.]+)\b")
GBPS_PATTERN = re.compile(r"([\d\.]+) GB/s")
FLOAT8_PATTERN = re.compile(r"float8\s*:\s*([\d\.]+) GB/s")
FLOAT16_PATTERN = re.compile(r"float16\s*:\s*([\d\.]+) GB/s")
GEMM_INT8_PATTERN = re.compile(r"Average performance:\s*([\d\.]+)TF")
CCL_TEST_PATTERN = re.compile(r"benchmarking:\s*(allreduce|allgather|alltoall)", re.I)

P2P_KEEP_SIZES = {"128 MB", "256 MB"}


def normalize_size(raw_size):
    return re.sub(r"\s+", " ", raw_size.strip())


def match_p2p_section(line):
    for section, pattern in P2P_SECTION_PATTERNS.items():
        if pattern.search(line):
            return section
    return None


# -------------------------------
# Parse all benchmark data in one pass
# -------------------------------
def parse_all_benchmarks(lines, target_bytes=134217728):
    results = []

    # p2p state
    current_p2p_section = None
    p2p_seen_data = False

    # GPU memory bandwidth state
    h2d = d2h = d2d_float8 = d2d_float16 = None
    in_global_memory_block = False

    # gemm int8 state
    in_int8_block = False
    gemm_recorded = False

    # oneCCL state
    current_ccl_test = None

    for line in lines:
        stripped = line.strip()

        # Section transition for P2P (single check per line).
        matched_p2p_section = match_p2p_section(line)
        if matched_p2p_section:
            current_p2p_section = matched_p2p_section
            p2p_seen_data = False
            continue

        # Parse P2P rows while inside a section.
        if current_p2p_section:
            if stripped == "":
                current_p2p_section = None
                p2p_seen_data = False
            else:
                match = P2P_ROW_PATTERN.search(line)
                if match:
                    p2p_seen_data = True
                    size = normalize_size(match.group(1))
                    if size in P2P_KEEP_SIZES:
                        bw = float(match.group(2))
                        results.append(["p2p", current_p2p_section, size, bw])
                elif p2p_seen_data:
                    current_p2p_section = None
                    p2p_seen_data = False

        # Parse GPU memory bandwidth.
        if "GPU Copy Host to Shared Memory" in line:
            match = GBPS_PATTERN.search(line)
            if match:
                h2d = float(match.group(1))
        elif "GPU Copy Shared Memory to Host" in line:
            match = GBPS_PATTERN.search(line)
            if match:
                d2h = float(match.group(1))
        elif "Global memory bandwidth" in line:
            in_global_memory_block = True
        elif in_global_memory_block and stripped == "":
            in_global_memory_block = False
        elif in_global_memory_block:
            if "float8" in line:
                match = FLOAT8_PATTERN.search(line)
                if match:
                    d2d_float8 = float(match.group(1))
            elif "float16" in line:
                match = FLOAT16_PATTERN.search(line)
                if match:
                    d2d_float16 = float(match.group(1))

        # Parse GEMM int8.
        if not gemm_recorded:
            if "matrix multiplication" in line and "int8 precision" in line:
                in_int8_block = True
            elif in_int8_block and "Average performance" in line:
                match = GEMM_INT8_PATTERN.search(line)
                if match:
                    results.append(["gemm", "int8", "", float(match.group(1))])
                    gemm_recorded = True
                    in_int8_block = False

        # Parse oneCCL bus bandwidth.
        clean_line = line.lstrip("# ").strip()
        match = CCL_TEST_PATTERN.match(clean_line)
        if match:
            current_ccl_test = match.group(1).lower()
            continue

        if current_ccl_test and re.match(r"^\d", clean_line):
            cols = clean_line.split()
            if len(cols) >= 9:
                bytes_val = int(cols[0])
                busbw_val = float(cols[8])
                if bytes_val == target_bytes:
                    results.append(["1ccl", current_ccl_test, "128MB", busbw_val])
                    current_ccl_test = None

    if h2d is not None:
        results.append(["GPU memory bandwidth", "H2D", "", h2d])
    if d2h is not None:
        results.append(["GPU memory bandwidth", "D2H", "", d2h])
    if d2d_float8 is not None:
        results.append(["GPU memory bandwidth", "D2D", "float8", d2d_float8])
    if d2d_float16 is not None:
        results.append(["GPU memory bandwidth", "D2D", "float16", d2d_float16])

    return results


# -------------------------------
# Load reference values
# -------------------------------
def load_reference(reference_file):
    reference = {}
    with open(reference_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
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

    all_results = parse_all_benchmarks(lines)
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
