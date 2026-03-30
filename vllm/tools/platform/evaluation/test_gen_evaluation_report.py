from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).with_name("gen_evaluation_report.py")
SPEC = importlib.util.spec_from_file_location("gen_evaluation_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
parse_p2p_bandwidth = MODULE.parse_p2p_bandwidth


def test_parse_p2p_bandwidth_parses_rows_after_section_header():
    fixture = Path(__file__).with_name("testdata_p2p_bandwidth.log")
    lines = fixture.read_text().splitlines()

    rows = parse_p2p_bandwidth(lines)
    parsed = {(category, subcategory, size): value for category, subcategory, size, value in rows}

    assert parsed[("p2p", "Unidirectional Write", "128 MB")] == 100.5
    assert parsed[("p2p", "Unidirectional Write", "256 MB")] == 110.25
    assert parsed[("p2p", "Unidirectional Read", "128 MB")] == 98.0
    assert parsed[("p2p", "Unidirectional Read", "256 MB")] == 99.5
    assert all(size in {"128 MB", "256 MB"} for _, _, size, _ in rows)
