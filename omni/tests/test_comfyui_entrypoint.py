"""Tests for the focused ComfyUI image entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ENTRYPOINT = Path(__file__).parents[1] / "entrypoints" / "start_comfyui.sh"


def _run_entrypoint(
    tmp_path: Path,
    *,
    reserve: str | None = None,
    extra_arguments: tuple[str, ...] = ("--disable-all-custom-nodes",),
):
    capture = tmp_path / "args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$OMNI_TEST_CAPTURE\"\n"
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    environment["OMNI_TEST_CAPTURE"] = str(capture)
    if reserve is None:
        environment.pop("OMNI_COMFYUI_RESERVE_VRAM_GB", None)
    else:
        environment["OMNI_COMFYUI_RESERVE_VRAM_GB"] = reserve
    completed = subprocess.run(
        ["bash", str(ENTRYPOINT), *extra_arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    arguments = capture.read_text().splitlines() if capture.exists() else []
    return completed, arguments


def test_entrypoint_reserves_four_gib_by_default(tmp_path):
    completed, arguments = _run_entrypoint(tmp_path)

    assert completed.returncode == 0
    assert arguments == [
        "/llm/ComfyUI/main.py",
        "--listen",
        "0.0.0.0",
        "--port",
        "8188",
        "--extra-model-paths-config",
        "/llm/configs/comfyui_host_models.yaml",
        "--input-directory",
        "/data/input",
        "--output-directory",
        "/data/output",
        "--user-directory",
        "/data/user",
        "--reserve-vram",
        "4",
        "--enable-dynamic-vram",
        "--enable-manager",
        "--disable-all-custom-nodes",
    ]


def test_entrypoint_loads_custom_nodes_by_default(tmp_path):
    completed, arguments = _run_entrypoint(tmp_path, extra_arguments=())

    assert completed.returncode == 0
    assert "--enable-manager" in arguments
    assert "--enable-dynamic-vram" in arguments
    assert "--disable-all-custom-nodes" not in arguments


def test_entrypoint_allows_explicit_reserve_override(tmp_path):
    completed, arguments = _run_entrypoint(tmp_path, reserve="6.5")

    assert completed.returncode == 0
    reserve_index = arguments.index("--reserve-vram")
    assert arguments[reserve_index + 1] == "6.5"


def test_entrypoint_rejects_invalid_reserve(tmp_path):
    completed, arguments = _run_entrypoint(tmp_path, reserve="four")

    assert completed.returncode == 2
    assert arguments == []
    assert "must be a nonnegative number" in completed.stderr
