from types import SimpleNamespace

import pytest

from omni_xpu_kernel import device


@pytest.mark.parametrize(
    ("device_id", "expected"),
    [
        (0xE210, "b60"),
        (0xE211, "b60"),
        (0xE223, "b70"),
        (0xE20B, "unknown"),
        (0xFFFF, "unknown"),
    ],
)
def test_python_device_classifier_uses_exact_ids(
    monkeypatch, device_id, expected
):
    native = SimpleNamespace(
        device=SimpleNamespace(
            classify_bmg_device_id=lambda value: {
                0xE210: "b60",
                0xE211: "b60",
                0xE223: "b70",
            }.get(value, "unknown")
        )
    )
    monkeypatch.setattr(device, "_load_extension", lambda: native)

    assert device.classify_bmg_device_id(device_id) == expected


def test_device_info_and_sku_forward_to_native(monkeypatch):
    native = SimpleNamespace(
        device=SimpleNamespace(
            info=lambda index: {
                "index": index,
                "device_id": 0xE210,
                "bmg_sku": "b60",
            },
            bmg_sku=lambda index: "b60" if index == 1 else "b70",
        )
    )
    monkeypatch.setattr(device, "_load_extension", lambda: native)

    assert device.info(3) == {
        "index": 3,
        "device_id": 0xE210,
        "bmg_sku": "b60",
    }
    assert device.bmg_sku(1) == "b60"
    assert device.bmg_sku(0) == "b70"
