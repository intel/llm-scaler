"""Release admission stays in llm-scaler and does not import Torch."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / 'ComfyUI-OmniXPU/aimdo_release_compat.py'
spec = importlib.util.spec_from_file_location('aimdo_release_compat_test', SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize('version,accepted', [('2.14.0+xpu', True), ('2.13.0+xpu', False),
    ('2.14.0+cpu', False), ('2.14.0.dev20261008+xpu', False)])
def test_release_policy_has_one_supported_release(version, accepted):
    if accepted:
        release.validate_torch_release(version)
    else:
        with pytest.raises(release.UnsupportedRelease, match='llm-scaler AIMDO diagnostic supports Torch release'):
            release.validate_torch_release(version)


def test_build_script_checks_release_without_torch_installation(tmp_path):
    # A shadow package would fail on import. The explicit release CLI needs no
    # Torch module, native library, ABI manifest or device initialization.
    (tmp_path / 'torch.py').write_text('raise RuntimeError("Torch imported")\n')
    for version, expected in [('2.14.0+xpu', 0), ('2.13.0+xpu', 1)]:
        result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--torch-version', version],
                                cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path)},
                                capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
