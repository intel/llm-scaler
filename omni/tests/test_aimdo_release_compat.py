"""Release admission and native build provenance are owned by llm-scaler."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
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


@pytest.fixture
def sidecar(tmp_path):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('native release-export tests require a C compiler')

    def build(version='2.14.0+xpu', *, export=True):
        value = json.dumps(version) if version is not None else '0'
        name = 'aimdo_full_proxy_torch_version' if export else 'unrelated_export'
        library = tmp_path / 'native-owner.so'
        # There is deliberately no allocator install export. A passing check
        # must only read the release, without installing or exercising an XPU.
        source = f'const char *{name}(void) {{ return {value}; }}\n'
        subprocess.run([compiler, '-shared', '-fPIC', '-x', 'c', '-', '-o', str(library)],
                       input=source, capture_output=True, text=True, check=True)
        return library

    return build


@pytest.mark.parametrize('version', ['2.14.0+xpu', '2.13.0+xpu', 'unknown', None])
def test_native_export_must_match_installed_and_declared_release(sidecar, version):
    path = sidecar(version)
    if version == '2.14.0+xpu':
        release.validate_native_owner_release(
            path, installed_version='2.14.0+xpu', declared_version='2.14.0+xpu')
    else:
        with pytest.raises(release.UnsupportedRelease, match='native-owner was built for Torch release'):
            release.validate_native_owner_release(
                path, installed_version='2.14.0+xpu', declared_version='2.14.0+xpu')


def test_native_release_export_is_required(sidecar):
    with pytest.raises(release.UnsupportedRelease, match='Cannot read AIMDO native-owner Torch release'):
        release.validate_native_owner_release(
            sidecar(export=False), installed_version='2.14.0+xpu', declared_version='2.14.0+xpu')


def test_missing_native_library_is_rejected(tmp_path):
    with pytest.raises(release.UnsupportedRelease, match='Cannot read AIMDO native-owner Torch release'):
        release.validate_native_owner_release(
            tmp_path / 'missing.so', installed_version='2.14.0+xpu', declared_version='2.14.0+xpu')


def test_manifest_release_must_match_build_environment(tmp_path):
    with pytest.raises(release.UnsupportedRelease, match='provider declares Torch release'):
        release.validate_native_owner_release(
            tmp_path / 'not-loaded.so', installed_version='2.14.0+xpu', declared_version='2.13.0+xpu')


@pytest.mark.parametrize('built,installed,declared,accepted', [
    ('2.14.0+xpu', '2.14.0+xpu', '2.14.0+xpu', True),
    ('2.13.0+xpu', '2.14.0+xpu', '2.14.0+xpu', False),
    ('2.14.0+xpu', '2.14.0+xpu', '2.13.0+xpu', False),
    ('2.14.0+xpu', '2.13.0+xpu', '2.14.0+xpu', False),
])
def test_post_build_cli_checks_actual_export_and_torch(sidecar, tmp_path, built, installed, declared, accepted):
    path = sidecar(built)
    (tmp_path / 'torch.py').write_text(f'__version__ = {installed!r}\n')
    result = subprocess.run(
        [sys.executable, '-B', str(SCRIPT), '--torch-version', declared, '--native-owner', str(path)],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == (0 if accepted else 1), result.stderr
    if not accepted:
        assert 'Torch release' in result.stderr
