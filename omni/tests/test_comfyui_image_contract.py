import importlib.util
import re
import unittest
from pathlib import Path


OMNI_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = OMNI_ROOT / "docker" / "Dockerfile"
BUILD_SCRIPT = OMNI_ROOT / "build.sh"
VALIDATOR = OMNI_ROOT / "tools" / "validate_comfyui_image.py"

COMPONENT_PINS = {
    "COMFY_KITCHEN_REPOSITORY": (
        "KITCHEN_REPOSITORY",
        "https://github.com/xiangyuT/comfy-kitchen-xpu.git",
    ),
    "COMFY_KITCHEN_COMMIT": (
        "KITCHEN_COMMIT",
        "75eb682b18a8c9970908927ab1b98c9b125230ab",
    ),
    "COMFY_KITCHEN_VERSION": ("KITCHEN_VERSION", "0.2.18"),
    "COMFY_GGUF_REPOSITORY": (
        "GGUF_REPOSITORY",
        "https://github.com/analytics-zoo/ComfyUI-GGUF-XPU.git",
    ),
    "COMFY_GGUF_COMMIT": (
        "GGUF_COMMIT",
        "39671fe73117ba97de7011e7e06e32599dcda06d",
    ),
    "COMFY_NUNCHAKU_REPOSITORY": (
        "NUNCHAKU_REPOSITORY",
        "https://github.com/xiangyuT/ComfyUI-nunchaku-XPU.git",
    ),
    "COMFY_NUNCHAKU_COMMIT": (
        "NUNCHAKU_COMMIT",
        "5cf4fa9886f45abff102d1dd91af5247b4950148",
    ),
    "COMFY_NUNCHAKU_VERSION": ("NUNCHAKU_VERSION", "1.2.1+xpu.3"),
}


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_comfyui_image_under_test",
        VALIDATOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComfyUIImageContractTest(unittest.TestCase):
    def test_component_defaults_match_dockerfile_and_build_entrypoint(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

        for docker_argument, (shell_variable, expected) in COMPONENT_PINS.items():
            with self.subTest(argument=docker_argument):
                docker_match = re.search(
                    rf"^ARG {re.escape(docker_argument)}=(.+)$",
                    dockerfile,
                    flags=re.MULTILINE,
                )
                build_match = re.search(
                    (
                        rf"^{re.escape(shell_variable)}="
                        rf'"\$\{{{re.escape(docker_argument)}:-([^}}]+)\}}"$'
                    ),
                    build_script,
                    flags=re.MULTILINE,
                )
                self.assertIsNotNone(docker_match)
                self.assertIsNotNone(build_match)
                self.assertEqual(docker_match.group(1), expected)
                self.assertEqual(build_match.group(1), expected)
                self.assertIn(
                    f'--build-arg "{docker_argument}=${{{shell_variable}}}"',
                    build_script,
                )

    def test_quantized_integrations_install_and_validate_dependencies(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        validator = load_validator()

        self.assertIn(
            "pip install -r ComfyUI-GGUF-XPU/requirements.txt",
            dockerfile,
        )
        self.assertIn(
            "pip install -r ComfyUI-nunchaku-XPU/requirements.txt",
            dockerfile,
        )
        self.assertIn(
            "pip install --no-deps --no-build-isolation "
            "./ComfyUI-nunchaku-XPU",
            dockerfile,
        )
        self.assertEqual(
            validator.GGUF_DEPENDENCIES,
            {
                "gguf": "gguf",
                "sentencepiece": "sentencepiece",
                "protobuf": "google.protobuf",
            },
        )
        self.assertEqual(
            set(validator.PINNED_CHECKOUTS),
            {
                "Kitchen",
                "GGUF custom node",
                "combined Nunchaku custom node/runtime",
            },
        )
        self.assertIn(
            "dequantize_gguf",
            validator.REQUIRED_KITCHEN_CAPABILITIES,
        )

    def test_validator_rejects_noncanonical_component_revisions(self):
        validator = load_validator()

        for revision in ("", "39671fe", "g" * 40, "0" * 39, "A" * 40):
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "full 40-character Git commit",
                ):
                    validator.require_full_revision(
                        "component revision",
                        revision,
                    )


if __name__ == "__main__":
    unittest.main()
