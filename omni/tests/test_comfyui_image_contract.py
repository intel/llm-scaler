import importlib.util
import re
import sys
import unittest
from pathlib import Path
from unittest import mock


OMNI_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = OMNI_ROOT / "docker" / "Dockerfile"
BUILD_SCRIPT = OMNI_ROOT / "build.sh"
VALIDATOR = OMNI_ROOT / "tools" / "validate_comfyui_image.py"

COMPONENT_PINS = {
    "COMFYUI_REPOSITORY": (
        "COMFYUI_REPOSITORY",
        "https://github.com/Comfy-Org/ComfyUI.git",
    ),
    "COMFYUI_COMMIT": (
        "COMFYUI_COMMIT",
        "b1693ecba9f5b65f8c80ab36b195ab963ec92413",
    ),
    "COMFYUI_VERSION": ("COMFYUI_VERSION", "0.30.0"),
    "COMFYUI_FRONTEND_VERSION": (
        "COMFYUI_FRONTEND_VERSION",
        "1.47.12",
    ),
    "COMFYUI_WORKFLOW_TEMPLATES_VERSION": (
        "COMFYUI_WORKFLOW_TEMPLATES_VERSION",
        "0.11.28",
    ),
    "COMFYUI_MANAGER_VERSION": ("COMFYUI_MANAGER_VERSION", "4.2.2"),
    "COMFY_KITCHEN_REPOSITORY": (
        "KITCHEN_REPOSITORY",
        "https://github.com/xiangyuT/comfy-kitchen-xpu.git",
    ),
    "COMFY_KITCHEN_COMMIT": (
        "KITCHEN_COMMIT",
        "f7250fa44cb6f593969ba869be803e7d03c80ec8",
    ),
    "COMFY_KITCHEN_VERSION": ("KITCHEN_VERSION", "0.2.26"),
    "COMFY_AIMDO_REPOSITORY": (
        "AIMDO_REPOSITORY",
        "https://github.com/xiangyuT/comfy-aimdo-xpu.git",
    ),
    "COMFY_AIMDO_COMMIT": (
        "AIMDO_COMMIT",
        "6fda6e619e1647134d4ced4370e5fad488779d62",
    ),
    "COMFY_AIMDO_VERSION": ("AIMDO_VERSION", "0.4.13"),
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
                "ComfyUI",
                "Kitchen",
                "Comfy AIMDO",
                "GGUF custom node",
                "combined Nunchaku custom node/runtime",
            },
        )
        self.assertIn(
            "dequantize_gguf",
            validator.REQUIRED_KITCHEN_CAPABILITIES,
        )

    def test_aimdo_xpu_is_built_from_an_exact_remote_commit(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        validator = load_validator()

        self.assertIn("FROM python-base AS aimdo-wheel", dockerfile)
        self.assertIn(
            '"${COMFY_AIMDO_REPOSITORY}" comfy-aimdo-xpu',
            dockerfile,
        )
        self.assertIn(
            "git -C comfy-aimdo-xpu fetch --depth 1 origin",
            dockerfile,
        )
        self.assertIn("pip install setuptools-scm==10.2.1", dockerfile)
        self.assertIn("./scripts/build-linux-xpu.sh", dockerfile)
        self.assertIn(
            'SETUPTOOLS_SCM_PRETEND_VERSION="${COMFY_AIMDO_VERSION}"',
            dockerfile,
        )
        self.assertIn(
            "/wheels/comfy_aimdo-${COMFY_AIMDO_VERSION}-*.whl",
            dockerfile,
        )
        self.assertEqual(
            validator.PINNED_CHECKOUTS["Comfy AIMDO"],
            (
                Path("/llm/comfy-aimdo-xpu"),
                "OMNI_COMFY_AIMDO_REVISION",
            ),
        )

    def test_comfyui_v030_dependencies_are_pinned_and_validated(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        validator = load_validator()
        self.assertEqual(
            validator.PINNED_MINIMAX_H3_TEMPLATE_HASHES[
                "video_minimax_h3_t2v.json"
            ],
            "31ab33fdb053a7834cc866bd7aa08b887518fc656e4a796c89779c6b5e1786e6",
        )

        self.assertIn(
            '"comfyui-manager==${COMFYUI_MANAGER_VERSION}"',
            dockerfile,
        )
        self.assertNotIn(
            "https://github.com/ltdrdata/ComfyUI-Manager.git",
            dockerfile,
        )
        self.assertEqual(
            validator.COMFYUI_PACKAGE_ENVIRONMENT,
            {
                "comfyui-frontend-package": "OMNI_COMFYUI_FRONTEND_VERSION",
                "comfyui-workflow-templates": (
                    "OMNI_COMFYUI_WORKFLOW_TEMPLATES_VERSION"
                ),
                "comfyui-manager": "OMNI_COMFYUI_MANAGER_VERSION",
            },
        )
        self.assertEqual(len(validator.REQUIRED_MINIMAX_H3_TEMPLATES), 6)
        self.assertIn(
            "/llm/manifests/comfyui-python-freeze.txt",
            dockerfile,
        )
        self.assertIn("mkdir -p /llm/ComfyUI/user", dockerfile)
        self.assertEqual(
            validator.COMFYUI_DATABASE_DIRECTORY,
            Path("/llm/ComfyUI/user"),
        )

    def test_validator_adds_comfyui_root_before_manager_import(self):
        validator = load_validator()

        with mock.patch.object(sys, "path", ["sentinel"]):
            validator.add_comfyui_to_import_path()
            validator.add_comfyui_to_import_path()

            self.assertEqual(sys.path[0], "/llm/ComfyUI")
            self.assertEqual(sys.path.count("/llm/ComfyUI"), 1)

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
