import hashlib
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


OMNI_ROOT = Path(__file__).resolve().parents[1]
ROOT_README = OMNI_ROOT.parent / "README.md"
DOCKERFILE = OMNI_ROOT / "docker" / "Dockerfile"
FULL_DOCKERFILE = OMNI_ROOT / "docker" / "Dockerfile.full"
DOCKERIGNORE = OMNI_ROOT / ".dockerignore"
BUILD_SCRIPT = OMNI_ROOT / "build.sh"
VALIDATOR = OMNI_ROOT / "tools" / "validate_comfyui_image.py"
UPGRADE_VALIDATOR = OMNI_ROOT / "tools" / "validate_comfyui_upgrade.py"
UPDATE_COMFYUI = OMNI_ROOT / "tools" / "update_comfyui.sh"
RUNTIME_ENTRYPOINT = OMNI_ROOT / "entrypoints" / "omix_torch_runtime.sh"
COMFYUI_ENTRYPOINT = OMNI_ROOT / "entrypoints" / "start_comfyui.sh"
HOST_MODELS_CONFIG = OMNI_ROOT / "configs" / "comfyui_host_models.yaml"
ONEDNN_PATCH = (
    OMNI_ROOT
    / "patches"
    / "onednn-v3.11.2-enable-bf16-int4-dequantization.patch"
)
PUBLIC_BMG_DOCUMENTATION = (
    OMNI_ROOT / "README.md",
    OMNI_ROOT / "ComfyUI-OmniXPU" / "README.md",
    OMNI_ROOT / "docs" / "COMFYUI.md",
    OMNI_ROOT / "docs" / "IMAGE_BUILD.md",
)
COMFYUI_STARTUP_DOCUMENTATION = (
    OMNI_ROOT / "README.md",
    OMNI_ROOT / "docs" / "COMFYUI.md",
)
OMNI_DOCKER_DOCUMENTATION = tuple(
    path
    for path in OMNI_ROOT.rglob("*.md")
    if "docker run" in path.read_text(encoding="utf-8")
)
CACHE_DIT_COMMIT = "1d92bbd86ec59aa6223fe2368849b7413a1acb93"
DEMO_ASSETS = {
    "demo_qwen_image.gif",
    "demo_wan2.2_14b_i2v_multi_xpu.gif",
}

STACK_PINS = {
    "BASE_IMAGE": (
        "BASE_IMAGE",
        "intel/omix:0.3.0-devel-ubuntu24.04@sha256:"
        "53e2c4503beeea4aff906dea180933be672449bcf04eb38df3d89622a1cd0967",
    ),
    "TORCH_VERSION": ("TORCH_VERSION", "2.13.0+xpu"),
    "TORCHVISION_VERSION": ("TORCHVISION_VERSION", "0.28.0+xpu"),
    "TORCHAUDIO_VERSION": ("TORCHAUDIO_VERSION", "2.11.0+xpu"),
    "ONEDNN_VERSION": ("ONEDNN_VERSION", "2026.0.0"),
    "ONEDNN_SOURCE_REPOSITORY": (
        "ONEDNN_SOURCE_REPOSITORY",
        "https://github.com/uxlfoundation/oneDNN.git",
    ),
    "ONEDNN_SOURCE_COMMIT": (
        "ONEDNN_SOURCE_COMMIT",
        "03c022d3ffdcee958cfacbe720048e725fdf644c",
    ),
    "ONEDNN_PATCH_SHA256": (
        "ONEDNN_PATCH_SHA256",
        "0a7afff4134f115b4bc53f46301ca3d62b1c11dc02e32c64635c469769fcdaeb",
    ),
}

COMPONENT_PINS = {
    "COMFYUI_REPOSITORY": (
        "COMFYUI_REPOSITORY",
        "https://github.com/Comfy-Org/ComfyUI.git",
    ),
    "COMFYUI_COMMIT": (
        "COMFYUI_COMMIT",
        "7a131a3afadc8200120f67f9236311a2c48b7445",
    ),
    "COMFYUI_VERSION": ("COMFYUI_VERSION", "0.33.4"),
    "COMFYUI_FRONTEND_VERSION": (
        "COMFYUI_FRONTEND_VERSION",
        "1.49.6",
    ),
    "COMFYUI_WORKFLOW_TEMPLATES_VERSION": (
        "COMFYUI_WORKFLOW_TEMPLATES_VERSION",
        "0.11.46",
    ),
    "COMFYUI_MANAGER_VERSION": ("COMFYUI_MANAGER_VERSION", "4.2.2"),
    "COMFY_KITCHEN_REPOSITORY": (
        "KITCHEN_REPOSITORY",
        "https://github.com/xiangyuT/comfy-kitchen-xpu.git",
    ),
    "COMFY_KITCHEN_COMMIT": (
        "KITCHEN_COMMIT",
        "9eccb7fa42edf14bc4a4c41aafd645ff1f1dcb75",
    ),
    "COMFY_KITCHEN_VERSION": ("KITCHEN_VERSION", "0.2.31"),
    "COMFY_AIMDO_REPOSITORY": (
        "AIMDO_REPOSITORY",
        "https://github.com/xiangyuT/comfy-aimdo-xpu.git",
    ),
    "COMFY_AIMDO_COMMIT": (
        "AIMDO_COMMIT",
        "063d66e5345fea58d1a4e8aa6f160ccc0c593f16",
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
        "cc0f6236b6c329178ad4ef58452a874e774c7b8e",
    ),
    "COMFY_NUNCHAKU_VERSION": ("NUNCHAKU_VERSION", "1.2.1+xpu.3"),
    "COMFY_SOL_ATTN_REPOSITORY": (
        "SOL_ATTN_REPOSITORY",
        "https://github.com/xiangyuT/ComfyUI-SolAttn_xpu.git",
    ),
    "COMFY_SOL_ATTN_COMMIT": (
        "SOL_ATTN_COMMIT",
        "5f1c4aac3ca32a00b0b4c15ddbb7cb53fa43344d",
    ),
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
    def test_omni_docker_docs_use_portable_least_privilege_gpu_access(self):
        self.assertTrue(OMNI_DOCKER_DOCUMENTATION)
        hardcoded_drm_node = re.compile(
            r"--device(?:=|\s+)/dev/dri/(?:card|renderD)\d+"
        )
        for path in OMNI_DOCKER_DOCUMENTATION:
            with self.subTest(path=path):
                document = path.read_text(encoding="utf-8")
                self.assertNotIn("--privileged", document)
                self.assertIsNone(hardcoded_drm_node.search(document))

    def test_comfyui_docs_default_to_direct_startup_and_scope_dynamic_vram(self):
        for path in COMFYUI_STARTUP_DOCUMENTATION:
            with self.subTest(path=path):
                document = path.read_text(encoding="utf-8")
                direct_start = document.index("python main.py")
                memory_preset = document.index(
                    "/llm/entrypoints/start_comfyui.sh"
                )

                self.assertLess(direct_start, memory_preset)
                self.assertIn("known or observed XPU", document)
                self.assertIn("out-of-memory risk", document)
                self.assertIn("reduce performance", document)

    def test_comfyui_host_models_stay_outside_the_upgradable_checkout(self):
        config = HOST_MODELS_CONFIG.read_text(encoding="utf-8")
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        entrypoint = COMFYUI_ENTRYPOINT.read_text(encoding="utf-8")

        self.assertIn("base_path: /models/host", config)
        for model_type in (
            "checkpoints",
            "diffusion_models",
            "text_encoders",
            "vae",
            "loras",
            "controlnet",
        ):
            with self.subTest(model_type=model_type):
                self.assertRegex(config, rf"(?m)^  {model_type}:")

        self.assertIn(
            "COPY ./configs/comfyui_host_models.yaml "
            "/llm/configs/comfyui_host_models.yaml",
            dockerfile,
        )
        self.assertIn(
            "RUN mkdir -p /models/host /data/input /data/output /data/user",
            dockerfile,
        )
        self.assertIn(
            "--extra-model-paths-config "
            "/llm/configs/comfyui_host_models.yaml",
            entrypoint,
        )
        for option, directory in (
            ("--input-directory", "/data/input"),
            ("--output-directory", "/data/output"),
            ("--user-directory", "/data/user"),
        ):
            with self.subTest(option=option):
                self.assertIn(f"{option} {directory}", entrypoint)
        for path in COMFYUI_STARTUP_DOCUMENTATION:
            with self.subTest(path=path):
                document = path.read_text(encoding="utf-8")
                self.assertIn("/models/host:ro", document)
                for runtime_directory in ("models", "input", "output", "user"):
                    self.assertNotIn(
                        f":/llm/ComfyUI/{runtime_directory}", document
                    )
                for data_directory in ("input", "output", "user"):
                    self.assertIn(f"/data/{data_directory}", document)
                self.assertIn(
                    "/llm/configs/comfyui_host_models.yaml",
                    document,
                )

    def test_public_image_documentation_focuses_on_bmg_support(self):
        for path in PUBLIC_BMG_DOCUMENTATION:
            with self.subTest(path=path):
                document = path.read_text(encoding="utf-8")
                self.assertNotIn("Intel publishes", document)
                self.assertNotIn("PTL-H", document)
                self.assertNotIn("ptl-h", document)

        adapter_readme = (
            OMNI_ROOT / "ComfyUI-OmniXPU" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("## Adapter behavior", adapter_readme)
        self.assertNotIn("## Contribution boundary", adapter_readme)

    def test_cache_dit_uses_the_pinned_minimax_h3_revision(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn(
            "git -C ComfyUI-CacheDiT fetch --depth 1 origin \\\n"
            f"        {CACHE_DIT_COMMIT}",
            dockerfile,
        )

    def test_sol_attn_custom_node_uses_packaged_xpu_backend(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        validator = load_validator()

        self.assertIn(
            '"${COMFY_SOL_ATTN_REPOSITORY}" ComfyUI-SolAttn',
            dockerfile,
        )
        self.assertIn(
            'git -C ComfyUI-SolAttn checkout --detach FETCH_HEAD',
            dockerfile,
        )
        self.assertNotIn("ComfyUI-SolAttn/requirements.txt", dockerfile)
        self.assertIn("SOL_ATTN_XPU_EXPERIMENTAL=1", dockerfile)
        self.assertEqual(
            validator.PINNED_CHECKOUTS["Sol-Attn custom node"],
            (
                Path("/llm/ComfyUI/custom_nodes/ComfyUI-SolAttn"),
                "OMNI_COMFY_SOL_ATTN_REVISION",
            ),
        )
        self.assertEqual(
            validator.SOL_ATTN_XPU_ADAPTER,
            Path("/llm/ComfyUI/custom_nodes/ComfyUI-SolAttn/_xpu_fwd.py"),
        )

    def test_root_readme_omni_links_resolve_to_current_docs_and_assets(self):
        readme = ROOT_README.read_text(encoding="utf-8")

        for name in DEMO_ASSETS:
            with self.subTest(asset=name):
                path = OMNI_ROOT / "assets" / name
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertIn(f"./omni/assets/{name}", readme)
        self.assertIn(
            "omni/README.md#getting-started-with-the-omni-docker-image",
            readme,
        )
        self.assertIn("omni/docs/COMFYUI.md", readme)
        self.assertIn(
            "https://github.com/intel/llm-scaler/blob/"
            "omni-0.1.0-b8/omni/README.md#xinference",
            readme,
        )
        self.assertNotIn("omni/README.md/#comfyui", readme)
        self.assertNotIn("omni/README.md/#xinference", readme)

    def test_build_context_keeps_repository_workflow_and_input_directories(self):
        ignored = {
            line.strip()
            for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertNotIn("workflows/", ignored)
        self.assertNotIn("example_inputs/", ignored)

    def test_build_proxy_default_has_no_organization_domains(self):
        build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
        full_dockerfile = FULL_DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn(
            'NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"',
            build_script,
        )
        self.assertNotIn("intel.com", build_script)
        self.assertEqual(
            re.findall(
                r"^ARG no_proxy=(.+)$",
                full_dockerfile,
                flags=re.MULTILINE,
            ),
            ["localhost,127.0.0.1,::1"] * 2,
        )

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

    def test_base_and_python_stack_defaults_are_explicit_build_inputs(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

        for docker_argument, (shell_variable, expected) in STACK_PINS.items():
            with self.subTest(argument=docker_argument):
                docker_match = re.search(
                    rf"^ARG {re.escape(docker_argument)}=(.+)$",
                    dockerfile,
                    flags=re.MULTILINE,
                )
                build_match = re.search(
                    (
                        rf"^{re.escape(shell_variable)}="
                        rf'"\$\{{OMNI_{re.escape(docker_argument)}:-([^}}]+)\}}"$'
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

        self.assertIn(
            "'^(torch|torchvision|torchaudio)"
            "([[:space:]<>=!~].*)?$|^(",
            dockerfile,
        )
        self.assertIn(
            'ENTRYPOINT ["/llm/entrypoints/omix_torch_runtime.sh"]',
            dockerfile,
        )
        runtime_entrypoint = RUNTIME_ENTRYPOINT.read_text(encoding="utf-8")
        self.assertIn(
            'source /opt/intel/oneapi/setvars.sh --force',
            runtime_entrypoint,
        )
        self.assertIn(
            '"${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib',
            runtime_entrypoint,
        )
        self.assertIn(
            'export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:'
            '${torch_library_directories[0]}:${LD_LIBRARY_PATH:-}"',
            runtime_entrypoint,
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
                "Sol-Attn custom node",
            },
        )
        self.assertIn(
            "dequantize_gguf",
            validator.REQUIRED_KITCHEN_CAPABILITIES,
        )

    def test_onednn_bf16_int4_patch_is_pinned_and_validated(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        validator = load_validator()
        patch_sha256 = hashlib.sha256(ONEDNN_PATCH.read_bytes()).hexdigest()

        self.assertEqual(
            patch_sha256,
            STACK_PINS["ONEDNN_PATCH_SHA256"][1],
        )
        self.assertIn(
            "git -C /tmp/onednn-source apply --check",
            dockerfile,
        )
        self.assertIn(
            "-DONEDNN_BUILD_GRAPH=OFF",
            dockerfile,
        )
        self.assertIn(
            "/llm/manifests/onednn-runtime.env",
            dockerfile,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            patch_path = temporary_root / "patch"
            runtime_library = temporary_root / "libdnnl.so.3.11"
            runtime_link = temporary_root / "libdnnl.so.3"
            manifest_path = temporary_root / "onednn-runtime.env"
            patch_path.write_bytes(ONEDNN_PATCH.read_bytes())
            runtime_library.write_bytes(b"patched oneDNN runtime")
            runtime_link.symlink_to(runtime_library.name)
            library_sha256 = hashlib.sha256(runtime_library.read_bytes()).hexdigest()
            manifest_path.write_text(
                "\n".join(
                    (
                        "schema_version=1",
                        "package_version=2026.0.0",
                        "source_repository=https://github.com/uxlfoundation/oneDNN.git",
                        "source_revision=03c022d3ffdcee958cfacbe720048e725fdf644c",
                        f"patch_sha256={patch_sha256}",
                        f"library_path={runtime_library}",
                        f"library_sha256={library_sha256}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            provenance = validator.require_onednn_runtime_provenance(
                expected_package_version="2026.0.0",
                expected_source_repository=(
                    "https://github.com/uxlfoundation/oneDNN.git"
                ),
                expected_source_revision=(
                    "03c022d3ffdcee958cfacbe720048e725fdf644c"
                ),
                expected_patch_sha256=patch_sha256,
                manifest_path=manifest_path,
                patch_path=patch_path,
                runtime_library=runtime_library,
                runtime_link=runtime_link,
            )
            self.assertEqual(provenance["library_sha256"], library_sha256)

            runtime_library.write_bytes(b"tampered runtime")
            with self.assertRaisesRegex(RuntimeError, "oneDNN runtime SHA256"):
                validator.require_onednn_runtime_provenance(
                    expected_package_version="2026.0.0",
                    expected_source_repository=(
                        "https://github.com/uxlfoundation/oneDNN.git"
                    ),
                    expected_source_revision=(
                        "03c022d3ffdcee958cfacbe720048e725fdf644c"
                    ),
                    expected_patch_sha256=patch_sha256,
                    manifest_path=manifest_path,
                    patch_path=patch_path,
                    runtime_library=runtime_library,
                    runtime_link=runtime_link,
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
        self.assertIn(
            "test -f /opt/venv/include/unified-runtime/ur_api.h",
            dockerfile,
        )
        self.assertIn(
            "UR_INCLUDE_DIR=/opt/venv/include/unified-runtime",
            dockerfile,
        )
        self.assertIn(
            'UR_INCLUDE_DIR="/opt/venv/include/unified-runtime"',
            dockerfile,
        )
        self.assertIn("./scripts/build-linux-xpu.sh", dockerfile)
        self.assertIn(
            'SETUPTOOLS_SCM_PRETEND_VERSION="${COMFY_AIMDO_VERSION}"',
            dockerfile,
        )
        self.assertIn(
            "/wheels/aimdo-source/comfy_aimdo-${COMFY_AIMDO_VERSION}-*.whl",
            dockerfile,
        )
        self.assertIn(
            "python packaging/xpu_runtime_provider/build_wheel.py",
            dockerfile,
        )
        self.assertIn(
            "/wheels/providers/comfy_aimdo_xpu_runtime-"
            "${COMFY_AIMDO_VERSION}-*.whl",
            dockerfile,
        )
        self.assertEqual(
            validator.PINNED_CHECKOUTS["Comfy AIMDO"],
            (
                Path("/llm/comfy-aimdo-xpu"),
                "OMNI_COMFY_AIMDO_PROVIDER_REVISION",
            ),
        )
        self.assertEqual(
            validator.AIMDO_REQUIRED_XPU_TESTS,
            {
                "test_xpu_backend.py",
                "test_xpu_comfyui_opt_in.py",
            },
        )

    def test_comfyui_dependencies_are_pinned_and_validated(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        validator = load_validator()
        self.assertEqual(
            validator.PINNED_MINIMAX_H3_TEMPLATE_HASHES[
                "video_minimax_h3_t2v.json"
            ],
            "eb9cc8c78a4e83a2657c9926677b6353db0ed2947d91da736e9cdae194a3db4a",
        )

        self.assertIn(
            '"comfyui-manager==${COMFYUI_MANAGER_VERSION}"',
            dockerfile,
        )
        self.assertIn(
            "comfyui-workflow-templates|comfy-kitchen|comfy-aimdo",
            dockerfile,
        )
        self.assertIn(
            '"comfy-kitchen==${COMFY_KITCHEN_VERSION}"',
            dockerfile,
        )
        self.assertIn(
            '"comfy-aimdo==${COMFY_AIMDO_VERSION}"',
            dockerfile,
        )
        self.assertIn(
            "/llm/manifests/xpu-runtime-providers.sha256",
            dockerfile,
        )
        self.assertIn(
            "/llm/manifests/omni-runtime-constraints.txt",
            dockerfile,
        )
        self.assertIn(
            'PIP_CONSTRAINT="/llm/manifests/omni-runtime-constraints.txt"',
            dockerfile,
        )
        self.assertNotIn(
            "/wheels/comfy_kitchen-${COMFY_KITCHEN_VERSION}-*.whl",
            dockerfile,
        )
        self.assertNotIn(
            "/wheels/comfy_aimdo-${COMFY_AIMDO_VERSION}-*.whl",
            dockerfile,
        )
        self.assertNotIn(
            "https://github.com/ltdrdata/ComfyUI-Manager.git",
            dockerfile,
        )
        self.assertIn(
            "# Easy-Use v1.3.6 (release tag resolved to an immutable commit).",
            dockerfile,
        )
        self.assertIn(
            "b5e31ef12ad9d0b187b545c2707735cc7d581c52",
            dockerfile,
        )
        self.assertNotIn(
            "54d080bf6a4f52da287e984f305243c10db097f5",
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
        self.assertNotIn("libsycl-native-*.spv", dockerfile)
        self.assertEqual(
            validator.COMFYUI_DATABASE_DIRECTORY,
            Path("/llm/ComfyUI/user"),
        )

    def test_upgrade_path_preserves_provider_ownership_and_runtime_constraints(self):
        update_script = UPDATE_COMFYUI.read_text(encoding="utf-8")
        upgrade_validator = UPGRADE_VALIDATOR.read_text(encoding="utf-8")

        self.assertNotIn("git stash", update_script)
        self.assertNotIn("git pull origin master", update_script)
        self.assertIn('COMFYUI_UPGRADE_REF="${COMFYUI_UPGRADE_REF:-master}"', update_script)
        self.assertIn('git checkout --detach FETCH_HEAD', update_script)
        self.assertIn('python -m pip install --upgrade -r requirements.txt', update_script)
        self.assertIn("PIP_CONSTRAINT", update_script)
        for distribution in (
            "comfy-aimdo-xpu-runtime",
            "comfy-kitchen-xpu-runtime",
        ):
            with self.subTest(distribution=distribution):
                self.assertIn(distribution, upgrade_validator)
        self.assertIn("provider_snapshot()", upgrade_validator)
        self.assertIn("official package or ComfyUI upgrade changed", upgrade_validator)
        self.assertIn('"OMNIXPU_PROVIDER_BOOTSTRAP": mode', upgrade_validator)
        self.assertIn('activation_probe("auto")', upgrade_validator)
        self.assertIn('activation_probe("required")', upgrade_validator)

    def test_validator_adds_comfyui_root_before_manager_import(self):
        validator = load_validator()

        with mock.patch.object(sys, "path", ["sentinel"]):
            validator.add_comfyui_to_import_path()
            validator.add_comfyui_to_import_path()

            self.assertEqual(sys.path[0], "/llm/ComfyUI")
            self.assertEqual(sys.path.count("/llm/ComfyUI"), 1)

    def test_validator_completes_aimdo_device_lifecycle_before_allocation(self):
        validator = load_validator()
        control = mock.Mock()
        control.init_devices.return_value = True
        control.devctxs = [123]

        validator.require_aimdo_xpu_devices(control, (0,))

        control.init_devices.assert_called_once_with([0])

        control.init_devices.return_value = False
        with self.assertRaisesRegex(RuntimeError, "failed to initialize"):
            validator.require_aimdo_xpu_devices(control, (0,))

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
