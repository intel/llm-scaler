from typing import Any, Dict, Iterable, List, Optional
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os

LOG_PATH = "auto_test_log"
ANALYSIS_PATH = "analysis"

default_config = {
    "VERSION": "0.14.0-b8.1",
    "REPO": "intel/llm-scaler-vllm",
    "Port": 8000,
    "EnablePrefixCaching": True,
    "EnforceEager": False,
    "Path": {
        "ModelPath": "/home/intel/LLM/",
    },
    "Dataset": {
        "name": "random",
        "random-input-len": 1024,
        "random-output-len": 512
    },
    "XPU": "6",
    "Model": [
        {
            "name": "Qwen3-32B",
            "tp" : 1,
            "quantization": "fp8",
            "batch": "1,2,4",
            "extra_param": {},
        },
    ],
}

def current_date() -> str:
    return datetime.now().strftime("%Y_%m_%d")

@dataclass
class DatasetConfig:
    name: str
    path: str
    input_len: int
    output_len: int

    def __post_init__(self):
        if self.name == "random":
            if self.input_len <= 0 or self.output_len <= 0:
                raise ValueError("Token config error")
        
@dataclass
class PathConfig:
    ModelPath: List[str]
    TestPath: str
    LogPath: str
    AnalysisPath: str
    ModelPathMap: Dict[str,str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.ModelPath or len(self.ModelPath) == 0:
            raise ValueError("Path error")

        for i in range(len(self.ModelPath)):
            path = self.ModelPath[i]
            if not os.path.exists(path):
                raise ValueError("Path error")
            
            path = os.path.abspath(path)
            if not path.endswith(os.path.sep):
                path += os.path.sep
            self.ModelPath[i] = path
            self.ModelPathMap[path] = "/llm/models/%s/" % (path.split(os.path.sep)[-2])

        if not self.LogPath or self.LogPath == "":
            self.LogPath = LOG_PATH
        if not self.AnalysisPath or self.AnalysisPath == "":
            self.AnalysisPath = ANALYSIS_PATH

@dataclass            
class SpecConfig:
    method: str
    model: str
    num_speculative_tokens: int

@dataclass
class ModelSpec:
    name: str
    tag: str
    quantization: str
    revision: str = ""
    tp: int = 1
    batch: List[int] = field(default_factory=list)
    spec_config: SpecConfig = None
    extra_param: Dict[str, Any] = field(default_factory=dict)
    path: str = ""

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ModelSpec.name error")        
    
@dataclass
class ScriptConfig:
    VERSION: str
    REPO: str
    DATE: str = ""
    Port: int = 8000
    XPU: Optional[str] = None
    EnablePrefixCaching: bool = True
    EnforceEager: bool = False
    Path: PathConfig = field(default_factory=lambda: PathConfig(ModelPath=[], TestPath="", LogPath="",AnalysisPath=""))
    Dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="random", path="", input_len=1024, output_len=512))
    Model: List[ModelSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.REPO:
            raise ValueError("REPO error")
        if not self.VERSION:
            raise ValueError("VERSION error")
        if not isinstance(self.Port, int) or self.Port <= 0:
            raise ValueError("Port error")

        if not self.DATE:
            self.DATE = current_date()

        exist_models = []
        for model in self.Model:
            if model.spec_config:
                model_path, ok = self.check_and_get_model_path(model.spec_config.model)
                if not ok:
                    raise ValueError("model %s spec config not found" % model.name)
                model.spec_config.model = model_path
            if model.revision:
                model_path, ok = self.check_and_get_model_path_with_revision(model.name, model.revision)
            else:
                model_path, ok = self.check_and_get_model_path(model.name)

            if not ok:
                raise ValueError("model %s  not found" % model.name)
            model.path = model_path
            exist_models.append(model)

        self.Model = exist_models

        if self.Dataset.path:
            model_path, ok = self.check_and_get_model_path(self.Dataset.path)
            if not ok:
                raise ValueError("Dataset path error")
            self.Dataset.path = model_path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    def check_and_get_model_path(self, model_name: str):
        model_name = model_name.strip()
        for path in self.Path.ModelPath:
            for candidate in self._build_model_candidates(model_name):
                resolved_candidate = self._resolve_candidate(path, candidate)
                if resolved_candidate:
                    model_path = self.Path.ModelPathMap[path] + resolved_candidate
                    return model_path, True
        print("model %s not found" % model_name)
        return "", False

    def check_and_get_model_path_with_revision(self, model_name: str, revision: str):
        model_name = model_name.strip()
        revision = revision.strip()
        for path in self.Path.ModelPath:
            for candidate in self._build_model_candidates(model_name):
                resolved_candidate = self._resolve_candidate(path, candidate, revision)
                if resolved_candidate:
                    model_path = self.Path.ModelPathMap[path] + resolved_candidate
                    return model_path, True
        print("model %s (revision %s) not found" % (model_name, revision))
        return "", False

    @staticmethod
    def _resolve_candidate(base_path: str, candidate: str, revision: str = "") -> Optional[str]:
        candidate_path = os.path.join(base_path, candidate)
        if not os.path.exists(candidate_path):
            return None

        snapshot_relative = ScriptConfig._resolve_hf_snapshot_candidate(candidate_path, candidate, revision)
        if snapshot_relative:
            return snapshot_relative
        return candidate

    @staticmethod
    def _resolve_hf_snapshot_candidate(candidate_path: str, candidate: str, revision: str = "") -> Optional[str]:
        snapshots_dir = os.path.join(candidate_path, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return None

        if revision:
            revision_snapshot = ScriptConfig._resolve_snapshot_by_revision(candidate_path, snapshots_dir, revision)
            if revision_snapshot:
                return os.path.join(candidate, "snapshots", revision_snapshot)

        for snapshot in ScriptConfig._resolve_snapshot_candidates_from_refs(candidate_path, snapshots_dir):
            if ScriptConfig._snapshot_has_required_files(os.path.join(snapshots_dir, snapshot)):
                return os.path.join(candidate, "snapshots", snapshot)

        latest_snapshot = ScriptConfig._resolve_latest_snapshot(snapshots_dir)
        if latest_snapshot and ScriptConfig._snapshot_has_required_files(os.path.join(snapshots_dir, latest_snapshot)):
            return os.path.join(candidate, "snapshots", latest_snapshot)
        return None

    @staticmethod
    def _resolve_snapshot_by_revision(candidate_path: str, snapshots_dir: str, revision: str) -> Optional[str]:
        refs_revision_path = os.path.join(candidate_path, "refs", revision)
        if os.path.isfile(refs_revision_path):
            with open(refs_revision_path, "r", encoding="utf-8") as f:
                snapshot_hash = f.read().strip()
            if snapshot_hash and os.path.isdir(os.path.join(snapshots_dir, snapshot_hash)):
                if ScriptConfig._snapshot_has_required_files(os.path.join(snapshots_dir, snapshot_hash)):
                    return snapshot_hash

        if os.path.isdir(os.path.join(snapshots_dir, revision)):
            if ScriptConfig._snapshot_has_required_files(os.path.join(snapshots_dir, revision)):
                return revision
        return None

    @staticmethod
    def _resolve_snapshot_candidates_from_refs(candidate_path: str, snapshots_dir: str) -> List[str]:
        refs_dir = os.path.join(candidate_path, "refs")
        if not os.path.isdir(refs_dir):
            return []

        preferred_refs = ["main", "master"]
        other_refs = sorted(
            entry for entry in os.listdir(refs_dir)
            if entry not in preferred_refs and os.path.isfile(os.path.join(refs_dir, entry))
        )
        ordered_refs = preferred_refs + other_refs

        snapshots = []
        seen = set()
        for ref_name in ordered_refs:
            ref_path = os.path.join(refs_dir, ref_name)
            if not os.path.isfile(ref_path):
                continue
            with open(ref_path, "r", encoding="utf-8") as f:
                snapshot_hash = f.read().strip()
            if not snapshot_hash or snapshot_hash in seen:
                continue
            if os.path.isdir(os.path.join(snapshots_dir, snapshot_hash)):
                seen.add(snapshot_hash)
                snapshots.append(snapshot_hash)
        return snapshots

    @staticmethod
    def _snapshot_has_required_files(snapshot_path: str) -> bool:
        if not os.path.isdir(snapshot_path):
            return False

        entries = set(os.listdir(snapshot_path))
        has_config = "config.json" in entries
        has_tokenizer = any(
            name in entries for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        )
        has_weights = any(
            ScriptConfig._matches_any_pattern(
                name,
                ("*.safetensors", "*.bin", "*.gguf", "model.safetensors.index.json"),
            )
            for name in entries
        )
        return has_config and has_tokenizer and has_weights

    @staticmethod
    def _matches_any_pattern(value: str, patterns: Iterable[str]) -> bool:
        for pattern in patterns:
            if pattern == value:
                return True
            if pattern.startswith("*") and value.endswith(pattern[1:]):
                return True
        return False

    @staticmethod
    def _resolve_latest_snapshot(snapshots_dir: str) -> Optional[str]:
        snapshot_dirs = []
        for entry in os.listdir(snapshots_dir):
            entry_path = os.path.join(snapshots_dir, entry)
            if os.path.isdir(entry_path) and ScriptConfig._snapshot_has_required_files(entry_path):
                snapshot_dirs.append(entry_path)

        if not snapshot_dirs:
            return None

        latest_snapshot_path = max(snapshot_dirs, key=os.path.getmtime)
        return os.path.basename(latest_snapshot_path)

    @staticmethod
    def _build_model_candidates(model_name: str) -> List[str]:
        """
        Build possible local directory names for a model.
        This allows configs to use either local folder names
        (e.g. `Qwen3.5-9B-Instruct`) or Hub repo ids
        (e.g. `Qwen/Qwen3.5-9B-Instruct`).
        """
        candidates = [model_name]

        if "/" in model_name:
            _, suffix = model_name.split("/", 1)
            candidates.append(suffix)
            candidates.append(model_name.replace("/", "--"))
            candidates.append(f"models--{model_name.replace('/', '--')}")

        # Keep ordering stable while removing duplicates.
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _parse_model_paths(raw_model_paths: Any) -> List[str]:
        if isinstance(raw_model_paths, str):
            return [p.strip() for p in raw_model_paths.split(';') if p.strip()]
        if isinstance(raw_model_paths, list):
            parsed = []
            for idx, p in enumerate(raw_model_paths):
                if not isinstance(p, str):
                    raise TypeError(f"Path.ModelPath[{idx}] must be a string")
                normalized = p.strip()
                if normalized:
                    parsed.append(normalized)
            return parsed
        raise TypeError(
            f"Path.ModelPath must be a ';' separated string or list[str], got {type(raw_model_paths).__name__}"
        )

    @staticmethod
    def _parse_batch(raw_batch: Any, model_name: str) -> List[int]:
        if isinstance(raw_batch, str):
            tokens = [x.strip() for x in raw_batch.split(',') if x.strip()]
        elif isinstance(raw_batch, list):
            tokens = raw_batch
        else:
            raise TypeError(
                f"Model '{model_name}' batch must be CSV string or list[int], got {type(raw_batch).__name__}"
            )

        batch = []
        for idx, value in enumerate(tokens):
            try:
                parsed = int(value)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Model '{model_name}' batch[{idx}]='{value}' is not an integer"
                ) from err
            batch.append(parsed)
        return batch

    @staticmethod
    def build_sub_obj(data: Dict[str, Any]) :
        path = data.get("Path", {})
        dataset = data.get("Dataset", {})
        models = data.get("Model", [])

        path_obj = PathConfig(
            ModelPath=ScriptConfig._parse_model_paths(path.get("ModelPath", "")),
            TestPath=path.get("TestPath", ""),
            LogPath=path.get("LogPath", ""),
            AnalysisPath=path.get("AnalysisPath",""),
        )
        dataset_obj = DatasetConfig(
            name=dataset.get("name", "random"),
            path=dataset.get("path", ""),
            input_len=int(dataset.get("random-input-len", 1024)),
            output_len=int(dataset.get("random-output-len", 512)),
        )

        model_objs = []
        for model in models:
            spec_obj = None
            spec = model.get("speculative_config")
            if spec:
                spec_obj = SpecConfig(
                    method=spec.get("method"),
                    model=spec.get("model"),
                    num_speculative_tokens=spec.get("num_speculative_tokens"),
                )

            tp = int(model.get("tp", 1))
            if tp < 1:
                raise ValueError("ModelSpec.tp must be >= 1")

            model_obj = ModelSpec(
                name=model.get("name"),
                tag=model.get("tag",""),
                tp=tp,
                quantization=model.get("quantization"),
                revision=model.get("revision", ""),
                batch=ScriptConfig._parse_batch(model.get("batch", ""), model.get("name", "")),
                spec_config=spec_obj,
                extra_param=model.get("extra_param"),
            )
            model_objs.append(model_obj)

        return path_obj, dataset_obj, model_objs
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScriptConfig":
        path_obj, dataset_obj, model_objs = ScriptConfig.build_sub_obj(data)

        return cls(
            DATE=str(data.get("DATE", current_date())),
            VERSION=str(data.get("VERSION", "last")),
            REPO=str(data.get("REPO", "")),
            Port=int(data.get("Port", 8000)),
            XPU=(str(data["XPU"]) if "XPU" in data and data["XPU"] is not None else None),
            EnablePrefixCaching=bool(data.get("EnablePrefixCaching", True)),
            EnforceEager=bool(data.get("EnforceEager", False)),
            Path=path_obj,
            Dataset=dataset_obj,
            Model=model_objs,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ScriptConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        path_obj, dataset_obj, model_objs = ScriptConfig.build_sub_obj(data)
        data["Path"] = path_obj
        data["Dataset"] = dataset_obj
        data["Model"] = model_objs
        return cls(**data)
