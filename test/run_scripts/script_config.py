from typing import Any, Dict, List, Optional
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os

LOG_PATH = "auto_test_log"
ANALYSIS_PATH = "analysis"

default_config = {
    "VERSION": "0.10.2-b6",
    "REPO": "intel/llm-scaler-vllm",
    "Port": 8000,
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
    tp: int = 1
    batch: List[int] = field(default_factory=list)
    spec_config: SpecConfig = None
    extra_param: Dict[str, str] = field(default_factory=dict)
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
    Path: PathConfig = field(default_factory=lambda: PathConfig(ModelPath="", TestPath="", LogPath="",AnalysisPath=""))
    Dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="random", input_len=1024, output_len=512))
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
        for path in self.Path.ModelPath:
            if os.path.exists(path+model_name):
                model_path = self.Path.ModelPathMap[path] + model_name
                return model_path, True
        print("model %s not found" % model_name)
        return "", False

    def build_sub_obj(data: Dict[str, Any]) :
        path = data.get("Path", {})
        dataset = data.get("Dataset", {})
        models = data.get("Model", [])

        path_obj = PathConfig(
            ModelPath=path.get("ModelPath", "").split(';'),
            TestPath=path.get("TestPath", ""),
            LogPath=path.get("LogPath", ""),
            AnalysisPath=path.get("AnalysisPath",""),
        )
        dataset_obj = DatasetConfig(
            name=dataset.get("name", "random"),
            path=dataset.get("path"),
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

            model_obj = ModelSpec(
                name=model.get("name"),
                tag=model.get("tag",""),
                tp=int(model.get("tp")),
                quantization=model.get("quantization"),
                batch=[int(x) for x in model.get("batch").split(',')],
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