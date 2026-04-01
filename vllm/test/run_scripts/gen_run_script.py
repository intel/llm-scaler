import argparse
import json
import shlex
import re

from script_config import ScriptConfig, default_config, ModelSpec

SAFE_EXTRA_PARAM_KEY = re.compile(r"^--[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def validate_extra_param_keys(config: ScriptConfig):
    for model in config.Model:
        if not model.extra_param:
            continue
        for key in model.extra_param.keys():
            if not SAFE_EXTRA_PARAM_KEY.match(key):
                raise ValueError(
                    f"Invalid extra_param key '{key}' for model '{model.name}'. "
                    "Expected pattern: ^--[a-zA-Z0-9][a-zA-Z0-9_-]*$"
                )


def q(value):
    return shlex.quote(str(value))


def render_extra_param(flag, value):
    if value is None:
        return q(flag)
    return '%s=%s' % (q(flag), q(value))


def create_container(container,config:ScriptConfig):
    outstr = '# create container version %s\n' % config.VERSION
    outstr += 'sudo docker run -td '
    outstr += '--privileged '
    outstr += '--net=host '
    outstr += '--device=/dev/dri '
    outstr += '--name=%s ' % q(container)
    for path, path_map in config.Path.ModelPathMap.items():
        outstr += '-v %s:%s ' % (q(path), q(path_map))
    if config.Path.TestPath and config.Path.TestPath != "":
        outstr += '-v %s:%s ' % (q(config.Path.TestPath), q('/llm/test/'))
    outstr += '-e no_proxy=localhost,127.0.0.1 '
    outstr += '-e http_proxy=$http_proxy '
    outstr += '-e https_proxy=$https_proxy '
    outstr += '--shm-size="32g" '
    outstr += '--entrypoint /bin/bash '
    outstr += '%s:%s ' % (config.REPO, config.VERSION)
    print(outstr + "\n")

def start_container(container):
    outstr = 'sudo docker start %s ' % q(container)
    print(outstr + "\n")

def stop_container(container):
    outstr = 'sudo docker stop %s ' % q(container)
    print(outstr + "\n")

def rm_container(container): 
    outstr = 'sudo docker rm %s ' % q(container)
    print(outstr + "\n")

def run_model(container, model:ModelSpec, config:ScriptConfig):
    outstr = '# start model %s\n' % model.name
    if model.tag and model.tag != "":
        outstr += 'EXPDIR=%s && ' % q(f"{config.Path.LogPath}/{container}/{model.name}+{model.tag}")
    else:
        outstr += 'EXPDIR=%s && ' % q(f"{config.Path.LogPath}/{container}/{model.name}")
    outstr += 'mkdir -p ${EXPDIR} && touch "${EXPDIR}/model.log" && '
    outstr += 'docker exec -i %s bash -lc "' % q(container)
    if config.XPU:
        outstr += 'ZE_AFFINITY_MASK=%s ' % q(config.XPU)
    outstr += 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 '
    outstr += 'VLLM_WORKER_MULTIPROC_METHOD=spawn '
    outstr += 'python3 -m vllm.entrypoints.openai.api_server '
    outstr += '--model %s ' % q(model.path)
    outstr += '--served-model-name %s ' % q(model.name)
    outstr += '--dtype=float16 '
    if config.EnforceEager:
        outstr += '--enforce-eager '
    outstr += '--port %d ' % config.Port
    outstr += '--host 0.0.0.0 '
    outstr += '--trust-remote-code '
    outstr += '--gpu-memory-util=0.9 '
    outstr += '--max-num-batched-tokens=2048 '
    outstr += '--disable-log-requests '
    outstr += '--block-size 64 '
    if model.quantization:
        outstr += '--quantization %s ' % q(model.quantization)
        outstr += '--allow-deprecated-quantization '
    outstr += '-tp=%d ' % model.tp
    if model.spec_config:
        spec_dict = {
            "method": model.spec_config.method,
            "model": model.spec_config.model,
            "num_speculative_tokens": model.spec_config.num_speculative_tokens,
        }
        outstr += '--speculative_config=%s ' % shlex.quote(json.dumps(spec_dict))
    if model.extra_param:
        for flag, value in model.extra_param.items():
            outstr += '%s ' % render_extra_param(flag, value)
    outstr += '" 2>&1 | tee -a "${EXPDIR}/model.log" &'
    print(outstr + "\n")

def stop_model(container):
    outstr = '# stop model server\n'
    outstr += 'docker exec -i %s bash -lc "pkill -f \'vllm.entrypoints.openai.api_server\' || true"' % q(container)
    print(outstr + "\n")

def check_ready(config:ScriptConfig):
    max_wait_seconds = 600
    outstr = (
        'timeout %d bash -lc \''
        'attempt=0; '
        'until [ "$(curl -o /dev/null -s -w "%%{http_code}" http://localhost:%d/health)" = "200" ]; do '
        'attempt=$((attempt+1)); '
        'sleep_for=$((attempt<5 ? attempt : 5)); '
        'sleep "$sleep_for"; '
        'done'
        '\' || { '
        'echo "{\\"event\\":\\"model_readiness_timeout\\",\\"port\\":%d,\\"max_wait_seconds\\":%d}" 1>&2; '
        'exit 1; '
        '}'
    ) % (max_wait_seconds, config.Port, config.Port, max_wait_seconds)
    print(outstr + "\n")

def process_data(container, config:ScriptConfig):
    outstr = 'find %s -type f -name \'*.out\' -print0' % q(f"{config.Path.LogPath}/{container}")
    outstr += '| xargs -0 python run_scripts/process_data.py --raw_data --add_config_header --output %s ' % q(config.Path.AnalysisPath)
    print(outstr + "\n")

def post_process_data(config:ScriptConfig):
    outstr = 'find %s -type f -name \'*.csv\' -print0' % q(f"{config.Path.AnalysisPath}/{config.DATE}")
    outstr += '| xargs -0 python run_scripts/post_process_data.py'
    print(outstr + "\n")

def date():
    outstr = 'date'
    print(outstr + "\n")

def echo(model):
    outstr = 'echo "%s"' % model 
    print(outstr + "\n")

def run_bench(container, model:ModelSpec, batch,config:ScriptConfig):
    name = model.name
    name_and_tag = name
    if model.tag and model.tag != "":
        name_and_tag += "+" + model.tag
    outstr = '# start bench for model %s\n' % name
    outstr += 'EXPDIR=%s && ' % q(f"{config.Path.LogPath}/{container}/{name_and_tag}")
    outstr += 'mkdir -p "${EXPDIR}" && touch "${EXPDIR}/bs-%s.out" && ' % batch
    outstr += 'docker exec -i %s bash -lc "' % q(container)
    outstr += 'vllm bench serve '
    outstr += '--model %s ' % q(model.path)
    outstr += '--served-model-name %s ' % q(name)
    outstr += '--dataset-name %s ' % q(config.Dataset.name)
    if config.Dataset.name == "random":
        outstr += '--random-input-len=%d ' % config.Dataset.input_len
        outstr += '--random-output-len=%d ' % config.Dataset.output_len
    else:
        outstr += '--dataset-path %s ' % q(config.Dataset.path)
    outstr += '--ignore-eos '
    outstr += '--num-prompt %d ' % batch
    outstr += '--trust-remote-code '
    outstr += '--request-rate inf '
    outstr += '--backend vllm '
    outstr += '--port=%d ' % config.Port
    outstr += '" 2>&1 | tee -a "${EXPDIR}/bs-%s.out"' % batch
    print(outstr + "\n")

def gen_run_scripts(config:ScriptConfig):
    validate_extra_param_keys(config)
    print('#!/bin/bash\n')
    container = config.DATE + "-" + config.VERSION.replace(' ', "_").replace('(',"_").replace(')',"_")
    
    create_container(container=container,config=config)
    for model in config.Model:
        date()
        echo(model)
        run_model(container=container,model=model,config=config)
        check_ready(config=config)
        for batch in model.batch:
            run_bench(container=container,model=model,batch=batch,config=config)
        stop_model(container=container)
    stop_container(container=container)
    rm_container(container=container)
    date()
    process_data(container,config=config)
    post_process_data(config=config)

def main():
    parser = argparse.ArgumentParser(description="LLM Scaler Auto Test Config")
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        config = ScriptConfig.from_dict(default_config)
    else:
        config = ScriptConfig.from_yaml(config_path)
    try:
        gen_run_scripts(config)
    except ValueError as err:
        print(f"Config validation error: {err}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
