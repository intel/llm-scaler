import argparse

from script_config import ScriptConfig, default_config, ModelSpec

def create_container(container,config:ScriptConfig):
    outstr = '# create container version %s\n' % config.VERSION
    outstr += 'sudo docker run -td '
    outstr += '--privileged '
    outstr += '--net=host '
    outstr += '--device=/dev/dri '
    outstr += '--name=%s ' % container
    for path, path_map in config.Path.ModelPathMap.items():
        outstr += '-v %s:%s ' % (path, path_map)
    if config.Path.TestPath and config.Path.TestPath != "":
        outstr += '-v %s:/llm/test/ ' % (config.Path.TestPath)
    outstr += '-e no_proxy=localhost,127.0.0.1 '
    outstr += '-e http_proxy=$http_proxy '
    outstr += '-e https_proxy=$https_proxy '
    outstr += '--shm-size="32g" '
    outstr += '--entrypoint /bin/bash '
    outstr += '%s:%s ' % (config.REPO, config.VERSION)
    print(outstr + "\n")

def start_container(container):
    outstr = 'sudo docker start %s ' % container
    print(outstr + "\n")

def stop_container(container):
    outstr = 'sudo docker stop %s ' % container
    print(outstr + "\n")

def rm_container(container): 
    outstr = 'sudo docker rm %s ' % container
    print(outstr + "\n")

def run_model(container, model:ModelSpec, config:ScriptConfig):
    outstr = '# start model %s\n' % model.name
    if model.tag and model.tag != "":
        outstr += 'EXPDIR="%s/%s/%s+%s" && ' % (config.Path.LogPath, container, model.name, model.tag)
    else:
        outstr += 'EXPDIR="%s/%s/%s" && ' % (config.Path.LogPath, container, model.name)
    outstr += 'mkdir -p ${EXPDIR} && touch "${EXPDIR}/model.log" && '
    outstr += 'docker exec -i %s bash -lc "' % container
    if config.XPU:
        outstr += 'ZE_AFFINITY_MASK=%s ' % config.XPU 
    outstr += 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 '
    outstr += 'VLLM_WORKER_MULTIPROC_METHOD=spawn '
    outstr += 'python3 -m vllm.entrypoints.openai.api_server '
    outstr += '--model %s ' % model.path
    outstr += '--served-model-name %s ' % model.name
    outstr += '--dtype=float16 '
    outstr += '--enforce-eager '
    outstr += '--port %d ' % config.Port
    outstr += '--host 0.0.0.0 '
    outstr += '--trust-remote-code '
    outstr += '--disable-sliding-window '
    outstr += '--gpu-memory-util=0.9 '
    outstr += '--no-enable-prefix-caching '
    outstr += '--max-num-batched-tokens=2048 '
    outstr += '--disable-log-requests '
    outstr += '--max-model-len=2048 '
    outstr += '--block-size 64 '
    if model.quantization:
        outstr += '--quantization %s ' % model.quantization
    outstr += '-tp=%d ' % model.tp
    if model.spec_config:
        outstr += '--speculative_config=\'{\\"method\\": \\"%s\\", \
        \\"model\\": \\"%s\\", \\"num_speculative_tokens\\": %d}\' ' \
        % (model.spec_config.method, model.spec_config.model, 
           model.spec_config.num_speculative_tokens)
    if model.extra_param:
        for flag, value in model.extra_param.items():
            outstr += '%s=%s ' % (flag,value)
    outstr += '" 2>&1 | tee -a "${EXPDIR}/model.log" &'
    print(outstr + "\n")

def check_ready(config:ScriptConfig):
    outstr = 'until [ $(curl -o /dev/null -s -w "%%{http_code}\\n" http://localhost:%d/health) = "200" ]; do sleep 1; done' % config.Port
    print(outstr + "\n")

def process_data(container, config:ScriptConfig):
    outstr = 'find %s/%s -type f -name \'*.out\' -print0' % (config.Path.LogPath, container)
    outstr += '| xargs -0 -I{} python run_scripts/process_data.py --raw_data "{}" --add_config_header True --output %s ' % (config.Path.AnalysisPath) 
    print(outstr + "\n")

def post_process_data(config:ScriptConfig):
    outstr = 'find %s/%s -type f -name \'*.csv\' -print0' % (config.Path.AnalysisPath, config.DATE)
    outstr += '| xargs -0 -I{} python run_scripts/post_process_data.py "{}"'
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
    outstr += 'EXPDIR="%s/%s/%s" && ' % (config.Path.LogPath,container, name_and_tag)
    outstr += 'mkdir -p "${EXPDIR}" && touch "${EXPDIR}/bs-%s.out" && ' % batch
    outstr += 'docker exec -i %s bash -lc "' % container
    outstr += 'vllm bench serve '
    outstr += '--model %s ' % model.path
    outstr += '--served-model-name %s ' % name
    outstr += '--dataset-name %s ' %  config.Dataset.name
    if config.Dataset.name == "random":
        outstr += '--random-input-len=%d ' % config.Dataset.input_len
        outstr += '--random-output-len=%d ' % config.Dataset.output_len
    else:
        outstr += 'dataset-path %s' % config.Dataset.path
    outstr += '--ignore-eos '
    outstr += '--num-prompt %d ' % batch
    outstr += '--trust_remote_code '
    outstr += '--request-rate inf '
    outstr += '--backend vllm '
    outstr += '--port=%d ' % config.Port
    outstr += '" 2>&1 | tee -a "${EXPDIR}/bs-%s.out"' % batch
    print(outstr + "\n")

def gen_run_scripts(config:ScriptConfig):
    print('#!/bin/bash\n')
    container = config.DATE + "-" + config.VERSION.replace(' ', "_").replace('(',"_").replace(')',"_")
    
    for model in config.Model:
        create_container(container=container,config=config)
        date()
        echo(model)
        run_model(container=container,model=model,config=config)
        check_ready(config=config)
        for batch in model.batch:
            run_bench(container=container,model=model,batch=batch,config=config)
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
    gen_run_scripts(config)

if __name__ == "__main__":
    main()