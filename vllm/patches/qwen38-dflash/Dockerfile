FROM qwen36-b70-vllm:b3-maxperf-final-v7
# Kernel fix source: https://github.com/rmacy/vllm (fork of vllm-project/vllm, main branch)
COPY qwen3_dflash.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
COPY dflash.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py
COPY utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/utils.py
COPY gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
COPY registry.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/registry.py
RUN find /opt/venv/lib/python3.12/site-packages/vllm -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null; \
    python -c "from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM, VanillaMarkov, AcceptRatePredictor; print(\"DSpark model classes import OK\")" && \
    python -c "from vllm.v1.spec_decode.dflash import DFlashProposer; print(\"DSpark proposer import OK\")" && \
    python -c "from vllm.model_executor.models.registry import ModelRegistry; print(\"registry import OK\")"
