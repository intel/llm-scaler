#!/usr/bin/env bash
set -e

# ==== 配置区域 ====
COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${COMPOSE_DIR}"

HAPROXY_SOCK=127.0.0.1:9999
LOG_FILE=/tmp/vllm_rotate.log
CRON_CMD="* 3 * * * ${COMPOSE_DIR}/vllm_bootstrap_and_rotate.sh >> ${LOG_FILE} 2>&1"
echo "${CRON_CMD}"
# ==== Step 0: 确保 HAProxy + 至少一个 vLLM 运行 ====
echo "==> Ensuring HAProxy + at least one vLLM is running..."

docker compose -f docker-compose.yml -f docker-compose.env-override.yml up -d haproxy

# 检查 vLLM 容器是否运行
VLLM_1_RUNNING=$(docker ps --filter "name=vllm_1" --filter "status=running" | grep -q vllm_1 && echo 1 || echo 0)
VLLM_2_RUNNING=$(docker ps --filter "name=vllm_2" --filter "status=running" | grep -q vllm_2 && echo 1 || echo 0)

if [[ "$VLLM_1_RUNNING" == "0" && "$VLLM_2_RUNNING" == "0" ]]; then
  echo "==> No vLLM running, starting vllm_1..."
  docker compose -f docker-compose.yml -f docker-compose.env-override.yml up -d vllm_1

else
  echo "==> At least one vLLM already running, skipping initial start"
fi


# ==== Step 0.5: 等待 HAProxy socket 就绪 ====
echo "==> Waiting for HAProxy socket..."
for i in {1..20}; do
  if echo "show info" | socat -t 2 stdio TCP:${HAPROXY_SOCK} >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! echo "show info" | socat -t 2 stdio TCP:${HAPROXY_SOCK} >/dev/null 2>&1; then
  echo "[ERROR] HAProxy socket not ready"
  exit 1
fi

# ==== Step 1: 判断哪个 vLLM 是旧实例，哪个是新实例 ====
if docker ps --filter "name=vllm_1" --filter "status=running" | grep -q vllm_1; then
  OLD=vllm_1
  NEW=vllm_2
else
  OLD=vllm_2
  NEW=vllm_1
fi

METRIC_PORT_OLD=$([ "$OLD" == "vllm_1" ] && echo 8008 || echo 8009)
METRIC_PORT_NEW=$([ "$NEW" == "vllm_1" ] && echo 8008 || echo 8009)

# ==== Step 2: 启动新 vLLM ====
echo "==> Starting new vLLM: ${NEW}"
docker compose -f docker-compose.yml -f docker-compose.env-override.yml up -d ${NEW}

# ==== Step 3: 等待新 vLLM 健康 ====
echo "==> Waiting for ${NEW} to be healthy..."
until curl -sf http://127.0.0.1:${METRIC_PORT_NEW}/health > /dev/null; do
  sleep 5
done

# ==== Step 4: 启用新 backend ====
echo "==> Enabling ${NEW} in HAProxy..."
echo "enable server vllm_backend/${NEW}" | socat stdio TCP:${HAPROXY_SOCK}
sleep 2 

# ==== Step 5: 禁用旧 backend ====
echo "==> Disabling ${OLD} in HAProxy..."
echo "disable server vllm_backend/${OLD}" | socat stdio TCP:${HAPROXY_SOCK}

# ==== Step 6: 等待旧 vLLM drain ====
echo "==> Waiting for old vLLM to drain..."
while true; do
  RUNNING=$(curl -s http://127.0.0.1:${METRIC_PORT_OLD}/metrics \
    | grep '^vllm:num_requests_running' | awk '{print $2}')
  WAITING=$(curl -s http://127.0.0.1:${METRIC_PORT_OLD}/metrics \
    | grep '^vllm:num_requests_waiting' | awk '{print $2}')

  if [[ "${RUNNING}" == "0.0" && "${WAITING}" == "0.0" ]]; then
    break
  fi
  sleep 5
done

# ==== Step 7: 停止旧 vLLM ====
echo "==> Stopping old vLLM: ${OLD}"
docker stop ${OLD}

echo "==> Rotation complete ✅"

# 获取当前 crontab
CURRENT_CRON=$(crontab -l 2>/dev/null || true)

echo "=== 当前 crontab ==="
echo "$CURRENT_CRON"
echo "==================="

# 判断是否已经注册
if ! echo "$CURRENT_CRON" | grep -F -q "${COMPOSE_DIR}/vllm_bootstrap_and_rotate.sh"; then
    echo "==> Cron not found, registering..."
    # 保留原有 cron，追加新 cron
    (echo "$CURRENT_CRON"; echo "$CRON_CMD") | crontab -
    echo "Cron registered:"
    crontab -l
else
    echo "==> Cron already registered, skipping"
fi

