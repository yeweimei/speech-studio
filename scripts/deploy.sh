#!/usr/bin/env bash
# speech-studio —— 构建 + 启动一条龙（在 NUC12 上执行）
#
# 关键约束（见 docs/ops-notes.md）：
#   * 只挂 /dev/dri/renderD128（Iris Xe）。绝对不挂 renderD129（Arc A770M 留给 llama-studio）
#   * 挂载模型目录，模型不进镜像
#   * 端口 9300，容器内 PORT=9300
#   * 重启策略 unless-stopped，容器名 speech-studio
#   * 幂等：重复执行安全重建（按容器名 rm -f）
set -euo pipefail

IMG="speech-studio:latest"
NAME="speech-studio"
PORT=9300
DRI="/dev/dri/renderD128"

# ── CPU torch wheel 仓库（vendored，不进 git）────────────────────────
# tspy 的 PyPI 里 torch 默认是 CUDA 版（554MB+），必须用 download.pytorch.org
# 的 +cpu wheel（~196MB）。该源不稳定，这里预下到 runtime/wheels/（gitignore），
# Dockerfile 用本地文件安装，避免构建时依赖不稳定源。断点续传 + 重试。
WHL_DIR="$(cd "$(dirname "$0")/.." && pwd)/runtime/wheels"
TORCH_URL="https://download.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl"
TORCHAUDIO_URL="https://download.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl"
mkdir -p "$WHL_DIR"
_fetch() { # url dest min_bytes
  local url="$1" dest="$2" want="$3" have=0 i
  [ -f "$dest" ] && have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  [ "$have" -ge "$want" ] && { echo "[deploy] wheel 已存在: $(basename "$dest") ($have B)"; return 0; }
  echo "[deploy] 下载 $(basename "$dest")（断点续传 + 重试）..."
  for i in $(seq 1 40); do
    have=$([ -f "$dest" ] && stat -c%s "$dest" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && { echo "[deploy] 完成: $(basename "$dest") ($have B)"; return 0; }
    timeout 120 curl -sSL -C - -o "$dest" "$url" && continue
    echo "[deploy]   重试 $i..."; sleep 3
  done
  echo "[deploy] 下载失败: $(basename "$dest")" >&2; return 1
}
_fetch "$TORCH_URL" "$WHL_DIR/torch-2.14.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl" 190000000
_fetch "$TORCHAUDIO_URL" "$WHL_DIR/torchaudio-2.11.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl" 300000


# ── 前置校验：必须能看到核显 render node ───────────────────────────────
if [ ! -e "$DRI" ]; then
  echo "[deploy] 警告：$DRI 不存在，Iris Xe render node 未就绪（可能需把用户加入 render/video 组）" >&2
  echo "[deploy] 继续构建，但 GPU 推理可能失败。" >&2
fi

echo "[deploy] 构建镜像 ${IMG} ..."
docker build -t "$IMG" .

# ── 幂等重建：删除已存在的同名容器 ─────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[deploy] 移除旧容器 ${NAME}（幂等重建）"
  docker rm -f "$NAME" >/dev/null
fi

# ── 启动：只挂 renderD128 ──────────────────────────────────────────────
echo "[deploy] 启动容器 ${NAME}（仅挂 ${DRI}，端口 ${PORT}）..."
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --device "$DRI" \
  -p "${PORT}:${PORT}" \
  -e "PORT=${PORT}" \
  -e "HOST=0.0.0.0" \
  -e "MOSS_OV_DEVICE=GPU" \
  -v "${HOME}/moss-tts-nano:/opt/moss-tts-nano" \
  -v "${HOME}/models/whisper-small-ct2:/models/whisper-small-ct2" \
  "$IMG"

echo "[deploy] 已启动。等待 /health 就绪（最长 150s）..."
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/ss-health.json 2>/dev/null; then
    echo "[deploy] 就绪："
    cat /tmp/ss-health.json
    echo
    echo "[deploy] 调试台：http://<nuc12>:${PORT}/"
    exit 0
  fi
  sleep 5
done
echo "[deploy] 150s 内未就绪，请查看：docker logs ${NAME}" >&2
exit 1