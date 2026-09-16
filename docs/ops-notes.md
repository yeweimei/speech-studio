# 运维与踩坑记录（NUC12 部署）

> 给后续接手的人/agent：动手前先读完本页。这里每条都是实际踩出来的。

## 目标环境

| 项 | 值 |
|---|---|
| 部署主机 | **NUC12** / `192.168.3.246` / 用户 `zhangjiyu` |
| SSH 工具 | `ssh-manager__ssh_execute`，`server="nuc12"`（key 认证） |
| 远程 shell | **dash，不是 bash** —— 复杂脚本要用 `bash -c '...'` 包起来，顶层别用 `source` |
| 硬件 | i7-12700H（20 线程）/ Iris Xe 核显 / Arc A770M 16G 独显 / 61G 内存 |
| 工作目录 | 本地 `/home/jiyu/projects/speech-studio`（git 源）→ NUC12 `~/projects/speech-studio` |

## ⚠️ GPU 隔离（本项目最关键的设计）

```
renderD128 → pci-0000:00:02.0 → Iris Xe     ← 容器只挂这个
renderD129 → pci-0000:03:00.0 → Arc A770M   ← 绝对不要挂进容器
```

Arc 留给 `llama-studio` 上的 LLM 推理。**容器 run 时只 `--device /dev/dri/renderD128`**，
Arc 在内核层面不可见 —— 这是物理隔离，不靠约定。

另外：**Iris Xe 目前还被 llama-studio 的两个生产模型占用**
（`Qwen3-Embedding-0.6B`、`PaddleOCR-VL-1.6`，都是 `--device SYCL1`）。用户后续会搬走，
在此之前 TTS 会和它们共享核显。

## ⚠️ 容器内缺 Intel GPU 运行库（当前主要待解问题）

**openvino 的 pip wheel 不自带 Level Zero / IGC** —— 它只提供
`libopenvino_intel_gpu_plugin.so`，运行时需要系统提供：

```
libze_loader.so.1 / libze_intel_gpu.so.1 / libigc.so.2 / libigdfcl.so.2
```

基础镜像 `python:3.14-slim`（已 pull，191MB）**没有**这些。**Debian 官方源在国内 apt 会超时**，
必须换国内源（清华 tuna：`mirrors.tuna.tsinghua.edu.cn`）。

宿主机上这些库是有的（`/usr/lib/x86_64-linux-gnu/`），所以宿主机跑 OpenVINO 没问题 ——
**不要照着宿主机能跑来判断容器也能跑**。

## ⚠️ 绝对不要碰的东西

- **`llama-studio` 容器**：生产 LLM 网关（9100 端口，18 个模型注册，self_heal）。只读侦察可以，
  任何修改/重启都不要做。
- **宿主机 `/opt/intel/oneapi/compiler/2025.3`**：不完整（缺 dpcpp/icpx）。
  容器 `llama-studio:latest` 里那份才是完整的。

## ⚠️ 操作陷阱（都是血泪）

### 1. `pkill -f` / `pgrep -f` 会杀掉自己的 shell

只要你的命令行里**出现了进程名的字面量**（哪怕用 `[o]` 括号技巧也救不了 ——
因为启动命令里真的写了 `app_onnx.py`），`pgrep -f` 就会匹配到自己，然后 `kill` 把自己的 shell 干掉。
**表现：exit code 15，且完全没有任何输出**，看起来像"命令没执行"，极难排查。

**正确做法：按端口 kill**
```bash
PID=$(ss -ltnp | grep ':9300' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$PID" ] && kill $PID
```

### 2. `ssh_execute` 有约 5 分钟上限，长任务必须后台化

docker build / pip install 这类会超时（`MCP error -32001: Request timed out`），
而且**超时后你不知道任务是被杀了还是还在跑**。正确姿势：

```bash
setsid nohup bash -c '...' > /tmp/xxx.log 2>&1 < /dev/null &
# 然后轮询日志文件，不要靠 docker logs
```

### 3. `set -e` + `source setvars.sh` 静默退出

oneAPI 的 `setvars.sh` 可能返回非零，在 `set -e` 下会让脚本**立刻退出且没有任何输出**（exit 3）。
必须写 `source ... || true`。

### 4. torch 必须用 CPU 源

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
```
tuna 镜像上是 CUDA 版，镜像会大 2GB+。`download.pytorch.org` 国内可直连但慢（约 15 分钟）。

### 5. 国内镜像可用性

| 源 | 状态 |
|---|---|
| `pypi.tuna.tsinghua.edu.cn` | ✅ 快（pip 首选） |
| `hf-mirror.com` | ✅ 可直连（模型下载） |
| `ghfast.top` 代理 | ✅ GitHub 源码（NUC12 **不能直连 GitHub**） |
| `download.pytorch.org` | ⚠️ 可直连但慢 |
| `pypi.org` / `github.com` / `huggingface.co` | ❌ 不通 |

## 宿主机的既有资产（可复用）

| 路径 | 内容 |
|---|---|
| `~/moss-tts-nano/` | TTS 仓库 + `models/`（2.9G，**不复制进镜像，挂载**） |
| `~/models/whisper-small-ct2/` | ct2 模型（464M，STT 用） |
| `~/venv/tts` | python3.14 venv，TTS 依赖（含已调好的 openvino / torch） |
| `~/venv/stt` | python3.14 venv，faster-whisper + openvino-genai |
| `/dev/dri/renderD128` | **Iris Xe**，容器挂载点 |

### moss-tts-nano 上已打的补丁（**不要回退**）

1. `ov_session.py`
   - 默认 device 改读环境变量 `MOSS_OV_DEVICE`（原来硬编码 `"GPU"`，
     而双 GPU 机器上设备名是 `GPU.0`/`GPU.1`，**没有叫 `GPU` 的** → 会静默回退 CPU）
   - `"GPU"` 不在可用列表时自动取第一个 `GPU.*`
   - 加了一行 stderr 探针打印实际落盘设备
2. `ort_cpu_runtime.py`
   - `_session()` 默认值 + 3 处硬编码 `ov_device="GPU"` 全改读 `MOSS_OV_DEVICE`
3. `app_onnx.py`
   - `--execution-provider` 的 `choices` 补上 `"openvino"`（原来只有 cpu/cuda，
     导致 HTTP 服务**根本无法使用核显**）

## 为什么不用 moss 内置的 `app.py` 服务

它的启动 warmup 硬依赖 `WeTextProcessing` → `pynini` → **OpenFst（需编译）**，
Python 3.14 上装不了。warmup 失败会导致**每个请求 500**，且 `enable_text_normalization=0`
绕不过去（warmup 链路自己会调 `normalize()`）。

本服务改走 `ort_cpu_runtime` 路径，用 `enable_wetext=False` 绕开，功能不受影响。

## 性能基线（回归对比用）

| 项 | 数值 |
|---|---|
| TTS 合成（18 字 → 3.36s 音频，Iris Xe） | ~3.4s（RTF ≈ 1.02） |
| TTS 模型加载（仅启动一次） | ~3.2s |
| STT 52.5s 音频（ct2 int8，CPU 8 线程） | ~5.1s（10x 实时） |
| STT 单句延迟地板 | ~1.2s |

**别用 20 线程跑 ct2**：8 线程 5.09s，20 线程反而 9.42s（超线程负收益）。
