# speech-studio

统一音频模型服务：**一个容器**同时提供 TTS 与 STT，接口按 OpenAI 风格。

| 能力 | 接口 | 跑在哪 |
|---|---|---|
| TTS 语音合成 | `POST /v1/audio/speech` | **Iris Xe 核显**（OpenVINO） |
| STT 语音转写 | `POST /v1/audio/transcriptions` | **CPU**（faster-whisper ct2 int8，8 线程） |
| 调试台 | `GET /` | 单页 HTML，试听 + 试转写 |
| 健康检查 | `GET /health` | 就绪状态 / 设备 / 加载耗时 |
| 模型清单 | `GET /v1/models` | 含可用音色列表 |

## 为什么这么设计

- **核显做 TTS，Arc 留给 LLM。** 容器只挂 `renderD128`（Iris Xe）的 render node，
  `renderD129`（Arc A770M）**根本不进容器** —— 物理隔离，不是靠约定。
- **不走内置 `app.py` 服务。** 它启动时的 warmup 硬依赖 `WeTextProcessing`(pynini/OpenFst)，
  在 Python 3.14 上装不了，会导致每个请求 500。本服务改走 `ort_cpu_runtime` 路径，
  用 `enable_wetext=False` 绕开，功能不受影响。
- **模型进程内常驻。** CLI 每次要付 3.18s 模型加载；常驻后只剩合成时间。

## 快速开始

```bash
# 构建 + 启动（在 NUC12 上）
cd ~/projects/speech-studio && ./deploy.sh

# 试一句
curl -s -X POST http://127.0.0.1:9300/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"moss-tts-nano","input":"你好，这是一次测试。","voice":"Junhao","response_format":"wav"}' \
  -o out.wav

# 转写
curl -s -X POST http://127.0.0.1:9300/v1/audio/transcriptions \
  -F file=@out.wav -F language=zh
```

浏览器打开 `http://<nuc12>:9300/` 是调试台。

## 实测基线（供回归对比）

| 项 | 数值 |
|---|---|
| TTS 合成（18 字 → 3.36s 音频） | 核显 ~3.4s（RTF ≈ 1.02） |
| TTS 模型加载（仅启动时一次） | ~3.2s |
| STT 52.5s 音频 | ~5.1s（10x 实时） |
| STT 单句延迟地板 | ~1.2s |

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `9300` | 监听端口 |
| `MOSS_OV_DEVICE` | `GPU` | OpenVINO 设备；容器内只有一块 GPU，`GPU` 即可 |
| `MOSS_TTS_DIR` | `/opt/moss-tts-nano` | moss-tts-nano 仓库路径 |
| `STT_MODEL_DIR` | `/models/whisper-small-ct2` | ct2 模型目录 |
| `TTS_THREADS` | `8` | OpenVINO CPU 线程 |
| `STT_THREADS` | `8` | ct2 线程（**别调成 20，实测更慢**） |
