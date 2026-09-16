# speech-studio —— 统一音频服务容器
#   TTS   moss-tts-nano / OpenVINO / Iris Xe 核显（只挂 /dev/dri/renderD128）
#   STT   faster-whisper ct2 int8 / CPU
#
# 背景/设计（见 docs/ops-notes.md）：
#   * openvino 的 pip wheel 不带 Intel GPU 运行库（libze/libigc）。
#     Debian 官方 apt 不打包 Intel NEO 计算运行时（已实测：trixie 全分区 apt-cache
#     search 均无 intel-opencl-icd / libze-intel-gpu1 / libigc1），
#     且国内 apt 超时。解决方案：
#       - 换清华 tuna 源做 apt（快、稳）
#       - Intel GPU 运行库（loader/驱动/IGC）从 Intel 官方 GitHub 发布的 .deb 提取，
#         runtime/debs/ 里已内置（都经 ghfast.top 下载并核对 glibc 下限 ≤2.38，
#         与 Debian trixie 的 glibc 2.41 兼容）
#   * torch/torchaudio 必须走 CPU 源（tuna 是 CUDA 版，大 2GB+）
#   * 只 COPY app/；模型不进镜像，运行时挂载
#
# 基础镜像固定：python:3.14-slim（Debian trixie，glibc 2.41，NUC12 已 pull）
FROM python:3.14-slim

# ── 1. apt 换清华源 + 装运行所需系统库 ────────────────────────────────
# Debian 官方源国内会超时；tuna 镜像 main/contrib/non-free 全可用。
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' \
        /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
        libatomic1 \
        ca-certificates \
        ocl-icd-libopencl1 \
        && rm -rf /var/lib/apt/lists/*

# ── 2. Intel GPU 运行库（从 runtime/debs/ 内置的官方 .deb 解压） ───────
# 不会用 dpkg 安装（避免引入无关依赖），只 dpkg-deb -x 解开 .so 到标准路径，
# 再 ldconfig。含：Level Zero loader + NEO 驱动 + IGC 编译器 + GMM
# + OpenCL ICD（libigdrcl）。
# 关键经验：OpenVINO 2026.3 的 Intel GPU plugin 通过 **OpenCL 平台枚举**发现 GPU，
# 光有 Level Zero 栈不够——必须同时注册 Intel OpenCL ICD，否则 GPU 静默回退 CPU。
# 这里除 .so 外，还显式写 /etc/OpenCL/vendors/intel.icd 指向 libigdrcl.so。
COPY runtime/debs/ /tmp/intel-runtime-debs/
RUN set -eux; \
    mkdir -p /opt/intel-runtime && \
    for f in /tmp/intel-runtime-debs/*.deb; do \
        dpkg-deb -x "$f" /opt/intel-runtime; \
    done; \
    # 把 /opt/intel-runtime 下的 .so 统一归位到标准 ld 路径
    mkdir -p /usr/lib/x86_64-linux-gnu && \
    find /opt/intel-runtime -name '*.so*' -exec cp -a {} /usr/lib/x86_64-linux-gnu/ \; ; \
    # 注册 Intel OpenCL ICD（OpenVINO GPU 插件发现 GPU 依赖它）
    mkdir -p /etc/OpenCL/vendors && \
    echo "/usr/lib/x86_64-linux-gnu/libigdrcl.so" > /etc/OpenCL/vendors/intel.icd && \
    # 保留 SONAME 符号链接（cp -a 已带），重建 loader 缓存
    ldconfig; \
    rm -rf /tmp/intel-runtime-debs /opt/intel-runtime

# ── 3. Python 依赖 ────────────────────────────────────────────────────
# torch/torchaudio：用纯 CPU wheel（download.pytorch.org 的 +cpu 构建，无 CUDA，
# ~196MB）。tuna 的 PyPI 里 torch 默认是 CUDA 版（554MB+，含 CUDA 库），
# 上游已禁（见 ops-notes）。为避免构建时依赖不稳定的 download.pytorch.org，
# 这两个 +cpu wheel 由 deploy.sh 预下载到 runtime/wheels/（此目录 gitignore），
# 这里 COPY 进来用本地文件安装。其余依赖全部走清华 tuna 快源。
# 注意：本地 wheel 安装仍会解析 torch 的依赖（filelock/sympy 等），
# 必须同一命令挂 -i tuna，否则 pip 默认走 pypi.org（国内不通）会卡死。
COPY runtime/wheels/ /wheels/
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        /wheels/torch-*.whl \
        /wheels/torchaudio-*.whl \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "python-multipart>=0.0.9" \
        numpy \
        soundfile \
        sentencepiece \
        cn2an \
        "openvino>=2026.3" \
        "onnxruntime>=1.20" \
        "transformers>=4.54" \
        faster-whisper

# ── 4. 应用代码（只 COPY app/，模型走挂载，不进镜像） ─────────────────
WORKDIR /app
COPY app/ /app/

# ── 5. 运行环境 ───────────────────────────────────────────────────────
ENV \
    HOST=0.0.0.0 \
    PORT=9300 \
    MOSS_OV_DEVICE=GPU \
    MOSS_TTS_DIR=/opt/moss-tts-nano \
    STT_MODEL_DIR=/models/whisper-small-ct2 \
    TTS_THREADS=8 \
    STT_THREADS=8 \
    PYTHONUNBUFFERED=1

EXPOSE 9300

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ['PORT']+'/health',timeout=4); sys.exit(0)" || exit 1

CMD ["python", "main.py"]