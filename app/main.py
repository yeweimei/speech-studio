#!/usr/bin/env python3
"""
speech-studio —— 统一音频模型服务（OpenAI 风格接口）

  TTS  POST /v1/audio/speech          moss-tts-nano / OpenVINO / Iris Xe 核显
  STT  POST /v1/audio/transcriptions  faster-whisper ct2 int8 / CPU
  调试  GET  /                         单页试听 + 试转写
  健康  GET  /health
  模型  GET  /v1/models

设计要点：
  * 模型进程内常驻（避免每次 3.18s 加载），用锁串行化，单卡不并发。
  * TTS 走 ort_cpu_runtime 路径（`execution_provider="openvino"`），
    刻意绕开内置 app.py 的 WeTextProcessing(pynini) 硬依赖：
    enable_wetext=False + enable_normalize_tts_text=True。
  * 设备由环境变量 MOSS_OV_DEVICE 控制（GPU / GPU.0 / GPU.1 / CPU）。
"""
from __future__ import annotations

import glob
import io
import logging
import os
import sys
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

# ── 路径配置 ────────────────────────────────────────────────────────────
MOSS_DIR = Path(os.environ.get("MOSS_TTS_DIR", "/opt/moss-tts-nano"))
STT_MODEL_DIR = Path(os.environ.get("STT_MODEL_DIR", "/models/whisper-small-ct2"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

TTS_THREADS = int(os.environ.get("TTS_THREADS", "8"))
TTS_MAX_NEW_FRAMES = int(os.environ.get("TTS_MAX_NEW_FRAMES", "2000"))
STT_THREADS = int(os.environ.get("STT_THREADS", "8"))
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")
OUTPUT_DIR = Path(os.environ.get("TTS_OUTPUT_DIR", "/tmp/speech-studio-out"))

sys.path.insert(0, str(MOSS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("speech-studio")


# ── TTS ────────────────────────────────────────────────────────────────
class TTSService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.load_s = 0.0
        self.voices: list[str] = []
        self.sample_rate = 48000

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        from onnx_tts_runtime import OnnxTtsRuntime  # noqa: E402

        self.rt = OnnxTtsRuntime(
            model_dir=None,  # -> REPO_ROOT/models（由 MOSS_DIR 决定）
            thread_count=TTS_THREADS,
            max_new_frames=TTS_MAX_NEW_FRAMES,
            execution_provider="openvino",
            output_dir=OUTPUT_DIR,
        )
        self.voices = [str(v["voice"]) for v in self.rt.list_builtin_voices()]
        self.sample_rate = int(self.rt.codec_meta["codec_config"]["sample_rate"])
        self.load_s = time.time() - t0
        self.ready = True

    def default_voice(self) -> str:
        return self.voices[0] if self.voices else "Junhao"

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        sample_mode: str = "fixed",
        do_sample: bool = True,
        max_new_frames: int | None = None,
        seed: int | None = None,
    ) -> tuple[Path, float]:
        if not self.ready:
            raise RuntimeError(self.error or "TTS 未就绪")
        with self.lock:
            t0 = time.time()
            out = OUTPUT_DIR / f"tts-{int(time.time() * 1000)}-{os.getpid()}.wav"
            res = self.rt.synthesize(
                text=text,
                voice=voice or self.default_voice(),
                prompt_audio_path=None,
                output_audio_path=out,
                sample_mode=sample_mode,
                do_sample=do_sample,
                streaming=False,
                max_new_frames=max_new_frames or TTS_MAX_NEW_FRAMES,
                voice_clone_max_text_tokens=75,
                enable_wetext=False,           # ← 绕开 pynini/WeTextProcessing
                enable_normalize_tts_text=True,
                seed=seed,
            )
            return Path(str(res["audio_path"])), time.time() - t0


# ── STT ────────────────────────────────────────────────────────────────
class STTService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ready = False
        self.error: str | None = None
        self.load_s = 0.0
        self.model_path = self._resolve_model()

        t0 = time.time()
        from faster_whisper import WhisperModel  # noqa: E402

        self.m = WhisperModel(
            self.model_path,
            device="cpu",
            compute_type=STT_COMPUTE_TYPE,
            cpu_threads=STT_THREADS,
        )
        self.load_s = time.time() - t0
        self.ready = True

    @staticmethod
    def _resolve_model() -> str:
        explicit = os.environ.get("STT_MODEL_PATH")
        if explicit:
            return explicit
        snaps = sorted(glob.glob(str(STT_MODEL_DIR / "snapshots" / "*")))
        if snaps:
            return snaps[0]
        if STT_MODEL_DIR.is_dir():
            return str(STT_MODEL_DIR)
        raise FileNotFoundError(f"找不到 STT 模型: {STT_MODEL_DIR}")

    def transcribe(
        self, audio_path: str, language: str = "zh", beam_size: int = 5
    ) -> tuple[str, list[dict], float]:
        if not self.ready:
            raise RuntimeError(self.error or "STT 未就绪")
        lang = None if language in ("", "auto", None) else language
        with self.lock:
            t0 = time.time()
            segments, _info = self.m.transcribe(
                audio_path,
                language=lang,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                condition_on_previous_text=False,
                initial_prompt="以下是普通话的句子，请用简体中文输出。" if lang in (None, "zh") else None,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
            )
            segs = list(segments)
            text = "".join(s.text for s in segs).strip()
            detail = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segs
            ]
            return text, detail, time.time() - t0


TTS: TTSService | None = None
STT: STTService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global TTS, STT
    log.info("加载 TTS（设备=%s, provider=openvino）...", os.environ.get("MOSS_OV_DEVICE", "GPU"))
    try:
        TTS = TTSService()
        log.info("TTS 就绪 %.2fs，音色=%s", TTS.load_s, TTS.voices)
    except Exception as exc:  # 允许 STT-only 启动
        log.exception("TTS 加载失败")
        TTS = TTSService.__new__(TTSService)
        TTS.ready, TTS.error, TTS.load_s, TTS.voices, TTS.lock = False, str(exc), 0.0, [], threading.Lock()

    log.info("加载 STT（%s）...", STT_MODEL_DIR)
    try:
        STT = STTService()
        log.info("STT 就绪 %.2fs", STT.load_s)
    except Exception as exc:
        log.exception("STT 加载失败")
        STT = STTService.__new__(STTService)
        STT.ready, STT.error, STT.load_s, STT.lock = False, str(exc), 0.0, threading.Lock()

    yield
    log.info("speech-studio 退出")


app = FastAPI(title="speech-studio", version="0.1.0", lifespan=lifespan)


def _wav_bytes(path: Path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


# ── OpenAI 风格：语音合成 ───────────────────────────────────────────────
@app.post("/v1/audio/speech")
async def audio_speech(payload: dict):
    if TTS is None or not TTS.ready:
        raise HTTPException(503, f"TTS 不可用: {getattr(TTS, 'error', None)}")
    text = str(payload.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "input 不能为空")

    voice = payload.get("voice") or None
    if voice and voice not in TTS.voices:
        raise HTTPException(400, f"未知音色 {voice}；可用: {TTS.voices}")

    fmt = str(payload.get("response_format") or "wav").lower()
    if fmt != "wav":
        raise HTTPException(400, "当前仅支持 response_format=wav")

    try:
        out, elapsed = TTS.synthesize(
            text=text,
            voice=voice,
            sample_mode=str(payload.get("sample_mode") or "fixed"),
            do_sample=bool(payload.get("do_sample", True)),
            max_new_frames=payload.get("max_new_frames"),
            seed=payload.get("seed"),
        )
    except Exception as exc:
        log.exception("TTS 合成失败")
        raise HTTPException(500, f"合成失败: {exc}") from exc

    log.info("TTS 完成 %.2fs | %d 字 | voice=%s", elapsed, len(text), voice or TTS.default_voice())
    return Response(
        content=_wav_bytes(out),
        media_type="audio/wav",
        headers={
            "X-Gen-Seconds": f"{elapsed:.2f}",
            "X-Voice": voice or TTS.default_voice(),
            "Content-Disposition": f'inline; filename="{out.name}"',
        },
    )


# ── OpenAI 风格：语音转写 ───────────────────────────────────────────────
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-small"),
    language: str = Form("zh"),
    response_format: str = Form("json"),
    beam_size: int = Form(5),
):
    if STT is None or not STT.ready:
        raise HTTPException(503, f"STT 不可用: {getattr(STT, 'error', None)}")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = OUTPUT_DIR / f"stt-{int(time.time() * 1000)}-{os.getpid()}{suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
        text, detail, elapsed = STT.transcribe(str(tmp), language=language, beam_size=beam_size)
    except Exception as exc:
        log.exception("STT 转写失败")
        raise HTTPException(500, f"转写失败: {exc}") from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    log.info("STT 完成 %.2fs | %d 段 | %d 字", elapsed, len(detail), len(text))
    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    if response_format == "verbose_json":
        return JSONResponse({"text": text, "segments": detail, "gen_seconds": round(elapsed, 2)})
    return JSONResponse({"text": text})


# ── 元信息 ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok" if (TTS and TTS.ready and STT and STT.ready) else "degraded",
        "tts": {
            "ready": bool(TTS and TTS.ready),
            "error": getattr(TTS, "error", None),
            "device": os.environ.get("MOSS_OV_DEVICE", "GPU"),
            "provider": "openvino",
            "voices": getattr(TTS, "voices", []),
            "load_seconds": round(getattr(TTS, "load_s", 0.0), 2),
        },
        "stt": {
            "ready": bool(STT and STT.ready),
            "error": getattr(STT, "error", None),
            "model": getattr(STT, "model_path", None),
            "compute_type": STT_COMPUTE_TYPE,
            "cpu_threads": STT_THREADS,
            "load_seconds": round(getattr(STT, "load_s", 0.0), 2),
        },
    }


@app.get("/v1/models")
async def models():
    data = []
    if TTS and TTS.ready:
        data.append({"id": "moss-tts-nano", "object": "model", "owned_by": "local",
                     "task": "text-to-speech", "voices": TTS.voices})
    if STT and STT.ready:
        data.append({"id": "whisper-small", "object": "model", "owned_by": "local",
                     "task": "speech-to-text"})
    return {"object": "list", "data": data}


@app.get("/", response_class=HTMLResponse)
async def index():
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return HTMLResponse("<h1>speech-studio</h1><p>缺少调试页 static/index.html</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "9300")),
        log_level="info",
    )
