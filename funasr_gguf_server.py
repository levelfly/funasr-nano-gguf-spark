import sys
import os
import time
import tempfile
import subprocess
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

# cuDNN path for CUDA EP
os.environ["LD_LIBRARY_PATH"] = "/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, "/home/jayter")
from fun_asr_gguf.inference.asr_engine import create_asr_engine

app = FastAPI(title="Fun-ASR-Nano-GGUF API")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = "/home/jayter/models/Fun-ASR-Nano-GGUF"

print("Loading Fun-ASR-Nano-GGUF engine (CUDA)...")
t0 = time.time()
engine = create_asr_engine(
    encoder_onnx_path=os.path.join(MODEL_DIR, "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"),
    ctc_onnx_path=os.path.join(MODEL_DIR, "Fun-ASR-Nano-CTC.int4.onnx"),
    decoder_gguf_path=os.path.join(MODEL_DIR, "Fun-ASR-Nano-Decoder.q5_k.gguf"),
    tokens_path=os.path.join(MODEL_DIR, "tokens.txt"),
    enable_ctc=False,
    dml_enable=False,
    vulkan_enable=False,
    verbose=True,
)
print(f"Engine loaded in {time.time()-t0:.1f}s")

ERROR_PATTERNS = ["解码有误", "熔断", "/sil"]

def preprocess_audio(input_path: str) -> str:
    """Convert audio to 16kHz mono WAV with ffmpeg, pad short audio."""
    output_path = input_path + ".16k.wav"
    try:
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

        # Pad short audio (< 1s) with silence to 2 seconds
        filters = "aresample=16000"
        if duration < 1.0:
            pad_ms = int((2.0 - duration) * 1000)
            filters = f"aresample=16000,apad=pad_dur={pad_ms}ms"

        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-af", filters,
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", output_path],
            capture_output=True, timeout=30
        )
        return output_path
    except Exception as e:
        print(f"[!] ffmpeg preprocess failed: {e}")
        return input_path


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
):
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        f.flush()
        tmp_input = f.name

    preprocessed = None
    try:
        # Debug: save copy and log audio info
        import shutil, subprocess as sp
        debug_dir = "/tmp/voquill_debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"audio_{int(time.time())}{suffix}")
        shutil.copy2(tmp_input, debug_path)
        probe = sp.run(["ffprobe", "-v", "error", "-show_entries",
            "stream=sample_rate,channels,duration,codec_name",
            "-of", "json", tmp_input], capture_output=True, text=True, timeout=10)
        print(f"[DEBUG] Saved: {debug_path}, Info: {probe.stdout.strip()}")

        preprocessed = preprocess_audio(tmp_input)
        t0 = time.time()
        result = engine.transcribe(preprocessed, language=language, verbose=False)
        elapsed = time.time() - t0

        text = result.text or ""
        # Filter out error messages
        for pat in ERROR_PATTERNS:
            if pat in text:
                text = ""
                break
        text = text.strip()
    except Exception as e:
        print(f"[!] Transcription error: {e}")
        text = ""
        elapsed = 0
    finally:
        try:
            os.unlink(tmp_input)
        except OSError:
            pass
        if preprocessed and preprocessed != tmp_input:
            try:
                os.unlink(preprocessed)
            except OSError:
                pass

    return JSONResponse({"text": text})


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Fun-ASR-Nano-GGUF-CUDA"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "fun-asr-nano", "object": "model", "owned_by": "local"}],
    }


@app.get("/models")
async def list_models_alt():
    return {
        "object": "list",
        "data": [{"id": "fun-asr-nano", "object": "model", "owned_by": "local"}],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8104)
