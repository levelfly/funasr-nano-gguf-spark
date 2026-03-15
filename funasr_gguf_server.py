import sys
import os
import time
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

# cuDNN path for CUDA EP
os.environ["LD_LIBRARY_PATH"] = "/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, "/home/jayter")
from fun_asr_gguf.inference.asr_engine import create_asr_engine

app = FastAPI(title="Fun-ASR-Nano-GGUF API")

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


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form("zh"),
):
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        f.flush()
        tmp_input = f.name

    try:
        t0 = time.time()
        result = engine.transcribe(tmp_input, language=language, verbose=False)
        elapsed = time.time() - t0
    finally:
        os.unlink(tmp_input)

    return JSONResponse({
        "text": result.text,
        "duration_seconds": elapsed,
        "model": "Fun-ASR-Nano-GGUF-CUDA",
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Fun-ASR-Nano-GGUF-CUDA"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8104)
