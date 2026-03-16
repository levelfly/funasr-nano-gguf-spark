#!/bin/bash
# Fun-ASR-Nano-GGUF + CUDA 快速部署腳本（適用於 NVIDIA DGX Spark / GX10）
# 部署時間：約 10 分鐘
# 前提：已安裝 CUDA 13.0+、Python 3.12、git、cmake

set -e
echo "=== Fun-ASR-Nano-GGUF 快速部署 ==="

# 1. 建立虛擬環境
echo "[1/6] 建立 Python 虛擬環境..."
python3 -m venv ~/funasr-env
source ~/funasr-env/bin/activate

# 2. 安裝 onnxruntime-gpu（預編譯 wheel，免去 2-3 小時編譯）
echo "[2/6] 安裝 onnxruntime-gpu (預編譯 SM121 wheel)..."
pip install https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl
pip install nvidia-cudnn-cu12
pip install gguf pypinyin watchdog srt pydub rich fastapi uvicorn python-multipart

# 3. 編譯 llama.cpp（約 3 分鐘）
echo "[3/6] 編譯 llama.cpp..."
cd ~
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j$(nproc)
cd ~

# 4. 下載 GGUF 模型（約 2 分鐘）
echo "[4/6] 下載 Fun-ASR-Nano-GGUF 模型..."
mkdir -p ~/models/Fun-ASR-Nano-GGUF
cd /tmp
curl -L -o Fun-ASR-Nano-GGUF.zip "https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Fun-ASR-Nano-GGUF.zip"
unzip -o Fun-ASR-Nano-GGUF.zip -d ~/models/Fun-ASR-Nano-GGUF
# 修正雙層目錄
if [ -d ~/models/Fun-ASR-Nano-GGUF/Fun-ASR-Nano-GGUF ]; then
    mv ~/models/Fun-ASR-Nano-GGUF/Fun-ASR-Nano-GGUF/* ~/models/Fun-ASR-Nano-GGUF/
    rmdir ~/models/Fun-ASR-Nano-GGUF/Fun-ASR-Nano-GGUF
fi
rm /tmp/Fun-ASR-Nano-GGUF.zip

# 5. 下載 CapsWriter 推理程式碼
echo "[5/6] 下載推理程式碼..."
cd /tmp
git clone --depth 1 https://github.com/HaujetZhao/CapsWriter-Offline.git
cp -r CapsWriter-Offline/util/fun_asr_gguf ~/fun_asr_gguf

# 建立 llama.cpp .so 軟連結
mkdir -p ~/fun_asr_gguf/inference/bin
for f in ~/llama.cpp/build/bin/lib*.so; do
    ln -sf "$f" ~/fun_asr_gguf/inference/bin/
done

# 修改 encoder/ctc 使用 CUDA EP
sed -i "/providers = \['CPUExecutionProvider'\]/a\\        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():\n            providers.insert(0, 'CUDAExecutionProvider')" ~/fun_asr_gguf/inference/encoder.py
sed -i "/providers = \['CPUExecutionProvider'\]/a\\        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():\n            providers.insert(0, 'CUDAExecutionProvider')" ~/fun_asr_gguf/inference/ctc.py

# 修改 llama.py 啟用 GPU offload
sed -i "s/model_params = llama_model_default_params()/model_params = llama_model_default_params()\n    model_params.n_gpu_layers = 999/" ~/fun_asr_gguf/inference/llama.py

rm -rf /tmp/CapsWriter-Offline

# 6. 建立 server 腳本
echo "[6/6] 建立 FastAPI server..."
cat > ~/funasr_gguf_server.py << 'SERVEREOF'
import sys
import os
import time
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional

os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/funasr-env/lib/python3.12/site-packages/nvidia/cudnn/lib") + ":" + os.path.expanduser("~/funasr-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.insert(0, os.path.expanduser("~"))
from fun_asr_gguf.inference.asr_engine import create_asr_engine

app = FastAPI(title="Fun-ASR-Nano-GGUF API")
MODEL_DIR = os.path.expanduser("~/models/Fun-ASR-Nano-GGUF")

print("Loading Fun-ASR-Nano-GGUF engine (CUDA)...")
t0 = time.time()
engine = create_asr_engine(
    encoder_onnx_path=os.path.join(MODEL_DIR, "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"),
    ctc_onnx_path=os.path.join(MODEL_DIR, "Fun-ASR-Nano-CTC.int4.onnx"),
    decoder_gguf_path=os.path.join(MODEL_DIR, "Fun-ASR-Nano-Decoder.q5_k.gguf"),
    tokens_path=os.path.join(MODEL_DIR, "tokens.txt"),
    enable_ctc=False, dml_enable=False, vulkan_enable=False, verbose=True)
print(f"Engine loaded in {time.time()-t0:.1f}s")

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = Form("zh")):
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await file.read()); tmp = f.name
    try:
        t0 = time.time()
        result = engine.transcribe(tmp, language=language, verbose=False)
        elapsed = time.time() - t0
    finally:
        os.unlink(tmp)
    return JSONResponse({"text": result.text, "duration_seconds": elapsed, "model": "Fun-ASR-Nano-GGUF-CUDA"})

@app.get("/health")
async def health():
    return {"status": "ok", "model": "Fun-ASR-Nano-GGUF-CUDA"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8104)
SERVEREOF

echo ""
echo "=== 部署完成！==="
echo ""
echo "啟動指令：（使用 wrapper 腳本，自動設定 cuDNN 路徑）"
echo "  nohup ~/start_gguf_asr.sh > ~/funasr_gguf_server.log 2>&1 &"
echo ""
echo "測試："
echo "  curl -X POST http://localhost:8104/v1/audio/transcriptions -F 'file=@test.wav' -F 'language=zh'"
echo ""
echo "效能：每段音檔約 0.29 秒，比 PyTorch 版快 7 倍"

# 7. 建立啟動 wrapper 腳本（解決重開機後 LD_LIBRARY_PATH 遺失問題）
echo "[7/7] 建立啟動腳本..."
cat > ~/start_gguf_asr.sh << 'WRAPEOF'
#!/bin/bash
export LD_LIBRARY_PATH=/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
exec /home/jayter/funasr-env/bin/python /home/jayter/funasr_gguf_server.py
WRAPEOF
chmod +x ~/start_gguf_asr.sh
