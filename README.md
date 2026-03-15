# Fun-ASR-Nano-GGUF for DGX Spark

在 NVIDIA DGX Spark (GB10 Blackwell) 上部署 Fun-ASR-Nano 語音辨識，比 PyTorch 版快 **7 倍**。

Deploy Fun-ASR-Nano ASR on NVIDIA DGX Spark (GB10 Blackwell) with **7x speedup** over PyTorch.

## 效能對比 / Performance

| 配置 / Configuration | 20 段音檔 | 每段平均 | 加速比 |
|---------------------|----------|---------|--------|
| PyTorch (FunASR AutoModel) | 40.4s | 2.02s | 1.0x |
| ONNX+GGUF (CPU encoder) | 9.23s | 0.46s | 4.4x |
| **ONNX+GGUF (CUDA encoder)** | **5.75s** | **0.29s** | **7.0x** |

## 原理 / How It Works

將 Fun-ASR-Nano 從 PyTorch 整體推理拆成三個組件，各用最佳推理引擎：

Split Fun-ASR-Nano into 3 components, each using the optimal inference engine:

```
音檔 Audio
  -> Mel 特徵提取 (NumPy, 無依賴 no dependency)
  -> ONNX Encoder (int4 量化, onnxruntime-gpu CUDAExecutionProvider)
  -> GGUF Decoder (q5_k 量化, llama.cpp via ctypes)
  -> 文字 Text
```

| 組件 / Component | 格式 / Format | 加速方式 / Acceleration |
|------------------|--------------|----------------------|
| Encoder + Adaptor | ONNX int4 | onnxruntime-gpu CUDA EP |
| CTC 熱詞 / Hotword | ONNX int4 | onnxruntime-gpu CUDA EP |
| LLM Decoder (Qwen3-0.6B) | GGUF q5_k | llama.cpp (GPU offload) |

### 相比 PyTorch 版的優勢 / Advantages over PyTorch

- **去掉 PyTorch 依賴** -> 模型載入從 10 秒降到 1.2 秒
- **記憶體佔用** -> 從數 GB 降到約 500MB
- **No PyTorch dependency** -> model loading from 10s to 1.2s
- **Memory usage** -> from several GB to ~500MB

## 快速部署 / Quick Deploy (~10 分鐘 / minutes)

### 前提 / Prerequisites

- NVIDIA DGX Spark (GB10) 或 Jetson Thor
- CUDA 13.0+, Python 3.12, git, cmake
- 網路連線（下載模型和套件）

### 一鍵部署 / One-click Deploy

```bash
git clone https://github.com/levelfly/funasr-nano-gguf-spark.git
cd funasr-nano-gguf-spark
bash deploy.sh
```

### 手動部署 / Manual Deploy

#### Step 1: 建立虛擬環境 / Create venv

```bash
python3 -m venv ~/funasr-env
source ~/funasr-env/bin/activate
```

#### Step 2: 安裝 onnxruntime-gpu / Install onnxruntime-gpu

預編譯的 SM121 wheel，免去 2-3 小時的編譯時間：

Pre-built SM121 wheel, skip 2-3 hours of compilation:

```bash
pip install https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl
pip install nvidia-cudnn-cu12
pip install gguf pypinyin watchdog srt pydub rich fastapi uvicorn python-multipart
```

#### Step 3: 編譯 llama.cpp / Build llama.cpp (~3 分鐘)

```bash
cd ~
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j$(nproc)
```

#### Step 4: 下載模型 / Download models (~2 分鐘)

```bash
mkdir -p ~/models/Fun-ASR-Nano-GGUF
cd /tmp
curl -L -o Fun-ASR-Nano-GGUF.zip \
  "https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Fun-ASR-Nano-GGUF.zip"
unzip -o Fun-ASR-Nano-GGUF.zip -d ~/models/Fun-ASR-Nano-GGUF
mv ~/models/Fun-ASR-Nano-GGUF/Fun-ASR-Nano-GGUF/* ~/models/Fun-ASR-Nano-GGUF/ 2>/dev/null
rmdir ~/models/Fun-ASR-Nano-GGUF/Fun-ASR-Nano-GGUF 2>/dev/null
rm /tmp/Fun-ASR-Nano-GGUF.zip
```

模型檔案 / Model files (567MB total):
- `Fun-ASR-Nano-Encoder-Adaptor.int4.onnx` (122MB)
- `Fun-ASR-Nano-CTC.int4.onnx` (21MB)
- `Fun-ASR-Nano-Decoder.q5_k.gguf` (424MB)
- `tokens.txt`

#### Step 5: 下載推理程式碼 / Download inference code

```bash
cd /tmp
git clone --depth 1 https://github.com/HaujetZhao/CapsWriter-Offline.git
cp -r CapsWriter-Offline/util/fun_asr_gguf ~/fun_asr_gguf
rm -rf CapsWriter-Offline
```

建立 llama.cpp 軟連結 / Create symlinks:
```bash
mkdir -p ~/fun_asr_gguf/inference/bin
for f in ~/llama.cpp/build/bin/lib*.so; do
    ln -sf "$f" ~/fun_asr_gguf/inference/bin/
done
```

#### Step 6: 修改為 CUDA 加速 / Enable CUDA acceleration

在 `~/fun_asr_gguf/inference/encoder.py` 和 `~/fun_asr_gguf/inference/ctc.py` 中，找到：

```python
providers = ['CPUExecutionProvider']
```

在下方加入：

```python
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    providers.insert(0, 'CUDAExecutionProvider')
```

在 `~/fun_asr_gguf/inference/llama.py` 中，找到：

```python
model_params = llama_model_default_params()
```

在下方加入：

```python
model_params.n_gpu_layers = 999  # offload all layers to GPU
```

## 啟動 Server / Start Server

```bash
export LD_LIBRARY_PATH=~/funasr-env/lib/python3.12/site-packages/nvidia/cudnn/lib:~/funasr-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

nohup ~/funasr-env/bin/python ~/funasr_gguf_server.py > ~/funasr-gguf-server.log 2>&1 &
```

## API 使用 / API Usage

```bash
# 辨識音檔 / Transcribe audio
curl -X POST http://localhost:8104/v1/audio/transcriptions \
  -F "file=@your_audio.wav" \
  -F "language=zh"

# 健康檢查 / Health check
curl http://localhost:8104/health
```

### Python 範例 / Python example

```python
import requests

resp = requests.post("http://localhost:8104/v1/audio/transcriptions",
    files={"file": open("audio.wav", "rb")},
    data={"language": "zh"})
print(resp.json()["text"])
```

## 致謝 / Credits

- [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) by HaujetZhao — ONNX+GGUF 拆分架構的原創者 / Original ONNX+GGUF split architecture
- [FunAudioLLM/Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) — 原始 ASR 模型 / Original ASR model
- [spark-docs](https://github.com/thewh1teagle/spark-docs) — DGX Spark onnxruntime 編譯指南 / Build guide
- [onnxruntime-gpu wheel](https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121) — 預編譯的 SM121 aarch64 CUDA 13 wheel / Pre-built wheel

## 授權 / License

MIT
