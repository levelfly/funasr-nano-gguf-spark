FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv git ca-certificates libgomp1     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python venv + dependencies
RUN python3 -m venv /app/venv
ENV PATH=/app/venv/bin:$PATH
RUN pip install --no-cache-dir     https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl     nvidia-cudnn-cu13 nvidia-cuda-runtime nvidia-cublas     gguf pypinyin watchdog srt pydub rich fastapi uvicorn python-multipart

# cuDNN/CUDA library path
ENV LD_LIBRARY_PATH=/app/bin:/app/fun_asr_gguf/inference/bin:/app/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/app/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# Pre-compiled llama.cpp .so (SM121) + model files (from host)
COPY llama-libs/ /app/bin/
COPY models/ /app/models/

# Inference code from CapsWriter-Offline
RUN git clone --depth 1 https://github.com/HaujetZhao/CapsWriter-Offline.git /tmp/caps     && cp -r /tmp/caps/util/fun_asr_gguf /app/fun_asr_gguf     && rm -rf /tmp/caps

# Setup llama.cpp symlinks
RUN mkdir -p /app/fun_asr_gguf/inference/bin     && for f in /app/bin/lib*.so; do ln -sf "$f" /app/fun_asr_gguf/inference/bin/; done

# Patch: CUDA EP + GPU offload
RUN sed -i "/providers = \['CPUExecutionProvider'\]/a\        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():\n            providers.insert(0, 'CUDAExecutionProvider')" /app/fun_asr_gguf/inference/encoder.py     && sed -i "/providers = \['CPUExecutionProvider'\]/a\        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():\n            providers.insert(0, 'CUDAExecutionProvider')" /app/fun_asr_gguf/inference/ctc.py     && sed -i "s/model_params = llama_model_default_params()/model_params = llama_model_default_params()\n    model_params.n_gpu_layers = 999/" /app/fun_asr_gguf/inference/llama.py

# Server script
COPY funasr_gguf_server.py /app/funasr_gguf_server.py
RUN sed -i 's|/home/jayter/funasr-env/lib|/app/venv/lib|g' /app/funasr_gguf_server.py     && sed -i 's|/home/jayter/models/Fun-ASR-Nano-GGUF|/app/models|g' /app/funasr_gguf_server.py     && sed -i 's|sys.path.insert(0, os.path.expanduser("~"))|sys.path.insert(0, "/app")|g' /app/funasr_gguf_server.py     && sed -i 's|sys.path.insert(0, "/home/jayter")|sys.path.insert(0, "/app")|g' /app/funasr_gguf_server.py

EXPOSE 8104
CMD ["python", "/app/funasr_gguf_server.py"]
