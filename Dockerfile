FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv ca-certificates ffmpeg libgomp1     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /app/venv
ENV PATH=/app/venv/bin:$PATH
RUN pip install --no-cache-dir     https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl     nvidia-cudnn-cu13 nvidia-cuda-runtime nvidia-cublas     gguf pypinyin watchdog srt pydub rich fastapi uvicorn python-multipart

ENV LD_LIBRARY_PATH=/app/bin:/app/fun_asr_gguf/inference/bin:/app/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/app/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

COPY llama-libs/ /app/bin/
COPY models/ /app/models/
COPY fun_asr_gguf/ /app/fun_asr_gguf/

# Setup llama.cpp symlinks in inference/bin
RUN mkdir -p /app/fun_asr_gguf/inference/bin     && for f in /app/bin/lib*.so; do ln -sf "$f" /app/fun_asr_gguf/inference/bin/; done

COPY funasr_gguf_server.py /app/funasr_gguf_server.py
RUN sed -i 's|/home/jayter/funasr-env/lib|/app/venv/lib|g' /app/funasr_gguf_server.py     && sed -i 's|/home/jayter/models/Fun-ASR-Nano-GGUF|/app/models|g' /app/funasr_gguf_server.py     && sed -i 's|sys.path.insert(0, os.path.expanduser("~"))|sys.path.insert(0, "/app")|g' /app/funasr_gguf_server.py     && sed -i 's|sys.path.insert(0, "/home/jayter")|sys.path.insert(0, "/app")|g' /app/funasr_gguf_server.py

EXPOSE 8104
CMD ["python", "/app/funasr_gguf_server.py"]
