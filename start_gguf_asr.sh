#!/bin/bash
export LD_LIBRARY_PATH=/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/jayter/funasr-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
exec /home/jayter/funasr-env/bin/python /home/jayter/funasr_gguf_server.py
