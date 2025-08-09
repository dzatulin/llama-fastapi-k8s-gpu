# llama-fastapi-k8s-gpu

Local LLM chat service on GPU (CUDA) using **llama.cpp + FastAPI**, packaged for **Kubernetes** with S3 model bootstrap, health checks, and back-pressure.

## Why

- **Cost & privacy:** keep tokens and data in your infra.
- **Predictable latency:** run close to your app, no external quotas.
- **MLOps controls:** models, probes, metrics, alerts.

## Features

- GPU-accelerated **llama.cpp** via `llama-cpp-python` (cuBLAS).
- **FastAPI** service with `/response`, `/health`, OpenAPI docs at `/docs`.
- **Context control**: rough token estimation and truncation.
- **Helm-ready**: initContainer pulls GGUF model from S3 to a mounted volume.

## Stack

- CUDA 12.2 base image, `llama-cpp-python==0.2.77` (cuBLAS wheel)
- FastAPI, Uvicorn, Gunicorn
- Kubernetes + NVIDIA device plugin
