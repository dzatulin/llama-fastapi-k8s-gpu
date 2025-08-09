FROM myregistry/llama-cpp-cuda-base:0.1.5

COPY ./docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api.py /app/
COPY data /app/data/
RUN mkdir -p /app/models

WORKDIR /app

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--bind", "0.0.0.0:8000"]
