FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY src/data_pipeline/streaming/producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/data_pipeline/streaming/producer /app/src/data_pipeline/streaming/producer

# Set PYTHONPATH để Python tìm 'src'
ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "src.data_pipeline.streaming.producer.main:app", "--host", "0.0.0.0", "--port", "8002"]
