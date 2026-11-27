FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk libglib2.0-0 libgl1 gosu ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', '/app/face_landmarker.task')"
RUN useradd -m appuser && mkdir -p /data && chown -R appuser:appuser /app /data
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
ENV SMILE_DATA_DIR=/data MAX_ACTIVE_SESSIONS=3
EXPOSE 8000
CMD ["python", "server.py"]
