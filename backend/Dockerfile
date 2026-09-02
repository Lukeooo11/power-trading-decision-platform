FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/data ./data
COPY private-data ./private-data

ENV POWER_TRADING_ENV=production
ENV POWER_TRADING_CORS_ORIGINS=https://lukeooo11.github.io
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8002}"]
