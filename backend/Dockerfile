FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/agent_app ./agent_app
COPY backend/model_assets ./model_assets
COPY data ./data
COPY private-data/henan-2026 ./private-data/henan-2026

ENV POWER_TRADING_ENV=production
ENV POWER_TRADING_CORS_ORIGINS=https://lukeooo11.github.io
ENV AGENT_ENV=production
ENV AGENT_PLATFORM_BASE_URL=https://power-trading-api-0902.onrender.com
ENV AGENT_CORS_ORIGINS=https://lukeooo11.github.io
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8002}"]
