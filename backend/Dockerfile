FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY backend/agent_app ./agent_app
COPY data ./data
COPY private-data ./private-data

ENV POWER_TRADING_ENV=production
ENV POWER_TRADING_CORS_ORIGINS=https://lukeooo11.github.io
ENV AGENT_ENV=production
ENV AGENT_PLATFORM_BASE_URL=https://power-trading-api-0902.onrender.com
ENV AGENT_CORS_ORIGINS=https://lukeooo11.github.io
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8002}"]
