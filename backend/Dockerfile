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
COPY index.html ./index.html
COPY vendor ./vendor

# 电源滞后来源策略：允许 ACTUAL 作为最低优先级来源。
# 现状只有完整的实际出力、缺少预测出力，属权衡后的临时口径；
# 输出会带 lag_leakage_risk 标记。补齐预测出力后应关闭此项。
ENV POWER_TRADING_ALLOW_ACTUAL_LAG=1
ENV POWER_TRADING_ENV=production
ENV POWER_TRADING_CORS_ORIGINS=https://lukeooo11.github.io
ENV AGENT_ENV=production
ENV AGENT_PLATFORM_BASE_URL=https://power-trading-api-0902.onrender.com
ENV AGENT_CORS_ORIGINS=https://lukeooo11.github.io
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8002}"]
