from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str
    environment: str
    host: str
    port: int
    database_path: Path
    retention_days: int
    platform_base_url: str
    platform_api_key: str
    platform_timeout_seconds: float
    platform_poll_interval_seconds: float
    platform_run_timeout_seconds: int
    model_id: str
    model_version: str
    spread_attention_threshold: float
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: float
    cors_origins: tuple[str, ...]
    debug: bool

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    database_path = Path(
        os.getenv(
            "AGENT_DATABASE_PATH",
            os.getenv("AGENT_DB_PATH", str(project_root / "data" / "agent.sqlite3")),
        )
    ).expanduser().resolve()
    raw_origins = os.getenv(
        "AGENT_CORS_ORIGINS",
        os.getenv(
            "AGENT_ALLOWED_ORIGINS",
            "http://127.0.0.1:8000,http://localhost:8000",
        ),
    )
    return Settings(
        service_name="power-trading-ai-agent",
        environment=os.getenv("AGENT_ENV", "local"),
        host=os.getenv("AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_PORT", "8010")),
        database_path=database_path,
        retention_days=max(
            1,
            int(
                os.getenv(
                    "AGENT_RUN_RETENTION_DAYS",
                    os.getenv("AGENT_RETENTION_DAYS", "180"),
                )
            ),
        ),
        platform_base_url=os.getenv(
            "AGENT_PLATFORM_BASE_URL",
            os.getenv("PLATFORM_API_BASE_URL", "http://127.0.0.1:8002"),
        ).rstrip("/"),
        platform_api_key=os.getenv(
            "AGENT_PLATFORM_API_KEY",
            os.getenv(
                "PLATFORM_API_KEY", os.getenv("POWER_TRADING_API_KEY", "")
            ),
        ),
        platform_timeout_seconds=max(
            1.0,
            float(
                os.getenv(
                    "AGENT_PLATFORM_TIMEOUT_SECONDS",
                    os.getenv("PLATFORM_TIMEOUT_SECONDS", "180"),
                )
            ),
        ),
        platform_poll_interval_seconds=max(
            0.1, float(os.getenv("AGENT_PLATFORM_POLL_INTERVAL_SECONDS", "1"))
        ),
        platform_run_timeout_seconds=max(
            10, min(600, int(os.getenv("AGENT_PLATFORM_RUN_TIMEOUT_SECONDS", "180")))
        ),
        model_id="price-forecast",
        model_version=os.getenv("AGENT_MODEL_VERSION", "price-forecast-v1.2.0"),
        spread_attention_threshold=max(
            0.0, float(os.getenv("AGENT_SPREAD_ATTENTION_THRESHOLD", "45"))
        ),
        llm_base_url=os.getenv("AGENT_LLM_BASE_URL", "").rstrip("/"),
        llm_api_key=os.getenv("AGENT_LLM_API_KEY", ""),
        llm_model=os.getenv("AGENT_LLM_MODEL", ""),
        llm_timeout_seconds=max(
            1.0, float(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "30"))
        ),
        cors_origins=tuple(
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ),
        debug=_env_bool("AGENT_DEBUG", False),
    )


settings = get_settings()
