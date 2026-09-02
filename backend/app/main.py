from __future__ import annotations

import json
import math
import os
import re
import secrets
import sqlite3
import uuid
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .policy_agent_client import PolicyAgentClient, PolicyAgentError
from .price_forecast_model import run_price_forecast


# Local runs place this file under ``<repo>/backend/app`` while the Docker
# image copies the package to ``/app/app``. Resolve the data root for both.
_FILE_ROOT = Path(__file__).resolve()
ROOT = _FILE_ROOT.parents[1] if (_FILE_ROOT.parents[1] / "private-data").exists() else _FILE_ROOT.parents[2]
CUSTOMER_DATA = ROOT / "customer-data" / "weifang-caixin"
PRIVATE_DATA = ROOT / "private-data" / "shandong-2026h1"
DB_PATH = ROOT / "backend" / "data" / "platform.db"
SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
API_VERSION = "1.0.0"
SERVICE_ENVIRONMENT = os.getenv("POWER_TRADING_ENV", "test")
V1_API_KEY = os.getenv("POWER_TRADING_API_KEY", "")
MAX_PRICE_UPLOAD_RECORDS = 10_000
MAX_PRICE_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_RUN_TIMEOUT_SECONDS = 120
MAX_RUN_TIMEOUT_SECONDS = 600
RUN_RETENTION_DAYS = 180
BUILTIN_DATA_VERSION = "sd-hourly-weather-supply-snapshot-2026h1-v3"
REQUIRED_STRATEGY_DOMAINS = {
    "prices",
    "load_actual",
    "load_forecast",
    "weather",
    "renewable_forecast",
    "medium_long_term_positions",
    "retail_contracts",
    "day_ahead_declarations",
    "real_time_declarations",
    "clearing_results",
    "manual_adjustments",
    "deviation_assessment",
    "complete_settlement",
    "unit_status",
    "outages",
    "congestion",
    "trading_limits",
    "market_rules",
}

MARKETS = [
    ("BJ", "北京", "华北"), ("TJ", "天津", "华北"), ("HE", "河北", "华北"),
    ("SX", "山西", "华北"), ("NM", "内蒙古", "华北"), ("LN", "辽宁", "东北"),
    ("JL", "吉林", "东北"), ("HL", "黑龙江", "东北"), ("SH", "上海", "华东"),
    ("JS", "江苏", "华东"), ("ZJ", "浙江", "华东"), ("AH", "安徽", "华东"),
    ("FJ", "福建", "华东"), ("JX", "江西", "华中"), ("SD", "山东", "华东"),
    ("HA", "河南", "华中"), ("HB", "湖北", "华中"), ("HN", "湖南", "华中"),
    ("GD", "广东", "南方"), ("GX", "广西", "南方"), ("HI", "海南", "南方"),
    ("CQ", "重庆", "西南"), ("SC", "四川", "西南"), ("GZ", "贵州", "南方"),
    ("YN", "云南", "南方"), ("XZ", "西藏", "西南"), ("SN", "陕西", "西北"),
    ("GS", "甘肃", "西北"), ("QH", "青海", "西北"), ("NX", "宁夏", "西北"),
    ("XJ", "新疆", "西北"),
]

ORG_ROLES = [
    {"code": "management", "name": "公司管理层", "scope": "全国经营概览、风险摘要和跨市场复盘"},
    {"code": "trader", "name": "交易员", "scope": "授权市场的盘前、现货、中长期和复盘"},
    {"code": "analyst", "name": "数据分析", "scope": "数据质量、模型版本、评估和数据血缘"},
    {"code": "operations", "name": "客户运营", "scope": "客户、合同、结算和经营诊断案例"},
]


class ForecastRunRequest(BaseModel):
    date: str
    model_version: str = "lag-baseline-v0.3"
    market_code: str = "SD"


class ForecastResultPoint(BaseModel):
    timePoint: str
    predictedPrice: float
    lower: float | None = None
    upper: float | None = None
    spreadDirection: str
    confidence: float
    drivers: list[str] = Field(default_factory=list)
    recommendation: str | None = None


class ForecastResultImportRequest(BaseModel):
    marketCode: str
    businessDate: str
    intervalMinutes: int
    modelVersion: str
    predictionBatchId: str
    featureBatchId: str
    sourceSystem: str = "agent-box"
    points: list[ForecastResultPoint]


class StrategyResultPoint(BaseModel):
    timePoint: str
    action: str
    suggestedEnergyMwh: float | None = None
    suggestedPriceYuanMwh: float | None = None
    confidence: float
    reason: str
    requiresManualReview: bool = True


class StrategyResultImportRequest(BaseModel):
    marketCode: str
    businessDate: str
    intervalMinutes: int
    strategyVersion: str
    strategyBatchId: str
    predictionBatchId: str
    sourceSystem: str = "agent-box"
    points: list[StrategyResultPoint]


class PolicyDocumentRegistration(BaseModel):
    documentId: str
    marketCode: str
    title: str
    version: str
    effectiveDate: str | None = None
    storageUri: str
    checksumSha256: str
    sourceAuthorized: bool = False


class PolicyAgentConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class PolicyAgentQueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    marketCode: str = Field(default="SD", min_length=2, max_length=8)
    marketHint: str = Field(default="零售市场", min_length=1, max_length=128)
    conversationHistory: list[PolicyAgentConversationTurn] = Field(default_factory=list, max_length=8)


class AdapterNormalizeRequest(BaseModel):
    sourceSystem: str
    marketCode: str
    dataDomain: str
    importBatchId: str
    fieldMapping: dict[str, str]
    records: list[dict[str, Any]]


class HourlyDataRecordV1(BaseModel):
    market_date: str
    period: int = Field(ge=1, le=24)
    datetime: str
    day_ahead_price_yuan_per_mwh: float | None = None
    real_time_price_yuan_per_mwh: float | None = None
    load_actual_mwh: float | None = Field(default=None, ge=0)
    load_forecast_mwh: float | None = Field(default=None, ge=0)
    temperature_2m_c: float | None = None
    wind_speed_10m_mps: float | None = Field(default=None, ge=0)
    wind_speed_100m_mps: float | None = Field(default=None, ge=0)
    shortwave_radiation_ghi_w_per_m2: float | None = Field(default=None, ge=0)
    cloud_cover_pct: float | None = Field(default=None, ge=0, le=100)
    precipitation_mm: float | None = Field(default=None, ge=0)
    relative_humidity_pct: float | None = Field(default=None, ge=0, le=100)
    weather_forecast_issue_time: str | None = None
    customer_id: str | None = None


class PriceAssetUploadRequestV1(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    market_code: str = Field(default="SD", min_length=2, max_length=8)
    data_version: str = Field(min_length=1, max_length=128)
    source_system: str = Field(min_length=1, max_length=128)
    updated_at: str
    records: list[HourlyDataRecordV1] = Field(min_length=1, max_length=MAX_PRICE_UPLOAD_RECORDS)


class ModelRunCreateRequestV1(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    market_code: str = Field(default="SD", min_length=2, max_length=8)
    market_date: str
    model_id: Literal["price-forecast", "lag-baseline"]
    model_version: str = Field(min_length=1, max_length=128)
    data_version: str = Field(default=BUILTIN_DATA_VERSION, min_length=1, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=DEFAULT_RUN_TIMEOUT_SECONDS, ge=10, le=MAX_RUN_TIMEOUT_SECONDS)


class PriceQuantilesV1(BaseModel):
    p10: float
    p50: float
    p90: float


class RiskSignalV1(BaseModel):
    probability: float = Field(ge=0, le=1)
    level: Literal["LOW", "MEDIUM", "HIGH"]


class StrategySuggestionV1(BaseModel):
    action: Literal["HOLD", "BUY", "SELL"] = "HOLD"
    volume_mwh: float = Field(default=0, ge=0)
    price_yuan_per_mwh: float | None = None
    confidence: float = Field(default=0, ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)


class ForecastPeriodV1(BaseModel):
    period: int = Field(ge=1, le=24)
    datetime: str
    day_ahead_price_yuan_per_mwh: PriceQuantilesV1
    real_time_price_yuan_per_mwh: PriceQuantilesV1
    spread_day_ahead_minus_real_time_yuan_per_mwh: float
    negative_price_risk: RiskSignalV1
    high_price_risk: RiskSignalV1
    risk_reason_codes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    data_completeness: Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"]
    strategy_suggestion: StrategySuggestionV1 = Field(default_factory=StrategySuggestionV1)


class ModelIdentityV1(BaseModel):
    id: str
    name: str
    version: str


class DataSnapshotV1(BaseModel):
    version: str
    created_at: str
    source_versions: dict[str, str] = Field(default_factory=dict)
    available_domains: list[str] = Field(default_factory=list)
    missing_domains: list[str] = Field(default_factory=list)


class BacktestMetricsV1(BaseModel):
    window_start: str | None = None
    window_end: str | None = None
    sample_count: int = Field(default=0, ge=0)
    mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    rmse_yuan_per_mwh: float | None = Field(default=None, ge=0)
    bias_yuan_per_mwh: float | None = None
    negative_price_direction_accuracy: float | None = Field(default=None, ge=0, le=1)
    extreme_high_price_recall: float | None = Field(default=None, ge=0, le=1)
    evaluation_method: str | None = None
    feature_set: str | None = None
    weather_data_version: str | None = None
    weather_known_before_declaration: bool | None = None
    weather_used_in_final_model: bool | None = None
    baseline_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    weather_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    weather_mae_improvement_yuan_per_mwh: float | None = None
    weather_mae_improvement_pct: float | None = None
    real_time_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    real_time_rmse_yuan_per_mwh: float | None = Field(default=None, ge=0)
    real_time_baseline_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    real_time_weather_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    real_time_weather_mae_improvement_pct: float | None = None
    day_ahead_model_name: str | None = None
    real_time_model_name: str | None = None
    supply_data_version: str | None = None
    supply_used_in_final_model: bool | None = None
    supply_issue_time_available: bool | None = None
    supply_backtest_leakage_safe: bool | None = None
    supply_usage_boundary: str | None = None
    day_ahead_supply_candidate_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    real_time_supply_candidate_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    day_ahead_final_improvement_vs_weather_pct: float | None = None
    spread_direction_accuracy: float | None = Field(default=None, ge=0, le=1)
    spread_mae_yuan_per_mwh: float | None = Field(default=None, ge=0)
    day_ahead_interval_coverage: float | None = Field(default=None, ge=0, le=1)
    real_time_interval_coverage: float | None = Field(default=None, ge=0, le=1)
    adaptive_ensemble: dict[str, Any] | None = None
    ensemble_weights: dict[str, Any] | None = None
    deep_challengers: dict[str, Any] | None = None
    consistency_constraint: str | None = None
    post_day_ahead_realtime: dict[str, Any] | None = None


class ForecastStrategyResultV1(BaseModel):
    schema_version: str = API_VERSION
    request_id: str
    run_id: str
    market_code: str
    market_date: str
    timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"
    model: ModelIdentityV1
    data_snapshot: DataSnapshotV1
    backtest: BacktestMetricsV1
    periods: list[ForecastPeriodV1]
    strategy_ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class RunActionRequestV1(BaseModel):
    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)


class RerunRequestV1(RunActionRequestV1):
    request_id: str = Field(min_length=1, max_length=128)


class StrategyReviewRequestV1(BaseModel):
    action: Literal["SUBMIT", "APPROVE", "MODIFY", "REJECT"]
    reviewer: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=1000)
    modified_suggestions: list[dict[str, Any]] | None = None


CANONICAL_REQUIRED_FIELDS = {
    "spotPrice": ["businessDate", "timePoint", "dayAheadPrice", "realtimePrice"],
    "customerLoad": ["businessDate", "timePoint", "loadMwh"],
    "tradeExecution": ["businessDate", "timePoint", "declaredEnergy", "clearedEnergy"],
    "mediumPosition": ["contractMonth", "contractedEnergy", "contractPrice"],
}

INTEGRATION_CONTRACT_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shanghai_timestamp(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_TIMESTAMP", "message": f"Invalid RFC3339 timestamp: {value}"}) from error
    if parsed.tzinfo is None:
        raise HTTPException(status_code=422, detail={"code": "TIMEZONE_REQUIRED", "message": f"Timestamp must include a timezone: {value}"})
    return parsed.astimezone(SHANGHAI_TZ).isoformat()


def period_start_timestamp(market_date: str, period: int) -> str:
    validate_business_date(market_date)
    start = datetime.strptime(market_date, "%Y-%m-%d").replace(tzinfo=SHANGHAI_TZ)
    return (start + timedelta(hours=period - 1)).isoformat()


def api_error(status_code: int, code: str, message: str, **details: Any) -> HTTPException:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return HTTPException(status_code=status_code, detail=payload)


def require_v1_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if V1_API_KEY and (x_api_key is None or not secrets.compare_digest(x_api_key, V1_API_KEY)):
        raise api_error(401, "UNAUTHORIZED", "A valid X-API-Key header is required")


def validate_customer_alias(value: str | None) -> None:
    if value is not None and not re.fullmatch(r"客户-\d{3,}", value):
        raise api_error(422, "INVALID_CUSTOMER_ID", "customer_id must use the stable alias format 客户-001")


def validate_hourly_timestamp(market_date: str, period: int, value: str) -> None:
    normalized = shanghai_timestamp(value)
    if normalized != period_start_timestamp(market_date, period):
        raise api_error(
            422,
            "PERIOD_TIMESTAMP_MISMATCH",
            "datetime must be the Asia/Shanghai start timestamp for the specified market_date and period",
            expected=period_start_timestamp(market_date, period),
            received=normalized,
        )


def strategy_readiness(input_summary: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    available = {str(item) for item in input_summary.get("available_domains", [])}
    missing = sorted(REQUIRED_STRATEGY_DOMAINS - available)
    return not missing, sorted(available), missing


def validate_quantiles(quantiles: PriceQuantilesV1, label: str, period: int) -> None:
    if not quantiles.p10 <= quantiles.p50 <= quantiles.p90:
        raise api_error(422, "INVALID_QUANTILE_ORDER", f"{label} must satisfy p10 <= p50 <= p90", period=period)


def validate_forecast_result(result: ForecastStrategyResultV1, expected: sqlite3.Row | None = None) -> None:
    validate_business_date(result.market_date)
    if len(result.periods) != 24 or {point.period for point in result.periods} != set(range(1, 25)):
        raise api_error(422, "INCOMPLETE_PERIODS", "Forecast output must contain exactly one point for each period 1 through 24")
    for point in result.periods:
        validate_hourly_timestamp(result.market_date, point.period, point.datetime)
        validate_quantiles(point.day_ahead_price_yuan_per_mwh, "day_ahead_price_yuan_per_mwh", point.period)
        validate_quantiles(point.real_time_price_yuan_per_mwh, "real_time_price_yuan_per_mwh", point.period)
        expected_spread = point.day_ahead_price_yuan_per_mwh.p50 - point.real_time_price_yuan_per_mwh.p50
        if abs(expected_spread - point.spread_day_ahead_minus_real_time_yuan_per_mwh) > 0.01:
            raise api_error(422, "SPREAD_MISMATCH", "Spread must equal day-ahead P50 minus real-time P50", period=point.period)
        if not result.strategy_ready and point.strategy_suggestion.action != "HOLD":
            raise api_error(409, "STRATEGY_INPUTS_INCOMPLETE", "Incomplete strategy inputs require HOLD for every period", period=point.period)
    if expected is not None:
        checks = {
            "request_id": (result.request_id, expected["request_id"]),
            "run_id": (result.run_id, expected["run_id"]),
            "market_code": (result.market_code.upper(), expected["market_code"]),
            "market_date": (result.market_date, expected["market_date"]),
            "model.id": (result.model.id, expected["model_id"]),
            "model.version": (result.model.version, expected["model_version"]),
            "data_snapshot.version": (result.data_snapshot.version, expected["data_version"]),
        }
        mismatches = {key: {"received": values[0], "expected": values[1]} for key, values in checks.items() if values[0] != values[1]}
        if mismatches:
            raise api_error(409, "RUN_CONTRACT_MISMATCH", "Result identity does not match the stored run", mismatches=mismatches)


def shandong_data_available() -> bool:
    return all((PRIVATE_DATA / name).exists() for name in [
        "spot_prices_2026h1.json",
        "portfolio_load_hourly_2026h1.json",
        "data_quality_2026h1.json",
    ])


def load_private_json(name: str) -> Any:
    path = PRIVATE_DATA / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Private data product not found: {name}. Run scripts/process-shandong-data.cjs first.")
    return json.loads(path.read_text(encoding="utf-8"))


def contract_position_assets() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_private_json("retail_contract_packages_2026h1.json"),
        load_private_json("medium_long_term_positions_2026h1.json"),
    )


def position_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_rows = [
        row for row in rows
        if isinstance(row.get("positionMwh"), (int, float))
    ]
    positive_priced = [
        row for row in rows
        if isinstance(row.get("positionMwh"), (int, float))
        and row["positionMwh"] > 0
        and isinstance(row.get("priceYuanMwh"), (int, float))
        and row["priceYuanMwh"] != 0
    ]
    positive_priced_mwh = sum(float(row["positionMwh"]) for row in positive_priced)
    positive_cost = sum(float(row["positionMwh"]) * float(row["priceYuanMwh"]) for row in positive_priced)
    return {
        "rowCount": len(rows),
        "netPositionMwh": sum(float(row["positionMwh"]) for row in numeric_rows),
        "positivePositionMwh": sum(float(row["positionMwh"]) for row in numeric_rows if row["positionMwh"] > 0),
        "negativePositionMwh": sum(float(row["positionMwh"]) for row in numeric_rows if row["positionMwh"] < 0),
        "positivePricedMwh": positive_priced_mwh,
        "positivePositionCostYuan": positive_cost,
        "positiveWeightedPriceYuanMwh": positive_cost / positive_priced_mwh if positive_priced_mwh else None,
        "zeroPriceAbsMwh": sum(abs(float(row["positionMwh"])) for row in numeric_rows if row.get("priceYuanMwh") == 0),
        "anomalyCount": sum(bool(row.get("qualityFlags")) for row in rows),
    }


def position_trade_type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trade_types = sorted({str(row.get("tradeType")) for row in rows if row.get("tradeType")})
    return [dict(position_summary([row for row in rows if row.get("tradeType") == trade_type]), tradeType=trade_type) for trade_type in trade_types]


def portfolio_load_for_date(date: str) -> float | None:
    row = next((item for item in load_private_json("portfolio_load_daily_2026h1.json") if item.get("date") == date), None)
    value = row.get("totalMwh") if row else None
    return float(value) if isinstance(value, (int, float)) else None


def volume_weighted_spot_prices(date: str) -> dict[str, Any]:
    loads = {
        row["time"]: float(row["totalMwh"])
        for row in load_private_json("portfolio_load_hourly_2026h1.json")
        if row.get("date") == date and isinstance(row.get("totalMwh"), (int, float))
    }
    prices = [row for row in load_private_json("spot_prices_2026h1.json") if row.get("date") == date]
    def weighted(field: str) -> float | None:
        pairs = [
            (loads[row["time"]], float(row[field]))
            for row in prices
            if row.get("time") in loads and isinstance(row.get(field), (int, float))
        ]
        volume = sum(item[0] for item in pairs)
        return sum(weight * price for weight, price in pairs) / volume if volume else None
    return {
        "dayAheadWeightedPriceYuanMwh": weighted("dayAheadPriceYuanMwh"),
        "realTimeWeightedPriceYuanMwh": weighted("realtimePriceYuanMwh"),
        "pricePointCount": len(prices),
        "loadPointCount": len(loads),
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def offset_date(value: str, days: int) -> str:
    try:
        base = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise HTTPException(status_code=422, detail="date must use YYYY-MM-DD") from error
    return (base + timedelta(days=days)).date().isoformat()


def validate_business_date(value: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise HTTPException(status_code=422, detail="businessDate must use YYYY-MM-DD") from error


def validate_interval(interval_minutes: int) -> None:
    if interval_minutes not in {15, 30, 60}:
        raise HTTPException(status_code=422, detail="intervalMinutes must be 15, 30 or 60")


def result_quality(time_points: list[str], interval_minutes: int) -> dict[str, Any]:
    validate_interval(interval_minutes)
    if not time_points:
        raise HTTPException(status_code=422, detail="points must not be empty")
    invalid = []
    for value in time_points:
        match = re.fullmatch(r"(\d{2}):(\d{2})", value)
        if not match:
            invalid.append(value)
            continue
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 24 or minute > 59 or (hour == 24 and minute != 0) or minute % interval_minutes != 0:
            invalid.append(value)
    if invalid:
        raise HTTPException(status_code=422, detail=f"Invalid timePoint values: {invalid[:5]}")
    duplicates = sorted({value for value in time_points if time_points.count(value) > 1})
    if duplicates:
        raise HTTPException(status_code=422, detail=f"Duplicate timePoint values: {duplicates[:5]}")
    expected_count = 1440 // interval_minutes
    return {
        "qualityStatus": "complete" if len(time_points) == expected_count else "partial",
        "pointCount": len(time_points),
        "expectedPointCount": expected_count,
        "usableForStrategy": len(time_points) == expected_count,
    }


def market_profile(market_code: str) -> dict[str, Any]:
    normalized = market_code.upper()
    market = next((row for row in MARKETS if row[0] == normalized), None)
    if market is None:
        raise HTTPException(status_code=404, detail=f"Unknown market code: {market_code}")
    code, name, region = market
    sample = code == "SD"
    connected = sample and shandong_data_available()
    return {
        "code": code,
        "name": name,
        "region": region,
        "stage": "sample" if sample else "template",
        "rules": "规则待补" if sample else "待配置",
        "data": "价格/组合负荷已接入" if connected else "待接入",
        "model": "历史回测基线" if connected else "待训练",
        "api": "本地数据联通" if connected else "接口预留",
        "forecast_available": connected,
    }


def require_sample_market(market_code: str) -> dict[str, Any]:
    profile = market_profile(market_code)
    if not profile["forecast_available"]:
        raise HTTPException(
            status_code=409,
            detail=f"{profile['name']}市场尚未接入可用预测数据，不能使用山东数据代替。",
        )
    return profile


def load_json(name: str) -> Any:
    path = CUSTOMER_DATA / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with closing(connect()) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS forecast_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_code TEXT NOT NULL DEFAULT 'SD',
                forecast_date TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source_file TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_date
            ON forecast_runs(forecast_date, model_version, created_at);
            CREATE TABLE IF NOT EXISTS integration_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_type TEXT NOT NULL,
                market_code TEXT NOT NULL,
                business_date TEXT NOT NULL,
                version TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                source_system TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                quality_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(result_type, batch_id)
            );
            CREATE INDEX IF NOT EXISTS idx_integration_results_lookup
            ON integration_results(result_type, market_code, business_date, created_at);
            CREATE TABLE IF NOT EXISTS policy_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL UNIQUE,
                market_code TEXT NOT NULL,
                title TEXT NOT NULL,
                version TEXT NOT NULL,
                effective_date TEXT,
                storage_uri TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                source_authorized INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS data_asset_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                market_code TEXT NOT NULL,
                data_version TEXT NOT NULL,
                source_system TEXT NOT NULL,
                request_id TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                quality_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(domain, market_code, data_version)
            );
            CREATE INDEX IF NOT EXISTS idx_data_asset_batches_lookup
            ON data_asset_batches(domain, market_code, created_at);
            CREATE TABLE IF NOT EXISTS model_runs_v1 (
                run_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                parent_run_id TEXT,
                market_code TEXT NOT NULL,
                market_date TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                data_version TEXT NOT NULL,
                status TEXT NOT NULL,
                review_status TEXT NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                request_json TEXT NOT NULL,
                input_snapshot_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                input_summary_json TEXT NOT NULL,
                result_json TEXT,
                current_strategy_json TEXT,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_model_runs_v1_lookup
            ON model_runs_v1(market_code, market_date, model_id, created_at);
            CREATE TABLE IF NOT EXISTS run_audit_logs_v1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_run_audit_logs_v1_run
            ON run_audit_logs_v1(run_id, created_at);
            CREATE TABLE IF NOT EXISTS strategy_reviews_v1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                reason TEXT,
                original_json TEXT,
                modified_json TEXT,
                model_version TEXT NOT NULL,
                data_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_strategy_reviews_v1_run
            ON strategy_reviews_v1(run_id, created_at);
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(forecast_runs)")}
        if "market_code" not in columns:
            connection.execute("ALTER TABLE forecast_runs ADD COLUMN market_code TEXT NOT NULL DEFAULT 'SD'")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecast_runs_market "
            "ON forecast_runs(market_code, forecast_date, model_version, created_at)"
        )


def write_run_audit(run_id: str, event_type: str, actor: str, details: dict[str, Any] | None = None) -> None:
    with closing(connect()) as connection, connection:
        connection.execute(
            "INSERT INTO run_audit_logs_v1 (run_id, event_type, actor, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, event_type, actor, json.dumps(details or {}, ensure_ascii=False), utc_now()),
        )


def get_model_run_row(run_id: str) -> sqlite3.Row:
    with closing(connect()) as connection, connection:
        row = connection.execute("SELECT * FROM model_runs_v1 WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise api_error(404, "RUN_NOT_FOUND", f"Unknown run_id: {run_id}")
    return row


def refresh_run_timeout(run_id: str) -> sqlite3.Row:
    row = get_model_run_row(run_id)
    if row["status"] not in {"QUEUED", "RUNNING", "PENDING_PROVIDER"}:
        return row
    created_at = datetime.fromisoformat(row["created_at"])
    if datetime.now(timezone.utc) <= created_at + timedelta(seconds=row["timeout_seconds"]):
        return row
    now = utc_now()
    with closing(connect()) as connection, connection:
        connection.execute(
            """
            UPDATE model_runs_v1
            SET status = 'TIMED_OUT', error_code = 'RUN_TIMEOUT', error_message = ?, completed_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (f"Run exceeded {row['timeout_seconds']} seconds", now, now, run_id),
        )
    write_run_audit(run_id, "RUN_TIMED_OUT", "platform", {"timeout_seconds": row["timeout_seconds"]})
    return get_model_run_row(run_id)


def public_model_run(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "run_id": row["run_id"],
        "parent_run_id": row["parent_run_id"],
        "market_code": row["market_code"],
        "market_date": row["market_date"],
        "model_id": row["model_id"],
        "model_version": row["model_version"],
        "data_version": row["data_version"],
        "status": row["status"],
        "review_status": row["review_status"],
        "timeout_seconds": row["timeout_seconds"],
        "input_summary": json.loads(row["input_summary_json"]),
        "error": {"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None,
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "updated_at": row["updated_at"],
        "result_url": f"/api/v1/model-runs/{row['run_id']}/results",
        "review_url": f"/api/v1/model-runs/{row['run_id']}/review",
        "execution_allowed": False,
    }


def purge_expired_runs() -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RUN_RETENTION_DAYS)).isoformat()
    with closing(connect()) as connection, connection:
        run_ids = [row[0] for row in connection.execute("SELECT run_id FROM model_runs_v1 WHERE created_at < ?", (cutoff,)).fetchall()]
        if not run_ids:
            return
        placeholders = ",".join("?" for _ in run_ids)
        connection.execute(f"DELETE FROM strategy_reviews_v1 WHERE run_id IN ({placeholders})", run_ids)
        connection.execute(f"DELETE FROM run_audit_logs_v1 WHERE run_id IN ({placeholders})", run_ids)
        connection.execute(f"DELETE FROM model_runs_v1 WHERE run_id IN ({placeholders})", run_ids)


def latest_uploaded_price_asset(market_code: str, market_date: str, data_version: str | None = None) -> dict[str, Any] | None:
    clauses = ["domain = 'prices'", "market_code = ?"]
    parameters: list[Any] = [market_code]
    if data_version:
        clauses.append("data_version = ?")
        parameters.append(data_version)
    with closing(connect()) as connection, connection:
        rows = connection.execute(
            f"SELECT * FROM data_asset_batches WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
            parameters,
        ).fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"])
        matching = [record for record in payload.get("records", []) if record.get("market_date") == market_date]
        if matching:
            return {
                "data_version": row["data_version"],
                "updated_at": row["updated_at"],
                "source_system": row["source_system"],
                "records": matching,
            }
    return None


def store_integration_result(
    result_type: str,
    market_code: str,
    business_date: str,
    version: str,
    batch_id: str,
    source_system: str,
    quality: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[int, str]:
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO integration_results
                (result_type, market_code, business_date, version, batch_id, source_system,
                 point_count, quality_status, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(result_type, batch_id) DO UPDATE SET
                market_code = excluded.market_code,
                business_date = excluded.business_date,
                version = excluded.version,
                source_system = excluded.source_system,
                point_count = excluded.point_count,
                quality_status = excluded.quality_status,
                payload_json = excluded.payload_json,
                created_at = excluded.created_at
            """,
            (
                result_type,
                market_code,
                business_date,
                version,
                batch_id,
                source_system,
                quality["pointCount"],
                quality["qualityStatus"],
                json.dumps(payload, ensure_ascii=False),
                created_at,
            ),
        )
        row = connection.execute(
            "SELECT id FROM integration_results WHERE result_type = ? AND batch_id = ?",
            (result_type, batch_id),
        ).fetchone()
    return int(row["id"]), created_at


def build_forecast(date: str, model_version: str, market_code: str = "SD") -> list[dict[str, Any]]:
    require_sample_market(market_code)
    prices = load_private_json("spot_prices_2026h1.json")
    loads = load_private_json("portfolio_load_hourly_2026h1.json")
    targets = sorted([row for row in prices if row.get("date") == date], key=lambda row: row["time"])
    if not targets:
        raise HTTPException(status_code=404, detail=f"No spot price rows for {date}")
    lag_one_date = offset_date(date, -1)
    lag_two_date = offset_date(date, -2)
    lag_one = {row["time"]: row for row in prices if row.get("date") == lag_one_date}
    lag_two = {row["time"]: row for row in prices if row.get("date") == lag_two_date}
    if not lag_one or not lag_two:
        raise HTTPException(status_code=409, detail="The lag baseline requires D-1 day-ahead and D-2 real-time price rows.")
    lag_one_load = {row["time"]: row for row in loads if row.get("date") == lag_one_date}
    results: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        time_point = target["time"]
        previous_day = lag_one.get(time_point)
        two_days_before = lag_two.get(time_point)
        if not previous_day or not two_days_before:
            continue
        historical_prices = [
            float(row["realtimePriceYuanMwh"])
            for row in prices
            if row.get("date", "") < date and row.get("time") == time_point and isinstance(row.get("realtimePriceYuanMwh"), (int, float))
        ][-14:]
        if not historical_prices:
            continue
        historical_mean = average(historical_prices)
        lag_two_rt = two_days_before.get("realtimePriceYuanMwh")
        lag_two_rt = float(lag_two_rt) if isinstance(lag_two_rt, (int, float)) else historical_mean
        day_ahead = float(previous_day["dayAheadPriceYuanMwh"])
        prior_loads = [
            float(row["totalMwh"])
            for row in loads
            if row.get("date", "") < lag_one_date and row.get("time") == time_point and isinstance(row.get("totalMwh"), (int, float))
        ][-7:]
        load_reference = average(prior_loads)
        lag_load_value = lag_one_load.get(time_point, {}).get("totalMwh")
        load_delta = (float(lag_load_value) - load_reference) / load_reference if isinstance(lag_load_value, (int, float)) and load_reference > 0 else 0.0
        load_adjustment = max(-0.3, min(0.3, load_delta)) * 90.0
        predicted = day_ahead * 0.5 + lag_two_rt * 0.3 + historical_mean * 0.2 + load_adjustment
        variance = average([(value - historical_mean) ** 2 for value in historical_prices])
        interval = max(28.0, min(180.0, math.sqrt(variance) * 1.15 or 45.0))
        spread = day_ahead - predicted
        direction = "实时高于日前" if spread < -10 else "实时低于日前" if spread > 10 else "价差收敛"
        drivers: list[str] = []
        if lag_two_rt > historical_mean * 1.15:
            drivers.append("D-2实时价格偏高")
        if day_ahead > historical_mean * 1.15:
            drivers.append("D-1日前价格偏高")
        if load_delta > 0.05:
            drivers.append("D-1组合负荷上升")
        if load_delta < -0.05:
            drivers.append("D-1组合负荷下降")
        if interval >= 100:
            drivers.append("同小时历史波动较大")
        results.append(
            {
                "date": date,
                "marketCode": market_code.upper(),
                "time": time_point,
                "pointIndex": index + 1,
                "dayAheadReference": round(day_ahead, 3),
                "predictedPrice": round(predicted, 3),
                "lower": round(predicted - interval, 3),
                "upper": round(predicted + interval, 3),
                "spread": round(spread, 3),
                "direction": direction,
                "confidence": round(max(0.55, min(0.84, 0.58 + len(historical_prices) * 0.015 - (0 if isinstance(lag_load_value, (int, float)) else 0.05))), 3),
                "drivers": "、".join(drivers or ["供需相对平稳"]),
                "modelVersion": model_version,
                "actualPrice": target.get("realtimePriceYuanMwh"),
                "actualSpread": round(day_ahead - float(target["realtimePriceYuanMwh"]), 3) if isinstance(target.get("realtimePriceYuanMwh"), (int, float)) else None,
                "sourceFile": "spot_prices_2026h1.json + portfolio_load_hourly_2026h1.json",
                "importBatchId": f"sd-backtest-{date.replace('-', '')}-01",
                "inputDates": {"dayAhead": lag_one_date, "realtime": lag_two_date, "load": lag_one_date},
            }
        )
    return results


def risk_signal(probability: float) -> dict[str, Any]:
    normalized = max(0.0, min(1.0, probability))
    level = "HIGH" if normalized >= 0.65 else "MEDIUM" if normalized >= 0.3 else "LOW"
    return {"probability": round(normalized, 4), "level": level}


def build_backtest_metrics(rows: list[dict[str, Any]], high_price_threshold: float) -> dict[str, Any]:
    paired = [row for row in rows if isinstance(row.get("actualPrice"), (int, float))]
    if not paired:
        return BacktestMetricsV1().model_dump()
    errors = [float(row["predictedPrice"]) - float(row["actualPrice"]) for row in paired]
    negative_correct = sum((float(row["predictedPrice"]) < 0) == (float(row["actualPrice"]) < 0) for row in paired)
    extreme_actual = [row for row in paired if float(row["actualPrice"]) >= high_price_threshold]
    extreme_recalled = sum(float(row["upper"]) >= high_price_threshold for row in extreme_actual)
    return {
        "window_start": paired[0]["date"],
        "window_end": paired[-1]["date"],
        "sample_count": len(paired),
        "mae_yuan_per_mwh": round(average([abs(value) for value in errors]), 4),
        "rmse_yuan_per_mwh": round(math.sqrt(average([value ** 2 for value in errors])), 4),
        "bias_yuan_per_mwh": round(average(errors), 4),
        "negative_price_direction_accuracy": round(negative_correct / len(paired), 4),
        "extreme_high_price_recall": round(extreme_recalled / len(extreme_actual), 4) if extreme_actual else None,
    }


def build_baseline_v1_result(request: ModelRunCreateRequestV1, run_id: str, rows: list[dict[str, Any]]) -> ForecastStrategyResultV1:
    high_price_threshold = float(request.parameters.get("high_price_threshold_yuan_per_mwh", 800))
    ready, available, missing = strategy_readiness(request.input_summary)
    periods: list[dict[str, Any]] = []
    for row in rows:
        day_ahead = float(row["dayAheadReference"])
        realtime_p10 = float(row["lower"])
        realtime_p50 = float(row["predictedPrice"])
        realtime_p90 = float(row["upper"])
        width = max(realtime_p90 - realtime_p10, 1.0)
        negative_probability = max(0.0, min(1.0, (0 - realtime_p10) / width)) if realtime_p10 < 0 else 0.0
        high_probability = max(0.0, min(1.0, (realtime_p90 - high_price_threshold) / width)) if realtime_p90 > high_price_threshold else 0.0
        reason_codes: list[str] = []
        if realtime_p10 < 0:
            reason_codes.append("NEGATIVE_INTERVAL_CROSSED")
        if realtime_p90 > high_price_threshold:
            reason_codes.append("HIGH_PRICE_INTERVAL_CROSSED")
        if float(row["upper"]) - float(row["lower"]) >= 200:
            reason_codes.append("WIDE_PREDICTION_INTERVAL")
        periods.append(
            {
                "period": int(row["pointIndex"]),
                "datetime": period_start_timestamp(request.market_date, int(row["pointIndex"])),
                "day_ahead_price_yuan_per_mwh": {"p10": day_ahead, "p50": day_ahead, "p90": day_ahead},
                "real_time_price_yuan_per_mwh": {"p10": realtime_p10, "p50": realtime_p50, "p90": realtime_p90},
                "spread_day_ahead_minus_real_time_yuan_per_mwh": round(day_ahead - realtime_p50, 3),
                "negative_price_risk": risk_signal(negative_probability),
                "high_price_risk": risk_signal(high_probability),
                "risk_reason_codes": reason_codes or ["NO_THRESHOLD_RISK"],
                "confidence": float(row["confidence"]),
                "data_completeness": "PARTIAL" if missing else "COMPLETE",
                "strategy_suggestion": {
                    "action": "HOLD",
                    "volume_mwh": 0,
                    "price_yuan_per_mwh": None,
                    "confidence": float(row["confidence"]),
                    "reason_codes": ["STRATEGY_INPUTS_INCOMPLETE"] if missing else ["MANUAL_REVIEW_REQUIRED"],
                },
            }
        )
    created_at = utc_now()
    result = ForecastStrategyResultV1(
        request_id=request.request_id,
        run_id=run_id,
        market_code=request.market_code.upper(),
        market_date=request.market_date,
        model={"id": request.model_id, "name": "历史时序回测基线", "version": request.model_version},
        data_snapshot={
            "version": request.data_version,
            "created_at": created_at,
            "source_versions": {
                "prices": "sd-spot-2026h1-v1",
                "portfolio_load": "sd-portfolio-load-2026h1-v1",
            },
            "available_domains": available,
            "missing_domains": missing,
        },
        backtest=build_backtest_metrics(rows, high_price_threshold),
        periods=periods,
        strategy_ready=ready,
        warnings=[
            "BASELINE_FOR_BACKTEST_ONLY",
            "DAY_AHEAD_QUANTILES_USE_A_DETERMINISTIC_REFERENCE",
            "NO_AUTOMATIC_TRADING",
        ],
    )
    validate_forecast_result(result)
    return result


def build_price_forecast_v1_result(request: ModelRunCreateRequestV1, run_id: str, raw: dict[str, Any]) -> ForecastStrategyResultV1:
    """Convert the supplied model output into the platform's fixed v1 contract."""
    actual_domains = set(request.input_summary.get("available_domains", [])) | {"prices", "weather"}
    if raw["summary"].get("supply_used_in_da_final") or raw["summary"].get("supply_used_in_rt_final"):
        actual_domains.add("market_supply_history")
    ready, available, missing = strategy_readiness({**request.input_summary, "available_domains": sorted(actual_domains)})
    high_price_threshold = float(request.parameters.get("high_price_threshold_yuan_per_mwh", 500))
    periods: list[dict[str, Any]] = []
    for row in raw["forecast"]:
        da = {"p10": row["da_p10"], "p50": row["da_p50"], "p90": row["da_p90"]}
        rt = {"p10": row["rt_p10"], "p50": row["rt_p50"], "p90": row["rt_p90"]}
        negative_probability = float(row["negative_risk_probability"])
        high_probability = float(row["high_price_risk_probability"])
        reason_codes: list[str] = []
        if negative_probability > 0:
            reason_codes.append("NEGATIVE_INTERVAL_CROSSED")
        if high_probability > 0:
            reason_codes.append("HIGH_PRICE_INTERVAL_CROSSED")
        if not reason_codes:
            reason_codes.append("NO_THRESHOLD_RISK")
        if missing:
            strategy_reason = ["STRATEGY_INPUTS_INCOMPLETE"]
        else:
            strategy_reason = ["MANUAL_REVIEW_REQUIRED", "NO_AUTOMATIC_TRADING"]
        periods.append(
            {
                "period": row["period"],
                "datetime": period_start_timestamp(request.market_date, row["period"]),
                "day_ahead_price_yuan_per_mwh": da,
                "real_time_price_yuan_per_mwh": rt,
                "spread_day_ahead_minus_real_time_yuan_per_mwh": round(da["p50"] - rt["p50"], 3),
                "negative_price_risk": risk_signal(negative_probability),
                "high_price_risk": risk_signal(high_probability),
                "risk_reason_codes": reason_codes,
                "confidence": row["confidence"],
                "data_completeness": "PARTIAL" if missing else "COMPLETE",
                "strategy_suggestion": {
                    "action": "HOLD",
                    "volume_mwh": 0,
                    "price_yuan_per_mwh": None,
                    "confidence": row["confidence"],
                    "reason_codes": strategy_reason,
                },
            }
        )
    da_metrics = raw["summary"]["da_metrics"]
    rt_metrics = raw["summary"]["rt_metrics"]
    weather_metrics = raw["summary"].get("da_weather_candidate_metrics") or {}
    baseline_metrics = raw["summary"].get("da_best_baseline_metrics") or {}
    weather_mae = weather_metrics.get("mae")
    baseline_mae = baseline_metrics.get("mae")
    weather_improvement = baseline_mae - weather_mae if baseline_mae is not None and weather_mae is not None else None
    rt_weather_metrics = raw["summary"].get("rt_weather_candidate_metrics") or {}
    rt_baseline_metrics = raw["summary"].get("rt_best_baseline_metrics") or {}
    rt_weather_mae = rt_weather_metrics.get("mae")
    rt_baseline_mae = rt_baseline_metrics.get("mae")
    rt_weather_improvement = rt_baseline_mae - rt_weather_mae if rt_baseline_mae is not None and rt_weather_mae is not None else None
    result = ForecastStrategyResultV1(
        request_id=request.request_id,
        run_id=run_id,
        market_code=request.market_code.upper(),
        market_date=request.market_date,
        model={"id": request.model_id, "name": "山东现货价格概率集成预测", "version": request.model_version},
        data_snapshot={
            "version": request.data_version,
            "created_at": utc_now(),
            "source_versions": {
                "prices": "sd-spot-2026h1-v1",
                "portfolio_load": "sd-portfolio-load-2026h1-v1",
                "weather": raw["summary"].get("weather_data_version", "sd-weather-gfs-hourly-v2"),
                "market_supply": raw["summary"].get("supply_data_version", "sd-system-output-hourly-2026h1-v1"),
                "algorithm_package": "wangyifan-111/-@33adaad1 integrated_price_forecast.py",
            },
            "available_domains": available,
            "missing_domains": missing,
        },
        backtest={
            "window_start": raw["summary"]["window_start"],
            "window_end": raw["summary"]["window_end"],
            "sample_count": raw["summary"]["sample_count"],
            "mae_yuan_per_mwh": da_metrics["mae"],
            "rmse_yuan_per_mwh": da_metrics["rmse"],
            "bias_yuan_per_mwh": da_metrics["bias"],
            "negative_price_direction_accuracy": da_metrics["negative_accuracy"],
            "extreme_high_price_recall": da_metrics["high_price_recall"],
            "evaluation_method": raw["summary"].get("evaluation_method"),
            "feature_set": raw["summary"].get("feature_set"),
            "weather_data_version": raw["summary"].get("weather_data_version"),
            "weather_known_before_declaration": raw["summary"].get("weather_known_before_declaration"),
            "weather_used_in_final_model": bool(
                raw["summary"].get("weather_used_in_da_final")
                and raw["summary"].get("weather_used_in_rt_final")
            ),
            "baseline_mae_yuan_per_mwh": baseline_mae,
            "weather_mae_yuan_per_mwh": weather_mae,
            "weather_mae_improvement_yuan_per_mwh": weather_improvement,
            "weather_mae_improvement_pct": (
                weather_improvement / baseline_mae * 100
                if weather_improvement is not None and baseline_mae
                else None
            ),
            "real_time_mae_yuan_per_mwh": rt_metrics["mae"],
            "real_time_rmse_yuan_per_mwh": rt_metrics["rmse"],
            "real_time_baseline_mae_yuan_per_mwh": rt_baseline_mae,
            "real_time_weather_mae_yuan_per_mwh": rt_weather_mae,
            "real_time_weather_mae_improvement_pct": (
                rt_weather_improvement / rt_baseline_mae * 100
                if rt_weather_improvement is not None and rt_baseline_mae
                else None
            ),
            "day_ahead_model_name": raw["summary"].get("da_selected"),
            "real_time_model_name": raw["summary"].get("rt_selected"),
            "supply_data_version": raw["summary"].get("supply_data_version"),
            "supply_used_in_final_model": bool(
                raw["summary"].get("supply_used_in_da_final")
                or raw["summary"].get("supply_used_in_rt_final")
            ),
            "supply_issue_time_available": raw["summary"].get("supply_issue_time_available"),
            "supply_backtest_leakage_safe": raw["summary"].get("supply_backtest_leakage_safe"),
            "supply_usage_boundary": raw["summary"].get("supply_usage_boundary"),
            "day_ahead_supply_candidate_mae_yuan_per_mwh": (raw["summary"].get("da_supply_candidate_metrics") or {}).get("mae"),
            "real_time_supply_candidate_mae_yuan_per_mwh": (raw["summary"].get("rt_supply_candidate_metrics") or {}).get("mae"),
            "day_ahead_final_improvement_vs_weather_pct": (
                (weather_mae - da_metrics["mae"]) / weather_mae * 100
                if weather_mae and da_metrics.get("mae") is not None
                else None
            ),
            "spread_direction_accuracy": raw["summary"].get("spread_direction_accuracy"),
            "spread_mae_yuan_per_mwh": raw["summary"].get("spread_mae"),
            "day_ahead_interval_coverage": raw["summary"].get("da_interval_coverage"),
            "real_time_interval_coverage": raw["summary"].get("rt_interval_coverage"),
            "adaptive_ensemble": raw["summary"].get("adaptive_ensemble"),
            "ensemble_weights": raw["summary"].get("ensemble_weights"),
            "deep_challengers": raw["summary"].get("deep_challengers"),
            "consistency_constraint": raw["summary"].get("consistency_constraint"),
            "post_day_ahead_realtime": raw["summary"].get("post_day_ahead_realtime"),
        },
        periods=periods,
        strategy_ready=ready,
        warnings=[
            "MODEL_INPUTS_INCLUDE_PRICES_CALENDAR_LAGS_PRE_DECLARATION_GFS_WEATHER_AND_LAGGED_SUPPLY_FORECASTS",
            "WEATHER_AVAILABILITY_CONFIRMED_BY_BUSINESS_OWNER",
            "SUPPLY_SOURCE_ISSUE_TIME_MISSING_D1_D7_LAGGED_USE_ONLY",
            "Q1_ACTUAL_SUPPLY_EXCLUDED_FROM_TARGET_DAY_FEATURES",
            "BACKTEST_USES_ROLLING_DAILY_PRE_DECLARATION_CUTOFF",
            f"RT_BACKTEST_MAE_YUAN_PER_MWH={rt_metrics['mae']:.3f}" if rt_metrics["mae"] is not None else "RT_BACKTEST_METRICS_UNAVAILABLE",
            f"SPREAD_DIRECTION_ACCURACY={raw['summary'].get('spread_direction_accuracy', 0):.3f}",
            "STRATEGY_INPUTS_INCOMPLETE_HOLD_ONLY" if missing else "MANUAL_REVIEW_REQUIRED",
            "NO_AUTOMATIC_TRADING",
        ],
    )
    validate_forecast_result(result)
    return result


def current_strategy_from_result(result: ForecastStrategyResultV1) -> list[dict[str, Any]]:
    return [
        {"period": point.period, "datetime": point.datetime, **point.strategy_suggestion.model_dump()}
        for point in result.periods
    ]


def complete_model_run(run_id: str, result: ForecastStrategyResultV1, actor: str) -> dict[str, Any]:
    row = get_model_run_row(run_id)
    validate_forecast_result(result, row)
    now = utc_now()
    strategy = current_strategy_from_result(result)
    with closing(connect()) as connection, connection:
        connection.execute(
            """
            UPDATE model_runs_v1
            SET status = 'SUCCEEDED', review_status = 'DRAFT', result_json = ?, current_strategy_json = ?,
                error_code = NULL, error_message = NULL, completed_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (
                json.dumps(result.model_dump(), ensure_ascii=False),
                json.dumps(strategy, ensure_ascii=False),
                now,
                now,
                run_id,
            ),
        )
    write_run_audit(run_id, "RESULT_STORED", actor, {"point_count": len(result.periods), "strategy_ready": result.strategy_ready})
    return public_model_run(get_model_run_row(run_id))


def create_model_run(request: ModelRunCreateRequestV1, parent_run_id: str | None = None, actor: str = "platform") -> tuple[dict[str, Any], bool]:
    purge_expired_runs()
    market_code = request.market_code.upper()
    market_profile(market_code)
    validate_business_date(request.market_date)
    if request.model_id == "lag-baseline":
        input_summary = dict(request.input_summary)
        available_domains = {str(item) for item in input_summary.get("available_domains", [])}
        input_summary["available_domains"] = sorted(available_domains | {"prices", "load_actual"})
        request = request.model_copy(update={"input_summary": input_summary})
    request_payload = request.model_dump()
    request_payload["market_code"] = market_code
    with closing(connect()) as connection, connection:
        existing = connection.execute("SELECT * FROM model_runs_v1 WHERE request_id = ?", (request.request_id,)).fetchone()
    if existing is not None:
        if json.loads(existing["request_json"]) != request_payload:
            raise api_error(409, "REQUEST_ID_CONFLICT", "request_id already exists with a different payload")
        return public_model_run(refresh_run_timeout(existing["run_id"])), True

    run_id = f"run-{uuid.uuid4().hex}"
    created_at = utc_now()
    ready, available, missing = strategy_readiness(request.input_summary)
    snapshot = {
        "version": request.data_version,
        "market_code": market_code,
        "market_date": request.market_date,
        "timezone": "Asia/Shanghai",
        "available_domains": available,
        "missing_domains": missing,
        "strategy_ready": ready,
        "hourly_data_url": f"/api/v1/hourly-data?market_code={market_code}&market_date={request.market_date}&data_version={request.data_version}",
        "captured_at": created_at,
    }
    status = "PENDING_PROVIDER"
    with closing(connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO model_runs_v1
                (run_id, request_id, parent_run_id, market_code, market_date, model_id, model_version,
                 data_version, status, review_status, timeout_seconds, request_json, input_snapshot_json,
                 parameters_json, input_summary_json, created_at, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                request.request_id,
                parent_run_id,
                market_code,
                request.market_date,
                request.model_id,
                request.model_version,
                request.data_version,
                status,
                request.timeout_seconds,
                json.dumps(request_payload, ensure_ascii=False),
                json.dumps(snapshot, ensure_ascii=False),
                json.dumps(request.parameters, ensure_ascii=False),
                json.dumps(request.input_summary, ensure_ascii=False),
                created_at,
                created_at if status == "RUNNING" else None,
                created_at,
            ),
        )
    write_run_audit(run_id, "RUN_CREATED", actor, {"status": status, "parent_run_id": parent_run_id})

    if False and request.model_id == "lag-baseline":
        try:
            rows = build_forecast(request.market_date, request.model_version, market_code)
            result = build_baseline_v1_result(request.model_copy(update={"market_code": market_code}), run_id, rows)
            complete_model_run(run_id, result, "platform-baseline")
        except Exception as error:
            now = utc_now()
            code = error.detail.get("code", "RUN_FAILED") if isinstance(error, HTTPException) and isinstance(error.detail, dict) else "RUN_FAILED"
            message = error.detail.get("message", str(error)) if isinstance(error, HTTPException) and isinstance(error.detail, dict) else str(error)
            with closing(connect()) as connection, connection:
                connection.execute(
                    """
                    UPDATE model_runs_v1
                    SET status = 'FAILED', error_code = ?, error_message = ?, completed_at = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (code, message, now, now, run_id),
                )
            write_run_audit(run_id, "RUN_FAILED", "platform-baseline", {"error_code": code, "message": message})
    elif request.model_id == "price-forecast":
        try:
            raw = run_price_forecast(PRIVATE_DATA, request.market_date)
            result = build_price_forecast_v1_result(request.model_copy(update={"market_code": market_code}), run_id, raw)
            complete_model_run(run_id, result, "price-forecast-local")
        except Exception as error:
            now = utc_now()
            code = error.detail.get("code", "RUN_FAILED") if isinstance(error, HTTPException) and isinstance(error.detail, dict) else "RUN_FAILED"
            message = error.detail.get("message", str(error)) if isinstance(error, HTTPException) and isinstance(error.detail, dict) else str(error)
            with closing(connect()) as connection, connection:
                connection.execute(
                    """
                    UPDATE model_runs_v1
                    SET status = 'FAILED', error_code = ?, error_message = ?, completed_at = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (code, message, now, now, run_id),
                )
            write_run_audit(run_id, "RUN_FAILED", "price-forecast-local", {"error_code": code, "message": message})
    return public_model_run(get_model_run_row(run_id)), False


def store_price_asset_v1(request: PriceAssetUploadRequestV1) -> tuple[dict[str, Any], bool]:
    market_code = request.market_code.upper()
    market_profile(market_code)
    normalized_updated_at = shanghai_timestamp(request.updated_at)
    seen: set[tuple[str, int]] = set()
    dates: set[str] = set()
    missing_day_ahead = 0
    missing_realtime = 0
    for record in request.records:
        validate_business_date(record.market_date)
        validate_hourly_timestamp(record.market_date, record.period, record.datetime)
        validate_customer_alias(record.customer_id)
        if record.customer_id is not None:
            raise api_error(422, "PRICE_ASSET_CUSTOMER_NOT_ALLOWED", "Price assets must use customer_id=null")
        if record.day_ahead_price_yuan_per_mwh is None:
            missing_day_ahead += 1
        if record.real_time_price_yuan_per_mwh is None:
            missing_realtime += 1
        key = (record.market_date, record.period)
        if key in seen:
            raise api_error(422, "DUPLICATE_PERIOD", "Duplicate market_date and period in price upload", market_date=record.market_date, period=record.period)
        seen.add(key)
        dates.add(record.market_date)
    payload = request.model_dump()
    payload["market_code"] = market_code
    payload["updated_at"] = normalized_updated_at
    asset_id = f"prices:{market_code}:{request.data_version}"
    quality = "complete" if all(sum(date == item[0] for item in seen) == 24 for date in dates) and missing_day_ahead == 0 and missing_realtime == 0 else "partial"
    with closing(connect()) as connection, connection:
        existing = connection.execute("SELECT * FROM data_asset_batches WHERE asset_id = ?", (asset_id,)).fetchone()
        if existing is not None:
            if json.loads(existing["payload_json"]) != payload:
                raise api_error(409, "DATA_VERSION_CONFLICT", "data_version is immutable and already exists with different content")
            return {
                "accepted": True,
                "idempotent_replay": True,
                "asset_id": asset_id,
                "data_version": request.data_version,
                "record_count": existing["record_count"],
                "quality_status": existing["quality_status"],
                "updated_at": existing["updated_at"],
            }, True
        created_at = utc_now()
        connection.execute(
            """
            INSERT INTO data_asset_batches
                (asset_id, domain, market_code, data_version, source_system, request_id,
                 record_count, quality_status, payload_json, updated_at, created_at)
            VALUES (?, 'prices', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                market_code,
                request.data_version,
                request.source_system,
                request.request_id,
                len(request.records),
                quality,
                json.dumps(payload, ensure_ascii=False),
                normalized_updated_at,
                created_at,
            ),
        )
    return {
        "accepted": True,
        "idempotent_replay": False,
        "asset_id": asset_id,
        "data_version": request.data_version,
        "record_count": len(request.records),
        "date_count": len(dates),
        "quality_status": quality,
        "missing_day_ahead_count": missing_day_ahead,
        "missing_realtime_count": missing_realtime,
        "updated_at": normalized_updated_at,
    }, False


def unified_hourly_data_v1(
    market_date: str,
    market_code: str = "SD",
    customer_id: str | None = None,
    data_version: str | None = None,
) -> dict[str, Any]:
    normalized_market = market_code.upper()
    require_sample_market(normalized_market)
    validate_business_date(market_date)
    validate_customer_alias(customer_id)
    uploaded = latest_uploaded_price_asset(normalized_market, market_date, data_version)
    if data_version and data_version != BUILTIN_DATA_VERSION and uploaded is None:
        raise api_error(404, "DATA_VERSION_NOT_FOUND", f"No price asset found for data_version={data_version} and market_date={market_date}")

    source_versions: dict[str, str] = {}
    if uploaded is not None:
        price_by_period = {int(row["period"]): row for row in uploaded["records"]}
        resolved_version = uploaded["data_version"]
        updated_at = uploaded["updated_at"]
        source_versions["prices"] = resolved_version
    else:
        prices = [row for row in load_private_json("spot_prices_2026h1.json") if row.get("date") == market_date]
        if not prices:
            raise api_error(404, "HOURLY_DATA_NOT_FOUND", f"No hourly price data for {normalized_market} on {market_date}")
        price_by_period = {int(row["time"][:2]): row for row in prices}
        resolved_version = BUILTIN_DATA_VERSION
        quality = load_private_json("data_quality_2026h1.json")
        updated_at = shanghai_timestamp(quality["generatedAt"])
        source_versions["prices"] = prices[0].get("sourceBatchId", "sd-spot-2026h1-v1")

    load_by_period: dict[int, float] = {}
    if customer_id:
        profiles = load_private_json("customer_load_profiles_2026h1.json")
        if customer_id not in {row.get("customerAlias") for row in profiles}:
            raise api_error(404, "CUSTOMER_NOT_FOUND", f"Unknown stable customer_id: {customer_id}")
        customer_rows = [
            row for row in load_private_json("customer_load_hourly_anon_2026h1.json")
            if row.get("date") == market_date and row.get("customerAlias") == customer_id
        ]
        if customer_rows:
            load_by_period = {index + 1: float(value) for index, value in enumerate(customer_rows[0].get("hourlyMwh", []))}
            source_versions["load_actual"] = customer_rows[0].get("sourceBatchId", "sd-customer-load-hourly-2026h1-v1")
    else:
        load_rows = [row for row in load_private_json("portfolio_load_hourly_2026h1.json") if row.get("date") == market_date]
        load_by_period = {int(row["time"][:2]): float(row["totalMwh"]) for row in load_rows}
        if load_rows:
            source_versions["load_actual"] = load_rows[0].get("sourceBatchId", "sd-portfolio-load-2026h1-v1")

    weather_asset = load_private_json("weather_hourly_gfs_20260501_20260701.json")
    weather_by_period = {
        int(row["period"]): row
        for row in weather_asset.get("rows", [])
        if row.get("marketDate") == market_date
    }
    if weather_by_period:
        source_versions["weather"] = weather_asset.get("dataVersion", "sd-weather-gfs-hourly-v1")

    rows: list[dict[str, Any]] = []
    for period in range(1, 25):
        price = price_by_period.get(period, {})
        if uploaded is not None:
            day_ahead = price.get("day_ahead_price_yuan_per_mwh")
            realtime = price.get("real_time_price_yuan_per_mwh")
        else:
            day_ahead = price.get("dayAheadPriceYuanMwh")
            realtime = price.get("realtimePriceYuanMwh")
        rows.append(
            {
                "market_date": market_date,
                "period": period,
                "datetime": period_start_timestamp(market_date, period),
                "day_ahead_price_yuan_per_mwh": day_ahead,
                "real_time_price_yuan_per_mwh": realtime,
                "load_actual_mwh": load_by_period.get(period),
                "load_forecast_mwh": None,
                "temperature_2m_c": weather_by_period.get(period, {}).get("temperature2mC"),
                "wind_speed_10m_mps": weather_by_period.get(period, {}).get("windSpeed10mMs"),
                "wind_speed_100m_mps": weather_by_period.get(period, {}).get("windSpeed100mMs"),
                "shortwave_radiation_ghi_w_per_m2": weather_by_period.get(period, {}).get("shortwaveRadiationGhiWm2"),
                "cloud_cover_pct": weather_by_period.get(period, {}).get("cloudCoverPct"),
                "precipitation_mm": weather_by_period.get(period, {}).get("precipitationMm"),
                "relative_humidity_pct": weather_by_period.get(period, {}).get("relativeHumidityPct"),
                "weather_forecast_issue_time": weather_by_period.get(period, {}).get("forecastIssueTime"),
                "customer_id": customer_id,
                "data_version": resolved_version,
                "updated_at": updated_at,
            }
        )
    missing_fields = {
        "day_ahead_price": sum(row["day_ahead_price_yuan_per_mwh"] is None for row in rows),
        "real_time_price": sum(row["real_time_price_yuan_per_mwh"] is None for row in rows),
        "load_actual": sum(row["load_actual_mwh"] is None for row in rows),
        "load_forecast": 24,
        "weather": sum(
            any(row[field] is None for field in (
                "temperature_2m_c",
                "wind_speed_10m_mps",
                "wind_speed_100m_mps",
                "shortwave_radiation_ghi_w_per_m2",
                "cloud_cover_pct",
                "precipitation_mm",
                "relative_humidity_pct",
            ))
            for row in rows
        ),
    }
    available_domains: set[str] = set()
    if missing_fields["day_ahead_price"] == 0 and missing_fields["real_time_price"] == 0:
        available_domains.add("prices")
    if missing_fields["load_actual"] == 0:
        available_domains.add("load_actual")
    if missing_fields["load_forecast"] == 0:
        available_domains.add("load_forecast")
    if missing_fields["weather"] == 0:
        available_domains.add("weather")
    strategy_ready, available_domains_list, strategy_missing_domains = strategy_readiness(
        {"available_domains": sorted(available_domains)}
    )
    return {
        "schema_version": API_VERSION,
        "market_code": normalized_market,
        "market_date": market_date,
        "timezone": "Asia/Shanghai",
        "customer_id": customer_id,
        "data_version": resolved_version,
        "updated_at": updated_at,
        "point_count": len(rows),
        "source_versions": source_versions,
        "missing_field_counts": missing_fields,
        "available_domains": available_domains_list,
        "strategy_ready": strategy_ready,
        "strategy_missing_domains": strategy_missing_domains,
        "rows": rows,
    }


def build_loss_analysis(
    date: str,
    market_code: str = "SD",
    revenue_price: float | None = None,
    other_cost: float = 0.0,
) -> dict[str, Any]:
    require_sample_market(market_code)
    validate_business_date(date)
    if revenue_price is not None and revenue_price < 0:
        raise HTTPException(status_code=422, detail="revenue_price must be non-negative")
    if other_cost < 0:
        raise HTTPException(status_code=422, detail="other_cost must be non-negative")

    prices = sorted(
        [row for row in load_private_json("spot_prices_2026h1.json") if row.get("date") == date],
        key=lambda row: row["time"],
    )
    if not prices:
        raise HTTPException(status_code=404, detail=f"No spot price rows for {date}")

    settlement = load_json("settlement_2026-06-21.json")
    settlement_rows = settlement.get("hourly", [])
    exact_settlement = settlement.get("summary", {}).get("date") == date and bool(settlement_rows)
    settlement_by_time = {row["time"]: row for row in settlement_rows} if exact_settlement else {}
    portfolio_by_time = {
        row["time"]: row
        for row in load_private_json("portfolio_load_hourly_2026h1.json")
        if row.get("date") == date
    }

    hourly: list[dict[str, Any]] = []
    for price in prices:
        time_point = price["time"]
        settlement_row = settlement_by_time.get(time_point)
        load_row = portfolio_by_time.get(time_point)
        energy = settlement_row.get("totalEnergyMwh") if settlement_row else load_row.get("totalMwh") if load_row else None
        actual_cost = settlement_row.get("totalSettlementAmountYuan") if settlement_row else None
        if not exact_settlement and isinstance(energy, (int, float)) and isinstance(price.get("realtimePriceYuanMwh"), (int, float)):
            actual_cost = float(energy) * float(price["realtimePriceYuanMwh"])
        day_ahead_benchmark = None
        realtime_benchmark = None
        if isinstance(energy, (int, float)):
            if isinstance(price.get("dayAheadPriceYuanMwh"), (int, float)):
                day_ahead_benchmark = float(energy) * float(price["dayAheadPriceYuanMwh"])
            if isinstance(price.get("realtimePriceYuanMwh"), (int, float)):
                realtime_benchmark = float(energy) * float(price["realtimePriceYuanMwh"])
        delta = actual_cost - day_ahead_benchmark if isinstance(actual_cost, (int, float)) and isinstance(day_ahead_benchmark, (int, float)) else None
        hourly.append({
            "date": date,
            "time": time_point,
            "energyMwh": energy,
            "actualCostYuan": actual_cost,
            "actualUnitCostYuanMwh": actual_cost / energy if isinstance(actual_cost, (int, float)) and isinstance(energy, (int, float)) and energy > 0 else None,
            "dayAheadPriceYuanMwh": price.get("dayAheadPriceYuanMwh"),
            "realtimePriceYuanMwh": price.get("realtimePriceYuanMwh"),
            "dayAheadBenchmarkYuan": day_ahead_benchmark,
            "realtimeBenchmarkYuan": realtime_benchmark,
            "deltaVsDayAheadYuan": delta,
            "dayAheadEnergyShare": (settlement_row.get("dayAheadEnergyMwh", 0) / energy) if settlement_row and isinstance(energy, (int, float)) and energy > 0 else None,
        })

    def finite_sum(key: str) -> float | None:
        values = [float(row[key]) for row in hourly if isinstance(row.get(key), (int, float))]
        return sum(values) if values else None

    total_energy = finite_sum("energyMwh")
    actual_cost = finite_sum("actualCostYuan")
    day_ahead_benchmark = finite_sum("dayAheadBenchmarkYuan")
    realtime_benchmark = finite_sum("realtimeBenchmarkYuan")
    day_ahead_energy = sum(float(row.get("dayAheadEnergyMwh", 0)) for row in settlement_rows) if exact_settlement else None
    contract_energy = sum(float(row.get("intraMarketContractEnergyMwh", 0)) for row in settlement_rows) if exact_settlement else None
    recorded_coverage = day_ahead_energy + contract_energy if exact_settlement else None
    revenue = total_energy * revenue_price if total_energy is not None and revenue_price is not None else None
    gross_margin = revenue - actual_cost - other_cost if revenue is not None and actual_cost is not None else None
    break_even = (actual_cost + other_cost) / total_energy if actual_cost is not None and total_energy and total_energy > 0 else None
    delta_vs_day_ahead = actual_cost - day_ahead_benchmark if actual_cost is not None and day_ahead_benchmark is not None else None
    delta_vs_realtime = actual_cost - realtime_benchmark if actual_cost is not None and realtime_benchmark is not None else None

    return {
        "marketCode": market_code.upper(),
        "date": date,
        "mode": "settlement-fact" if exact_settlement else "portfolio-cost-proxy",
        "dataStatus": "complete" if len([row for row in hourly if row["energyMwh"] is not None]) == len(prices) else "missing-energy",
        "basis": "客户结算案例-001 + 山东市场小时价格" if exact_settlement else "山东组合负荷 × 实时价格成本代理",
        "summary": {
            "totalEnergyMwh": total_energy,
            "actualCostYuan": actual_cost,
            "actualUnitCostYuanMwh": actual_cost / total_energy if actual_cost is not None and total_energy and total_energy > 0 else None,
            "dayAheadBenchmarkYuan": day_ahead_benchmark,
            "realtimeBenchmarkYuan": realtime_benchmark,
            "deltaVsDayAheadYuan": delta_vs_day_ahead,
            "deltaVsRealtimeYuan": delta_vs_realtime,
            "dayAheadEnergyMwh": day_ahead_energy,
            "contractEnergyMwh": contract_energy,
            "recordedCoverageMwh": recorded_coverage,
            "recordedCoverageShare": recorded_coverage / total_energy if recorded_coverage is not None and total_energy and total_energy > 0 else None,
            "revenuePriceYuanMwh": revenue_price,
            "otherCostYuan": other_cost,
            "revenueYuan": revenue,
            "grossMarginYuan": gross_margin,
            "grossMarginRate": gross_margin / revenue if gross_margin is not None and revenue and revenue > 0 else None,
            "breakEvenRevenuePriceYuanMwh": break_even,
        },
        "hourly": hourly,
        "missingData": ["零售收入", "完整合同持仓", "日前申报与成交", "其他财务费用"] if exact_settlement else ["客户结算电量", "零售收入", "合同持仓", "申报成交", "其他财务费用"],
        "conclusion": "现有数据只能确认批发侧结算成本和市场基准，不能确认真实利润或亏损。" if exact_settlement else "当前为组合成本代理，不等于客户结算成本，不能确认真实利润或亏损。",
        "privacy": "只返回客户案例代号和组合结果，不返回真实名称、户号、计量点或内部哈希。",
    }


def build_customer_profit_contribution(
    date: str,
    market_code: str = "SD",
    revenue_price: float | None = None,
    other_cost: float = 0.0,
) -> dict[str, Any]:
    portfolio = build_loss_analysis(date, market_code, revenue_price, other_cost)
    customer_rows = [
        row
        for row in load_private_json("customer_load_hourly_anon_2026h1.json")
        if row.get("date") == date
    ]
    profiles = {
        row["customerAlias"]: row
        for row in load_private_json("customer_load_profiles_2026h1.json")
        if row.get("customerAlias")
    }
    portfolio_hourly = {row["time"]: row for row in portfolio["hourly"]}
    total_customer_energy = sum(
        float(row.get("totalMwh", 0))
        for row in customer_rows
        if isinstance(row.get("totalMwh"), (int, float))
    )

    customers: list[dict[str, Any]] = []
    for customer_row in customer_rows:
        alias = customer_row.get("customerAlias")
        if not alias:
            continue
        hourly_energy = customer_row.get("hourlyMwh") or []
        hourly: list[dict[str, Any]] = []
        for index, raw_energy in enumerate(hourly_energy[:24]):
            time_point = f"{index + 1:02d}:00"
            group_row = portfolio_hourly.get(time_point, {})
            energy = float(raw_energy) if isinstance(raw_energy, (int, float)) else None
            unit_cost = group_row.get("actualUnitCostYuanMwh")
            day_ahead_price = group_row.get("dayAheadPriceYuanMwh")
            cost = energy * float(unit_cost) if energy is not None and isinstance(unit_cost, (int, float)) else None
            day_ahead_benchmark = energy * float(day_ahead_price) if energy is not None and isinstance(day_ahead_price, (int, float)) else None
            revenue = energy * revenue_price if energy is not None and revenue_price is not None else None
            hourly.append({
                "time": time_point,
                "energyMwh": energy,
                "allocatedCostYuan": cost,
                "allocatedUnitCostYuanMwh": unit_cost,
                "dayAheadPriceYuanMwh": day_ahead_price,
                "dayAheadBenchmarkYuan": day_ahead_benchmark,
                "deltaVsDayAheadYuan": cost - day_ahead_benchmark if cost is not None and day_ahead_benchmark is not None else None,
                "revenueYuan": revenue,
                "marginBeforeOtherCostYuan": revenue - cost if revenue is not None and cost is not None else None,
            })

        customer_energy = sum(float(row["energyMwh"]) for row in hourly if isinstance(row.get("energyMwh"), (int, float)))
        allocated_cost = sum(float(row["allocatedCostYuan"]) for row in hourly if isinstance(row.get("allocatedCostYuan"), (int, float)))
        day_ahead_benchmark = sum(float(row["dayAheadBenchmarkYuan"]) for row in hourly if isinstance(row.get("dayAheadBenchmarkYuan"), (int, float)))
        other_cost_allocation = other_cost * customer_energy / total_customer_energy if total_customer_energy > 0 else 0.0
        revenue = customer_energy * revenue_price if revenue_price is not None else None
        gross_margin = revenue - allocated_cost - other_cost_allocation if revenue is not None else None
        gross_margin_rate = gross_margin / revenue if gross_margin is not None and revenue and revenue > 0 else None
        break_even_price = (allocated_cost + other_cost_allocation) / customer_energy if customer_energy > 0 else None
        if gross_margin is None:
            status = "待补收入"
        elif gross_margin < -1:
            status = "情景负毛利"
        elif gross_margin > 1:
            status = "情景正毛利"
        else:
            status = "接近平衡"
        profile = profiles.get(alias, {})
        customers.append({
            "customerAlias": alias,
            "segment": profile.get("segment"),
            "energyMwh": customer_energy,
            "portfolioEnergyShare": customer_energy / total_customer_energy if total_customer_energy > 0 else None,
            "allocatedCostYuan": allocated_cost,
            "averageCostYuanMwh": allocated_cost / customer_energy if customer_energy > 0 else None,
            "dayAheadBenchmarkYuan": day_ahead_benchmark,
            "deltaVsDayAheadYuan": allocated_cost - day_ahead_benchmark,
            "otherCostAllocationYuan": other_cost_allocation,
            "revenueYuan": revenue,
            "grossMarginYuan": gross_margin,
            "grossMarginRate": gross_margin_rate,
            "breakEvenRevenuePriceYuanMwh": break_even_price,
            "status": status,
            "hourly": hourly,
        })

    customers.sort(
        key=lambda row: row["grossMarginYuan"] if row["grossMarginYuan"] is not None else -row["allocatedCostYuan"]
    )
    customer_cost = sum(row["allocatedCostYuan"] for row in customers)
    customer_revenue = sum(row["revenueYuan"] for row in customers if row["revenueYuan"] is not None) if revenue_price is not None else None
    customer_margin = sum(row["grossMarginYuan"] for row in customers if row["grossMarginYuan"] is not None) if revenue_price is not None else None
    portfolio_energy = portfolio["summary"].get("totalEnergyMwh")
    portfolio_cost = portfolio["summary"].get("actualCostYuan")
    return {
        "marketCode": market_code.upper(),
        "date": date,
        "mode": "customer-contribution-proxy",
        "basis": "客户分时负荷 × 组合逐时成本单价；其他成本按客户电量占比分摊",
        "dataStatus": "complete" if customers else "missing-customer-load",
        "summary": {
            "customerCount": len(customers),
            "customerEnergyMwh": total_customer_energy if customers else None,
            "portfolioEnergyMwh": portfolio_energy,
            "energyCoverageShare": total_customer_energy / portfolio_energy if customers and isinstance(portfolio_energy, (int, float)) and portfolio_energy > 0 else None,
            "customerAllocatedCostYuan": customer_cost if customers else None,
            "portfolioCostYuan": portfolio_cost,
            "unallocatedCostYuan": portfolio_cost - customer_cost if customers and isinstance(portfolio_cost, (int, float)) else None,
            "revenuePriceYuanMwh": revenue_price,
            "customerRevenueYuan": customer_revenue,
            "customerGrossMarginYuan": customer_margin,
            "otherCostYuan": other_cost,
        },
        "customers": customers,
        "missingData": ["逐客户零售合同", "客户级申报与预测", "偏差责任规则", "客户级财务结算"],
        "conclusion": "当前结果用于识别客户成本与负毛利情景贡献，不代表客户真实结算利润。",
        "privacy": "只返回稳定客户代号，不返回真实名称、户号、计量点或内部哈希。",
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="全国电力交易辅助决策平台服务",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "POWER_TRADING_CORS_ORIGINS",
            "http://127.0.0.1:8000,http://localhost:8000,https://lukeooo11.github.io",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or f"http-{uuid.uuid4().hex}"
    if request.url.path == "/api/v1/data-assets/prices":
        content_length = request.headers.get("content-length")
        try:
            oversized = bool(content_length) and int(content_length) > MAX_PRICE_UPLOAD_BYTES
        except ValueError:
            oversized = False
        if oversized:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "PAYLOAD_TOO_LARGE",
                        "message": f"JSON request body must not exceed {MAX_PRICE_UPLOAD_BYTES} bytes",
                        "request_id": request.state.request_id,
                    }
                },
                headers={"X-Request-ID": request.state.request_id},
            )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(HTTPException)
async def stable_http_exception(request: Request, error: HTTPException):
    if not request.url.path.startswith("/api/v1/"):
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail}, headers=error.headers)
    detail = error.detail if isinstance(error.detail, dict) else {"code": "HTTP_ERROR", "message": str(error.detail)}
    payload = {
        "error": {
            "code": detail.get("code", "HTTP_ERROR"),
            "message": detail.get("message", str(error.detail)),
            "details": detail.get("details"),
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    return JSONResponse(status_code=error.status_code, content=payload, headers=error.headers)


@app.exception_handler(RequestValidationError)
async def stable_validation_exception(request: Request, error: RequestValidationError):
    if not request.url.path.startswith("/api/v1/"):
        return JSONResponse(status_code=422, content={"detail": error.errors()})
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": error.errors(),
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.get("/api/system/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "power-trading-platform-upgrade",
        "scope": "company-national-multi-market",
        "market_count": len(MARKETS),
        "connected_market_count": int(shandong_data_available()),
        "database": "connected",
    }


@app.get("/api/markets")
def markets(region: str | None = None, stage: str | None = None) -> dict[str, Any]:
    rows = [market_profile(code) for code, _, _ in MARKETS]
    if region:
        rows = [row for row in rows if row["region"] == region]
    if stage:
        rows = [row for row in rows if row["stage"] == stage]
    return {"market_count": len(rows), "connected_market_count": sum(row["stage"] == "sample" for row in rows), "markets": rows}


@app.get("/api/markets/{market_code}/capabilities")
def market_capabilities(market_code: str) -> dict[str, Any]:
    profile = market_profile(market_code)
    profile["required_domains"] = [
        "市场规则与版本", "日前/实时价格", "系统负荷与新能源", "申报与成交",
        "客户实际负荷", "结算与财务", "中长期持仓", "运行约束与检修",
    ]
    return profile


@app.get("/api/org/roles")
def org_roles() -> dict[str, Any]:
    return {"mode": "placeholder", "roles": ORG_ROLES}


@app.get("/api/org/permissions")
def org_permissions(role: str = "trader") -> dict[str, Any]:
    selected = next((item for item in ORG_ROLES if item["code"] == role), None)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Unknown role: {role}")
    return {
        "mode": "placeholder",
        "role": selected,
        "authorization_enforced": False,
        "note": "正式权限需接入企业 SSO、组织目录和 RBAC 服务。",
    }


@app.get("/api/auth/session")
def auth_session() -> dict[str, Any]:
    return {
        "authenticated": False,
        "mode": "demo",
        "user": None,
        "roles": [],
        "authorized_markets": [],
        "note": "当前角色切换仅用于界面演示，不构成访问控制。",
    }


@app.get("/api/trading/context")
def trading_context(market_code: str = "SD") -> dict[str, Any]:
    profile = market_profile(market_code)
    if profile["stage"] != "sample":
        return {"market": profile, "customer": None, "settlement": None, "forecast_feature_dates": [], "mode": "template"}
    settlement = load_json("settlement_2026-06-21.json")
    prices = load_private_json("spot_prices_2026h1.json")
    quality = load_private_json("data_quality_2026h1.json")
    return {
        "customer": settlement.get("summary", {}).get("company"),
        "market": profile,
        "settlement": settlement.get("summary"),
        "forecast_feature_dates": sorted({row["date"] for row in prices})[2:],
        "quality": quality.get("datasets", {}),
        "mode": "partial-real-data",
    }


@app.get("/api/forecast/features")
def forecast_features(date: str = Query(...), market_code: str = "SD") -> dict[str, Any]:
    require_sample_market(market_code)
    prices = load_private_json("spot_prices_2026h1.json")
    loads = load_private_json("portfolio_load_hourly_2026h1.json")
    lag_one_date = offset_date(date, -1)
    lag_two_date = offset_date(date, -2)
    day_ahead = [row for row in prices if row.get("date") == lag_one_date]
    realtime = [row for row in prices if row.get("date") == lag_two_date]
    load_rows = [row for row in loads if row.get("date") == lag_one_date]
    weather_asset = load_private_json("weather_hourly_gfs_20260501_20260701.json")
    weather_rows = [row for row in weather_asset.get("rows", []) if row.get("marketDate") == date]
    supply_asset = load_private_json("market_supply_hourly_2026h1.json")
    supply_lag_one_rows = [
        row for row in supply_asset.get("rows", [])
        if row.get("marketDate") == lag_one_date and row.get("sourceType") == "FORECAST"
    ]
    supply_lag_seven_date = offset_date(date, -7)
    supply_lag_seven_rows = [
        row for row in supply_asset.get("rows", [])
        if row.get("marketDate") == supply_lag_seven_date and row.get("sourceType") == "FORECAST"
    ]
    return {
        "market_code": market_code.upper(),
        "date": date,
        "point_count": len(day_ahead),
        "input_dates": {"dayAhead": lag_one_date, "realtime": lag_two_date, "load": lag_one_date, "weather": date, "supplyLagOne": lag_one_date, "supplyLagSeven": supply_lag_seven_date},
        "day_ahead_rows": day_ahead,
        "realtime_rows": realtime,
        "portfolio_load_rows": load_rows,
        "weather_rows": weather_rows,
        "supply_lag_one_rows": supply_lag_one_rows,
        "supply_lag_seven_rows": supply_lag_seven_rows,
        "supply_data_version": supply_asset.get("dataVersion"),
        "supply_issue_time_available": supply_asset.get("sourceIssueTimeAvailable", False),
        "weather_data_version": weather_asset.get("dataVersion"),
        "weather_known_before_declaration": weather_asset.get("knownBeforeDeclaration", False),
        "weather_backtest_leakage_safe": weather_asset.get("backtestLeakageSafe", False),
        "note": "D-1日前、D-2实时和组合负荷按当前业务顺序组织；目标日GFS天气已按业务确认口径使用。Q2电源预测当前仅提供D-1/D-7滞后特征，目标日竞价空间、供需边界及精确forecast_issue_time仍待补。",
    }


def weather_hourly_response(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    require_sample_market(market_code)
    asset = load_private_json("weather_hourly_gfs_20260501_20260701.json")
    rows = asset.get("rows", [])
    if date:
        validate_business_date(date)
        rows = [row for row in rows if row.get("marketDate") == date]
    if start_date:
        validate_business_date(start_date)
        rows = [row for row in rows if row.get("marketDate", "") >= start_date]
    if end_date:
        validate_business_date(end_date)
        rows = [row for row in rows if row.get("marketDate", "") <= end_date]
    return {
        "schema_version": asset.get("schemaVersion"),
        "market_code": market_code.upper(),
        "region": asset.get("region"),
        "source_model": asset.get("sourceModel"),
        "data_version": asset.get("dataVersion"),
        "requested_range": asset.get("requestedRange"),
        "available_range": asset.get("availableRange"),
        "forecast_issue_time_available": asset.get("forecastIssueTimeAvailable", False),
        "known_before_declaration": asset.get("knownBeforeDeclaration", False),
        "backtest_leakage_safe": asset.get("backtestLeakageSafe", False),
        "leakage_control_level": asset.get("leakageControlLevel"),
        "availability_confirmation": asset.get("availabilityConfirmation"),
        "usage_boundary": asset.get("usageBoundary"),
        "quality": asset.get("quality"),
        "point_count": len(rows),
        "rows": rows,
    }


@app.get("/api/weather/hourly")
def weather_hourly_legacy(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    return weather_hourly_response(market_code, start_date, end_date, date)


def market_supply_hourly_response(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    require_sample_market(market_code)
    asset = load_private_json("market_supply_hourly_2026h1.json")
    rows = asset.get("rows", [])
    if date:
        validate_business_date(date)
        rows = [row for row in rows if row.get("marketDate") == date]
    if start_date:
        validate_business_date(start_date)
        rows = [row for row in rows if row.get("marketDate", "") >= start_date]
    if end_date:
        validate_business_date(end_date)
        rows = [row for row in rows if row.get("marketDate", "") <= end_date]
    if source_type:
        normalized_type = source_type.upper()
        if normalized_type not in {"ACTUAL", "FORECAST"}:
            raise api_error(422, "INVALID_SOURCE_TYPE", "source_type must be ACTUAL or FORECAST")
        rows = [row for row in rows if row.get("sourceType") == normalized_type]
    return {
        "schema_version": asset.get("schemaVersion"),
        "market_code": market_code.upper(),
        "region": asset.get("region"),
        "data_version": asset.get("dataVersion"),
        "available_range": asset.get("availableRange"),
        "coverage": asset.get("coverage"),
        "source_issue_time_available": asset.get("sourceIssueTimeAvailable", False),
        "backtest_leakage_safe": asset.get("backtestLeakageSafe", False),
        "usage_boundary": asset.get("usageBoundary"),
        "proxy_definition": asset.get("proxyDefinition"),
        "quality": asset.get("quality"),
        "point_count": len(rows),
        "rows": rows,
    }


@app.get("/api/market-supply/hourly")
def market_supply_hourly_legacy(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    return market_supply_hourly_response(market_code, start_date, end_date, date, source_type)


@app.get("/api/spot/prices")
def spot_prices(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    require_sample_market(market_code)
    rows = load_private_json("spot_prices_2026h1.json")
    if date:
        rows = [row for row in rows if row.get("date") == date]
    if start_date:
        rows = [row for row in rows if row.get("date", "") >= start_date]
    if end_date:
        rows = [row for row in rows if row.get("date", "") <= end_date]
    return {
        "market_code": market_code.upper(),
        "price_types": ["dayAheadPriceYuanMwh", "realtimePriceYuanMwh", "spreadYuanMwh"],
        "spread_definition": "日前价格 - 实时价格",
        "point_count": len(rows),
        "rows": rows,
    }


@app.get("/api/customer/load-profile")
def portfolio_load_profile(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    interval: int = 60,
    aggregation: str = "portfolio",
    customer_alias: str | None = None,
) -> dict[str, Any]:
    require_sample_market(market_code)
    if interval != 60:
        raise HTTPException(status_code=409, detail="当前已接入小时数据；15分钟和30分钟负荷待接入。")
    if aggregation == "portfolio":
        rows = load_private_json("portfolio_load_hourly_2026h1.json")
    elif aggregation == "customer":
        if not customer_alias or not re.fullmatch(r"客户-\d+", customer_alias):
            raise HTTPException(status_code=422, detail="aggregation=customer requires customer_alias such as 客户-001")
        rows = [row for row in load_private_json("customer_load_hourly_anon_2026h1.json") if row.get("customerAlias") == customer_alias]
        if not rows:
            raise HTTPException(status_code=404, detail=f"Unknown customer alias: {customer_alias}")
    else:
        raise HTTPException(status_code=422, detail="aggregation must be portfolio or customer")
    if start_date:
        rows = [row for row in rows if row.get("date", "") >= start_date]
    if end_date:
        rows = [row for row in rows if row.get("date", "") <= end_date]
    return {
        "market_code": market_code.upper(),
        "aggregation": aggregation,
        "customer_alias": customer_alias if aggregation == "customer" else None,
        "interval_minutes": 60,
        "point_count": len(rows),
        "rows": rows,
        "note": "前端和本接口不返回原始客户名称、户号、计量点或内部哈希。",
    }


@app.get("/api/customer/profiles")
def customer_profiles(market_code: str = "SD", segment: str | None = None) -> dict[str, Any]:
    require_sample_market(market_code)
    rows = load_private_json("customer_load_profiles_2026h1.json")
    public_rows = [{key: value for key, value in row.items() if key != "customerHash"} for row in rows]
    if segment:
        public_rows = [row for row in public_rows if row.get("segment") == segment]
    return {
        "market_code": market_code.upper(),
        "customer_count": len(public_rows),
        "profiles": public_rows,
        "privacy": "只返回稳定客户代号和统计画像，不返回真实名称、户号、计量点或内部哈希。",
    }


@app.post("/api/forecast/run")
def run_forecast(request: ForecastRunRequest) -> dict[str, Any]:
    raise api_error(503, "FORECAST_PROVIDER_NOT_CONFIGURED", "本地预测模型已移除；请通过统一模型运行接口提交外部算法服务结果")
    market_code = request.market_code.upper()
    results = build_forecast(request.date, request.model_version, market_code)
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(connect()) as connection, connection:
        cursor = connection.execute(
            """
            INSERT INTO forecast_runs
                (market_code, forecast_date, model_version, source_file, point_count, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_code,
                request.date,
                request.model_version,
                "spot_prices_2026h1.json + portfolio_load_hourly_2026h1.json",
                len(results),
                json.dumps(results, ensure_ascii=False),
                created_at,
            ),
        )
        run_id = cursor.lastrowid
    return {"run_id": run_id, "market_code": market_code, "created_at": created_at, "point_count": len(results), "results": results}


@app.get("/api/forecast/results")
def forecast_results(date: str, model_version: str = "lag-baseline-v0.3", market_code: str = "SD") -> dict[str, Any]:
    raise api_error(503, "FORECAST_PROVIDER_NOT_CONFIGURED", "本地预测模型已移除；当前接口仅保留外部结果接入能力")
    normalized_market = market_code.upper()
    require_sample_market(normalized_market)
    with closing(connect()) as connection, connection:
        row = connection.execute(
            """
            SELECT * FROM forecast_runs
            WHERE market_code = ? AND forecast_date = ? AND model_version = ?
            ORDER BY id DESC LIMIT 1
            """,
            (normalized_market, date, model_version),
        ).fetchone()
    if row is None:
        results = build_forecast(date, model_version, normalized_market)
        return {"run_id": None, "market_code": normalized_market, "mode": "preview", "point_count": len(results), "results": results}
    return {
        "run_id": row["id"],
        "market_code": normalized_market,
        "mode": "stored",
        "created_at": row["created_at"],
        "point_count": row["point_count"],
        "results": json.loads(row["result_json"]),
    }


@app.get("/api/data/quality")
def data_quality() -> dict[str, Any]:
    settlement = load_json("settlement_2026-06-21.json")
    quality = load_private_json("data_quality_2026h1.json") if shandong_data_available() else None
    return {
        "datasets": [
            {
                "domain": "客户结算单",
                "source_file": "settlement_2026-06-21.json",
                "point_count": len(settlement.get("hourly", [])),
                "expected_count": 24,
                "status": "complete" if len(settlement.get("hourly", [])) == 24 else "incomplete",
            },
            {
                "domain": "山东现货价格",
                "source_file": "spot_prices_2026h1.json",
                "point_count": quality["datasets"]["spotPrices"]["rowCount"] if quality else 0,
                "expected_count": 4344,
                "status": "partial" if quality and quality["datasets"]["spotPrices"]["missingRealtimeCount"] else "complete",
            },
            {
                "domain": "脱敏组合负荷",
                "source_file": "portfolio_load_hourly_2026h1.json",
                "point_count": 4080 if quality else 0,
                "expected_count": 4344,
                "status": "partial",
            },
        ],
        "quality": quality,
        "privacy": "逐户名称、户号和计量点不通过本接口返回；仅提供脱敏组合负荷。",
    }


@app.get("/api/analytics/market-features")
def market_features(market_code: str = "SD") -> dict[str, Any]:
    require_sample_market(market_code)
    summary = load_private_json("market_feature_summary_2026h1.json")
    return {
        "marketCode": market_code.upper(),
        "mode": "descriptive-baseline",
        "summary": summary,
        "privacy": "只返回市场和客户分群汇总，不返回逐客户画像或名称映射。",
    }


@app.get("/api/retail/contracts")
def retail_contracts(market_code: str = "SD") -> dict[str, Any]:
    require_sample_market(market_code)
    contracts, _ = contract_position_assets()
    customers = []
    for row in contracts.get("customers", []):
        flags = ["PRICING_FORMULA_UNCONFIRMED"]
        if row.get("packageType") == "双边协商零售交易":
            flags.append("BILATERAL_TERMS_REQUIRE_REVIEW")
        if row.get("pricingType") == "TIME_OF_USE_PARAMETER":
            flags.append("TIME_OF_USE_CURVE_MISSING")
        if isinstance(row.get("pricingParameter"), (int, float)) and row["pricingParameter"] < 0:
            flags.append("NEGATIVE_PARAMETER_REQUIRES_BASE_PRICE")
        if row.get("matchStatus") == "CONTRACT_ONLY":
            flags.append("LOAD_NOT_MATCHED")
        customers.append({
            "customerAlias": row.get("customerAlias"),
            "packageType": row.get("packageType"),
            "pricingTermRaw": row.get("pricingTermRaw"),
            "pricingType": row.get("pricingType"),
            "pricingParameter": row.get("pricingParameter"),
            "unit": row.get("unit"),
            "interpretationStatus": row.get("interpretationStatus"),
            "matchStatus": row.get("matchStatus"),
            "riskFlags": flags,
        })
    return {
        "marketCode": market_code.upper(),
        "dataVersion": contracts.get("dataVersion"),
        "generatedAt": contracts.get("generatedAt"),
        "dataStatus": "REAL_PARTIAL",
        "summary": contracts.get("summary", {}),
        "quality": contracts.get("quality", {}),
        "interpretationBoundary": contracts.get("interpretationBoundary"),
        "customers": customers,
        "privacy": "仅返回稳定客户代号；真实客户名称不通过本接口返回。",
    }


@app.get("/api/medium-contract/positions")
def medium_contract_positions(
    market_code: str = "SD",
    month: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    require_sample_market(market_code)
    if month is not None and not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=422, detail="month must use YYYY-MM")
    if date is not None:
        validate_business_date(date)
        if month is not None and not date.startswith(month):
            raise HTTPException(status_code=422, detail="date must belong to month")
    _, positions = contract_position_assets()
    rows = positions.get("records", [])
    if month:
        rows = [row for row in rows if row.get("month") == month]
    if date:
        rows = [row for row in rows if row.get("date") == date]
    if (month or date) and not rows:
        raise HTTPException(status_code=404, detail="No medium/long-term position rows for the requested period")
    return {
        "marketCode": market_code.upper(),
        "dataVersion": positions.get("dataVersion"),
        "generatedAt": positions.get("generatedAt"),
        "dataStatus": "REAL_PARTIAL",
        "filters": {"month": month, "date": date},
        "summary": position_summary(rows),
        "byTradeType": position_trade_type_summary(rows),
        "rows": rows,
        "quality": positions.get("quality", {}),
        "interpretationBoundary": positions.get("interpretationBoundary"),
        "executionAllowed": False,
    }


def monthly_contract_position_row(month: str, position_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in position_records if row.get("month") == month]
    summary = position_summary(rows)
    load_rows = [
        row for row in load_private_json("portfolio_load_daily_2026h1.json")
        if str(row.get("date", "")).startswith(month) and isinstance(row.get("totalMwh"), (int, float))
    ]
    portfolio_load = sum(float(row["totalMwh"]) for row in load_rows) if load_rows else None
    net_position = summary["netPositionMwh"]
    coverage = net_position / portfolio_load if portfolio_load else None
    return {
        "month": month,
        "portfolioLoadMwh": portfolio_load,
        "loadDayCount": len(load_rows),
        "positionDayCount": len({row.get("date") for row in rows}),
        **summary,
        "coverageRate": coverage,
        "residualExposureMwh": portfolio_load - net_position if portfolio_load is not None else None,
        "byTradeType": position_trade_type_summary(rows),
    }


@app.get("/api/analytics/contract-position-overview")
def contract_position_overview(market_code: str = "SD", month: str = "2026-06") -> dict[str, Any]:
    require_sample_market(market_code)
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise HTTPException(status_code=422, detail="month must use YYYY-MM")
    contracts, positions = contract_position_assets()
    position_records = positions.get("records", [])
    available_months = sorted({row.get("month") for row in position_records if row.get("month")})
    if month not in available_months:
        raise HTTPException(status_code=404, detail="No contract-position overview for the requested month")
    monthly_series = [monthly_contract_position_row(item, position_records) for item in available_months]
    selected = next(row for row in monthly_series if row["month"] == month)
    return {
        "marketCode": market_code.upper(),
        "dataVersion": positions.get("dataVersion"),
        "dataStatus": "REAL_PARTIAL",
        "selectedMonth": selected,
        "monthlySeries": monthly_series,
        "contractSummary": contracts.get("summary", {}),
        "quality": {
            "contract": contracts.get("quality", {}),
            "position": positions.get("quality", {}),
        },
        "strategyBoundary": "当前仅支持月度和日级历史分析；缺少分时持仓曲线、目标日负荷预测和正式山东申报规则，不生成可执行报价报量。",
        "executionAllowed": False,
    }


@app.get("/api/analytics/daily-exposure")
def daily_exposure(market_code: str = "SD", date: str = Query(...)) -> dict[str, Any]:
    require_sample_market(market_code)
    validate_business_date(date)
    _, positions = contract_position_assets()
    rows = [row for row in positions.get("records", []) if row.get("date") == date]
    if not rows:
        raise HTTPException(status_code=404, detail="No medium/long-term position rows for the requested date")
    summary = position_summary(rows)
    portfolio_load = portfolio_load_for_date(date)
    net_position = summary["netPositionMwh"]
    coverage = net_position / portfolio_load if portfolio_load else None
    residual = portfolio_load - net_position if portfolio_load is not None else None
    prices = volume_weighted_spot_prices(date)
    scenarios: dict[str, Any] = {
        "basis": "历史组合负荷加权实际价格情景；不是实际结算成本，不包含偏差、分摊或零售收入。",
        "dayAhead": None,
        "realTime": None,
    }
    for key, price_key in [("dayAhead", "dayAheadWeightedPriceYuanMwh"), ("realTime", "realTimeWeightedPriceYuanMwh")]:
        price = prices[price_key]
        if residual is not None and residual >= 0 and price is not None:
            remaining_cost = residual * price
            scenarios[key] = {
                "weightedPriceYuanMwh": price,
                "residualExposureCostYuan": remaining_cost,
                "totalPurchaseCostScenarioYuan": summary["positivePositionCostYuan"] + remaining_cost,
            }
    signals = []
    if coverage is None:
        signals.append({"code": "DATA_INSUFFICIENT", "label": "数据不足", "detail": "当日组合负荷缺失，不能计算覆盖率和剩余敞口。"})
    elif coverage < 0.35:
        signals.append({"code": "LOW_COVERAGE_ATTENTION", "label": "覆盖偏低关注", "detail": "当日净持仓覆盖率低于35%的历史分析关注线。"})
    if summary["negativePositionMwh"] < 0:
        signals.append({"code": "NEGATIVE_POSITION_REVIEW", "label": "负持仓待核验", "detail": "负持仓按调减或冲销待确认量保留，未推断卖出收益。"})
    comparable_prices = [prices[key] for key in ["dayAheadWeightedPriceYuanMwh", "realTimeWeightedPriceYuanMwh"] if prices[key] is not None]
    if comparable_prices and summary["positiveWeightedPriceYuanMwh"] is not None and summary["positiveWeightedPriceYuanMwh"] > max(comparable_prices):
        signals.append({"code": "POSITION_COST_ATTENTION", "label": "持仓成本关注", "detail": "正持仓加权价高于当日日前和实时负荷加权价格。"})
    signals.append({"code": "DATA_INSUFFICIENT_FOR_BIDDING", "label": "数据不足", "detail": "缺少分时持仓曲线和正式申报规则，不生成24点或96点报价报量。"})
    return {
        "marketCode": market_code.upper(),
        "date": date,
        "dataVersion": positions.get("dataVersion"),
        "dataStatus": "REAL_PARTIAL" if portfolio_load is not None else "PARTIAL_MISSING_LOAD",
        "portfolioLoadMwh": portfolio_load,
        **summary,
        "coverageRate": coverage,
        "residualExposureMwh": residual,
        "prices": prices,
        "costScenarios": scenarios,
        "signals": signals,
        "byTradeType": position_trade_type_summary(rows),
        "executionAllowed": False,
        "strategyBoundary": "仅供历史复盘和人工复核，不可直接申报。",
    }


@app.get("/api/analytics/loss-analysis", include_in_schema=False)
@app.get("/api/analytics/profit-loss-analysis")
def loss_analysis(
    date: str = Query(...),
    market_code: str = "SD",
    revenue_price: float | None = Query(default=None, ge=0),
    other_cost: float = Query(default=0, ge=0),
) -> dict[str, Any]:
    return build_loss_analysis(date, market_code, revenue_price, other_cost)


@app.get("/api/analytics/customer-profit-contribution")
def customer_profit_contribution(
    date: str = Query(...),
    market_code: str = "SD",
    revenue_price: float | None = Query(default=None, ge=0),
    other_cost: float = Query(default=0, ge=0),
) -> dict[str, Any]:
    return build_customer_profit_contribution(date, market_code, revenue_price, other_cost)


@app.get("/api/integration/status")
def integration_status() -> dict[str, Any]:
    return {
        "contractVersion": INTEGRATION_CONTRACT_VERSION,
        "workflow": ["data-quality", "forecast", "spot-judgement", "manual-review", "execution", "report"],
        "providers": [
            {"code": "platform", "name": "公司交易辅助平台", "status": "connected", "owner": "陆璟行"},
            {"code": "agent-box", "name": "预测与策略算法", "status": "contract-ready", "owner": "王伊梵"},
            {"code": "trading-terminal", "name": "交易模型与执行终端", "status": "contract-ready", "owner": "任怡铭"},
            {"code": "policy-rag", "name": "政策知识库与问答", "status": "adapter-ready", "owner": "任怡铭"},
        ],
        "automationEnabled": False,
        "note": "当前只接受辅助决策结果，所有策略均需人工复核后才能进入执行系统。",
    }


@app.get("/api/integration/contracts")
def integration_contracts() -> dict[str, Any]:
    return {
        "version": INTEGRATION_CONTRACT_VERSION,
        "canonicalKeys": ["marketCode", "businessDate", "timePoint", "importBatchId"],
        "contracts": [
            {
                "name": "forecast-result",
                "endpoint": "/api/integration/forecast-results",
                "method": "POST",
                "batchFields": ["marketCode", "businessDate", "intervalMinutes", "modelVersion", "predictionBatchId", "featureBatchId", "sourceSystem"],
                "pointFields": ["timePoint", "predictedPrice", "lower", "upper", "spreadDirection", "confidence", "drivers", "recommendation"],
            },
            {
                "name": "strategy-result",
                "endpoint": "/api/integration/strategy-results",
                "method": "POST",
                "batchFields": ["marketCode", "businessDate", "intervalMinutes", "strategyVersion", "strategyBatchId", "predictionBatchId", "sourceSystem"],
                "pointFields": ["timePoint", "action", "suggestedEnergyMwh", "suggestedPriceYuanMwh", "confidence", "reason", "requiresManualReview"],
            },
            {
                "name": "source-adapter",
                "endpoint": "/api/adapters/normalize",
                "method": "POST",
                "domains": CANONICAL_REQUIRED_FIELDS,
            },
            {
                "name": "policy-document-registration",
                "endpoint": "/api/policy/documents",
                "method": "POST",
                "note": "登记已获授权的规则文件；知识治理与正式发布仍由独立 Agent 服务负责。",
            },
            {
                "name": "policy-answer",
                "endpoint": "/api/policy-agent/query",
                "method": "POST",
                "inputFields": ["question", "marketCode", "marketHint", "conversationHistory"],
                "outputFields": ["answer", "grounded", "citations", "quality_gate", "coverage", "refusal", "release_id"],
                "note": "只读严格引用代理；禁用联网即时证据、自动补库和发布写操作。",
            },
        ],
    }


def policy_agent_error(error: PolicyAgentError) -> HTTPException:
    if error.unavailable:
        return api_error(503, "POLICY_AGENT_UNAVAILABLE", str(error))
    return api_error(502, "POLICY_AGENT_UPSTREAM_ERROR", str(error), upstreamStatus=error.status_code)


@app.get("/api/policy-agent/health")
def policy_agent_health() -> dict[str, Any]:
    try:
        return PolicyAgentClient().health()
    except PolicyAgentError as error:
        raise policy_agent_error(error) from error


@app.get("/api/policy-agent/capabilities")
def policy_agent_capabilities() -> dict[str, Any]:
    try:
        return PolicyAgentClient().capabilities()
    except PolicyAgentError as error:
        raise policy_agent_error(error) from error


@app.get("/api/policy-agent/active-release")
def policy_agent_active_release() -> dict[str, Any]:
    try:
        return PolicyAgentClient().active_release()
    except PolicyAgentError as error:
        raise policy_agent_error(error) from error


@app.get("/api/policy-agent/supported-scope")
def policy_agent_supported_scope() -> dict[str, Any]:
    try:
        return PolicyAgentClient().supported_scope()
    except PolicyAgentError as error:
        raise policy_agent_error(error) from error


@app.post("/api/policy-agent/query")
def policy_agent_query(request: PolicyAgentQueryRequest) -> dict[str, Any]:
    profile = market_profile(request.marketCode)
    try:
        return PolicyAgentClient().query(
            request.question.strip(),
            region_hint=profile["name"],
            market_hint=request.marketHint.strip(),
            conversation_history=[turn.model_dump() for turn in request.conversationHistory],
        )
    except PolicyAgentError as error:
        raise policy_agent_error(error) from error


@app.post("/api/integration/forecast-results")
def import_forecast_results(request: ForecastResultImportRequest) -> dict[str, Any]:
    market_code = request.marketCode.upper()
    market_profile(market_code)
    validate_business_date(request.businessDate)
    quality = result_quality([point.timePoint for point in request.points], request.intervalMinutes)
    for point in request.points:
        if not 0 <= point.confidence <= 1:
            raise HTTPException(status_code=422, detail=f"confidence must be between 0 and 1 at {point.timePoint}")
        if point.lower is not None and point.predictedPrice < point.lower:
            raise HTTPException(status_code=422, detail=f"predictedPrice is below lower at {point.timePoint}")
        if point.upper is not None and point.predictedPrice > point.upper:
            raise HTTPException(status_code=422, detail=f"predictedPrice is above upper at {point.timePoint}")
    payload = request.model_dump()
    payload["marketCode"] = market_code
    payload["quality"] = quality
    result_id, created_at = store_integration_result(
        "forecast",
        market_code,
        request.businessDate,
        request.modelVersion,
        request.predictionBatchId,
        request.sourceSystem,
        quality,
        payload,
    )
    return {
        "resultId": result_id,
        "accepted": True,
        "createdAt": created_at,
        "quality": quality,
        "nextStep": "spot-judgement" if quality["usableForStrategy"] else "complete-missing-points",
    }


@app.post("/api/integration/strategy-results")
def import_strategy_results(request: StrategyResultImportRequest) -> dict[str, Any]:
    market_code = request.marketCode.upper()
    market_profile(market_code)
    validate_business_date(request.businessDate)
    quality = result_quality([point.timePoint for point in request.points], request.intervalMinutes)
    for point in request.points:
        if not 0 <= point.confidence <= 1:
            raise HTTPException(status_code=422, detail=f"confidence must be between 0 and 1 at {point.timePoint}")
        if not point.requiresManualReview:
            raise HTTPException(status_code=409, detail="Current platform mode requires manual review for every strategy point")
    payload = request.model_dump()
    payload["marketCode"] = market_code
    payload["quality"] = quality
    result_id, created_at = store_integration_result(
        "strategy",
        market_code,
        request.businessDate,
        request.strategyVersion,
        request.strategyBatchId,
        request.sourceSystem,
        quality,
        payload,
    )
    return {
        "resultId": result_id,
        "accepted": True,
        "createdAt": created_at,
        "quality": quality,
        "executionAllowed": False,
        "nextStep": "manual-review",
    }


@app.get("/api/integration/results")
def integration_results(
    result_type: str,
    market_code: str,
    business_date: str,
) -> dict[str, Any]:
    if result_type not in {"forecast", "strategy"}:
        raise HTTPException(status_code=422, detail="result_type must be forecast or strategy")
    normalized_market = market_code.upper()
    market_profile(normalized_market)
    validate_business_date(business_date)
    with closing(connect()) as connection, connection:
        rows = connection.execute(
            """
            SELECT id, result_type, market_code, business_date, version, batch_id, source_system,
                   point_count, quality_status, payload_json, created_at
            FROM integration_results
            WHERE result_type = ? AND market_code = ? AND business_date = ?
            ORDER BY created_at DESC
            """,
            (result_type, normalized_market, business_date),
        ).fetchall()
    return {
        "resultType": result_type,
        "marketCode": normalized_market,
        "businessDate": business_date,
        "count": len(rows),
        "results": [
            {
                "resultId": row["id"],
                "version": row["version"],
                "batchId": row["batch_id"],
                "sourceSystem": row["source_system"],
                "pointCount": row["point_count"],
                "qualityStatus": row["quality_status"],
                "createdAt": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ],
    }


@app.post("/api/adapters/normalize")
def normalize_adapter_records(request: AdapterNormalizeRequest) -> dict[str, Any]:
    market_code = request.marketCode.upper()
    market_profile(market_code)
    required_fields = CANONICAL_REQUIRED_FIELDS.get(request.dataDomain)
    if required_fields is None:
        raise HTTPException(status_code=422, detail=f"Unsupported dataDomain: {request.dataDomain}")
    if not request.records:
        raise HTTPException(status_code=422, detail="records must not be empty")
    if len(request.records) > 5000:
        raise HTTPException(status_code=413, detail="One adapter request can contain at most 5000 records")

    sensitive_targets = {"customerName", "accountNumber", "meterPoint"}
    normalized_rows = []
    validation_errors = []
    for index, source_row in enumerate(request.records):
        normalized = {
            "marketCode": market_code,
            "importBatchId": request.importBatchId,
            "sourceSystem": request.sourceSystem,
        }
        for source_field, canonical_field in request.fieldMapping.items():
            if canonical_field in sensitive_targets:
                continue
            normalized[canonical_field] = source_row.get(source_field)
        missing = [field for field in required_fields if normalized.get(field) is None or normalized.get(field) == ""]
        if missing:
            validation_errors.append({"row": index + 1, "missingFields": missing})
        normalized_rows.append(normalized)
    return {
        "accepted": not validation_errors,
        "marketCode": market_code,
        "dataDomain": request.dataDomain,
        "recordCount": len(normalized_rows),
        "requiredFields": required_fields,
        "validationErrors": validation_errors[:100],
        "privacy": "customerName、accountNumber、meterPoint 不在适配器响应中返回；应在受控数据服务中映射为稳定客户代号。",
        "records": normalized_rows,
    }


@app.post("/api/policy/documents")
def register_policy_document(request: PolicyDocumentRegistration) -> dict[str, Any]:
    market_code = request.marketCode.upper()
    market_profile(market_code)
    if request.effectiveDate:
        validate_business_date(request.effectiveDate)
    if not re.fullmatch(r"[A-Fa-f0-9]{64}", request.checksumSha256):
        raise HTTPException(status_code=422, detail="checksumSha256 must contain 64 hexadecimal characters")
    if not request.sourceAuthorized:
        raise HTTPException(status_code=409, detail="Policy document source authorization must be confirmed before registration")
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO policy_documents
                (document_id, market_code, title, version, effective_date, storage_uri,
                 checksum_sha256, source_authorized, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                market_code = excluded.market_code,
                title = excluded.title,
                version = excluded.version,
                effective_date = excluded.effective_date,
                storage_uri = excluded.storage_uri,
                checksum_sha256 = excluded.checksum_sha256,
                source_authorized = excluded.source_authorized,
                status = excluded.status,
                created_at = excluded.created_at
            """,
            (
                request.documentId,
                market_code,
                request.title,
                request.version,
                request.effectiveDate,
                request.storageUri,
                request.checksumSha256.lower(),
                1,
                "registered-rag-pending",
                created_at,
            ),
        )
    return {
        "documentId": request.documentId,
        "marketCode": market_code,
        "status": "registered-rag-pending",
        "createdAt": created_at,
        "nextStep": "policy-rag-indexing-provider-pending",
    }


@app.get("/api/policy/documents")
def policy_documents(market_code: str | None = None) -> dict[str, Any]:
    query = "SELECT document_id, market_code, title, version, effective_date, status, created_at FROM policy_documents"
    parameters: tuple[Any, ...] = ()
    if market_code:
        normalized_market = market_code.upper()
        market_profile(normalized_market)
        query += " WHERE market_code = ?"
        parameters = (normalized_market,)
    query += " ORDER BY created_at DESC"
    with closing(connect()) as connection, connection:
        rows = connection.execute(query, parameters).fetchall()
    return {"count": len(rows), "documents": [dict(row) for row in rows], "ragProviderConnected": False}


@app.get("/api/v1/health")
def health_v1() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "power-trading-platform-api",
        "api_version": API_VERSION,
        "environment": SERVICE_ENVIRONMENT,
        "timezone": "Asia/Shanghai",
        "authentication": {
            "mode": "api_key" if V1_API_KEY else "none_test_only",
            "header": "X-API-Key",
            "production_requirement": "POWER_TRADING_API_KEY must be configured",
        },
        "limits": {
            "price_upload_format": "application/json",
            "price_upload_max_bytes": MAX_PRICE_UPLOAD_BYTES,
            "price_upload_max_records": MAX_PRICE_UPLOAD_RECORDS,
            "default_run_timeout_seconds": DEFAULT_RUN_TIMEOUT_SECONDS,
            "max_run_timeout_seconds": MAX_RUN_TIMEOUT_SECONDS,
            "run_retention_days": RUN_RETENTION_DAYS,
        },
        "execution_allowed": False,
    }


v1 = APIRouter(prefix="/api/v1", dependencies=[Depends(require_v1_api_key)])


@v1.get("/models")
def models_v1() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "models": [
            {
                "id": "lag-baseline",
                "name": "历史时序回测基线",
                "status": "interface_only",
                "versions": [],
                "run_mode": "synchronous",
                "output_contract": "forecast-strategy-contract-v1",
            },
            {
                "id": "price-forecast",
                "name": "山东现货价格概率集成预测",
                "status": "connected_local",
                "versions": ["integrated-price-forecast-v2.0.0"],
                "run_mode": "synchronous_local",
                "result_callback": "/api/v1/model-runs/{run_id}/results",
                "output_contract": "forecast-strategy-contract-v1",
            },
        ],
        "note": "已接入 wangyifan-111/- 一体化价格预测协议；使用平台现有山东脱敏数据运行，结果仍需人工复核。",
    }


@v1.get("/weather/hourly")
def weather_hourly_v1(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    return weather_hourly_response(market_code, start_date, end_date, date)


@v1.get("/market-supply/hourly")
def market_supply_hourly_v1(
    market_code: str = "SD",
    start_date: str | None = None,
    end_date: str | None = None,
    date: str | None = None,
    source_type: str | None = None,
) -> dict[str, Any]:
    return market_supply_hourly_response(market_code, start_date, end_date, date, source_type)


@v1.get("/data-assets")
def data_assets_v1(market_code: str = "SD") -> dict[str, Any]:
    normalized_market = market_code.upper()
    market_profile(normalized_market)
    quality = load_private_json("data_quality_2026h1.json") if normalized_market == "SD" and shandong_data_available() else None
    assets = [
        {"domain": "prices", "status": "REAL_CONNECTED", "coverage": "2026-01-01/2026-06-30", "data_version": "sd-spot-2026h1-v1", "notes": "4344点；2026-02-12实时价格缺24点"},
        {"domain": "load_actual", "status": "REAL_CONNECTED", "coverage": "170个有效日", "data_version": "sd-portfolio-load-2026h1-v1", "notes": "组合负荷及125家稳定脱敏客户负荷"},
        {"domain": "load_forecast", "status": "REAL_PARTIAL", "coverage": "2026-04-01/2026-06-30历史预测", "data_version": "sd-system-output-hourly-2026h1-v1", "notes": "已接入直调负荷预测历史；7月1日目标日预测和发布时间待补，统一小时接口当前仍返回null"},
        {"domain": "weather", "status": "REAL_PARTIAL", "coverage": "2026-05-01/2026-07-01", "data_version": "sd-weather-gfs-hourly-20260501-20260701-v2", "notes": "1488个完整小时点；业务方确认均在申报前可得，已进入v1.2逐日滚动训练与回测；精确forecast_issue_time仍待补"},
        {"domain": "renewable_forecast", "status": "REAL_PARTIAL", "coverage": "2026-04-01/2026-06-30历史预测", "data_version": "sd-system-output-hourly-2026h1-v1", "notes": "已接入风电、光伏96点预测并聚合至24点；当前仅使用D-1/D-7滞后，目标日预测及发布时间待补"},
        {"domain": "market_supply_history", "status": "REAL_PARTIAL", "coverage": "2026-01-01/2026-06-30", "data_version": "sd-system-output-hourly-2026h1-v1", "notes": "Q1实际出力与Q2预测出力已分口径接入；Q2抽蓄全空，残余需求字段不是正式竞价空间"},
        {"domain": "medium_long_term_positions", "status": "REAL_PARTIAL", "coverage": "2026-01-01/2026-06-30", "data_version": "sd-contract-position-2026h1-v1", "notes": "647条日级持仓；缺24/96点曲线，负持仓和0价口径待确认"},
        {"domain": "retail_contracts", "status": "REAL_PARTIAL", "coverage": "128家套餐", "data_version": "sd-contract-position-2026h1-v1", "notes": "125家匹配负荷；价格参数计价公式、合同电量和偏差责任待确认"},
        {"domain": "declarations_and_clearing", "status": "MISSING", "coverage": None, "data_version": None, "notes": "日前/实时申报、成交价量待接入"},
        {"domain": "deviation_assessment", "status": "MISSING", "coverage": None, "data_version": None, "notes": "偏差责任和考核规则待接入"},
        {"domain": "settlement", "status": "REAL_PARTIAL", "coverage": "2026-06-21单日案例", "data_version": "settlement-case-001-v1", "notes": "不能代表完整财务利润"},
        {"domain": "unit_outage_congestion", "status": "MISSING", "coverage": None, "data_version": None, "notes": "机组状态、检修和阻塞信息待接入"},
        {"domain": "market_rules", "status": "MISSING", "coverage": None, "data_version": None, "notes": "山东市场规则版本待确认"},
    ]
    with closing(connect()) as connection, connection:
        uploaded = connection.execute(
            """
            SELECT asset_id, domain, data_version, source_system, record_count, quality_status, updated_at, created_at
            FROM data_asset_batches WHERE market_code = ? ORDER BY created_at DESC
            """,
            (normalized_market,),
        ).fetchall()
    return {
        "market_code": normalized_market,
        "generated_at": utc_now(),
        "assets": assets if normalized_market == "SD" else [dict(item, status="MISSING", coverage=None, data_version=None) for item in assets],
        "uploaded_assets": [dict(row) for row in uploaded],
        "quality_snapshot": quality,
        "truth_labels": ["REAL_CONNECTED", "REAL_PARTIAL", "MISSING", "STATIC_UI_ONLY"],
    }


@v1.get("/data-dictionary")
def data_dictionary_v1() -> dict[str, Any]:
    return {
        "version": API_VERSION,
        "primary_key": ["market_code", "market_date", "period", "customer_id", "data_version"],
        "null_rule": "Unknown or unavailable values are null; zero is only used for measured zero.",
        "fields": [
            {"name": "market_date", "type": "date", "unit": None, "time_basis": "交易日", "nullable": False},
            {"name": "period", "type": "integer", "unit": "1-24", "time_basis": "period 1 starts at 00:00", "nullable": False},
            {"name": "datetime", "type": "RFC3339 timestamp", "unit": None, "time_basis": "Asia/Shanghai interval start", "nullable": False},
            {"name": "day_ahead_price_yuan_per_mwh", "type": "number", "unit": "CNY/MWh", "time_basis": "hourly", "nullable": True},
            {"name": "real_time_price_yuan_per_mwh", "type": "number", "unit": "CNY/MWh", "time_basis": "hourly", "nullable": True},
            {"name": "load_actual_mwh", "type": "number", "unit": "MWh", "time_basis": "hourly energy", "nullable": True},
            {"name": "load_forecast_mwh", "type": "number", "unit": "MWh", "time_basis": "hourly energy", "nullable": True},
            {"name": "temperature_2m_c", "type": "number", "unit": "°C", "time_basis": "hourly forecast", "nullable": True},
            {"name": "wind_speed_10m_mps", "type": "number", "unit": "m/s", "time_basis": "hourly forecast", "nullable": True},
            {"name": "wind_speed_100m_mps", "type": "number", "unit": "m/s", "time_basis": "hourly forecast", "nullable": True},
            {"name": "shortwave_radiation_ghi_w_per_m2", "type": "number", "unit": "W/m²", "time_basis": "hourly forecast", "nullable": True},
            {"name": "cloud_cover_pct", "type": "number", "unit": "%", "time_basis": "hourly forecast", "nullable": True},
            {"name": "precipitation_mm", "type": "number", "unit": "mm", "time_basis": "hourly forecast", "nullable": True},
            {"name": "relative_humidity_pct", "type": "number", "unit": "%", "time_basis": "hourly forecast", "nullable": True},
            {"name": "weather_forecast_issue_time", "type": "RFC3339 timestamp/null", "unit": None, "time_basis": "forecast publication time", "nullable": True, "rule": "recommended for exact audit; current weather batch uses business-confirmed pre-declaration availability"},
            {"name": "source_type", "type": "ACTUAL|FORECAST", "unit": None, "time_basis": "market supply data definition", "nullable": True, "rule": "Q1 actual and Q2 forecast must not be merged as the same feature"},
            {"name": "direct_dispatch_load_mw", "type": "number", "unit": "MW", "time_basis": "hourly mean of four 15-minute points", "nullable": True},
            {"name": "interconnector_mw", "type": "number", "unit": "MW", "time_basis": "hourly mean of four 15-minute points", "nullable": True},
            {"name": "wind_mw", "type": "number", "unit": "MW", "time_basis": "hourly mean of four 15-minute points", "nullable": True},
            {"name": "solar_mw", "type": "number", "unit": "MW", "time_basis": "hourly mean of four 15-minute points", "nullable": True},
            {"name": "fixed_output_proxy_mw", "type": "number", "unit": "MW", "time_basis": "hourly derived proxy", "nullable": True},
            {"name": "residual_demand_proxy_mw", "type": "number", "unit": "MW", "time_basis": "hourly derived proxy", "nullable": True, "rule": "not formal bidding space; pumped storage and conventional hydro are incomplete"},
            {"name": "supply_forecast_issue_time", "type": "RFC3339 timestamp/null", "unit": None, "time_basis": "forecast publication time", "nullable": True, "rule": "currently missing; v1.2 only uses D-1/D-7 lagged forecast rows"},
            {"name": "customer_id", "type": "string", "unit": None, "time_basis": None, "nullable": True, "rule": "stable alias such as 客户-001; null means portfolio"},
            {"name": "data_version", "type": "string", "unit": None, "time_basis": None, "nullable": False},
            {"name": "updated_at", "type": "RFC3339 timestamp", "unit": None, "time_basis": "source update time", "nullable": False},
        ],
    }


@v1.get("/hourly-data")
def hourly_data_endpoint_v1(
    market_date: str = Query(...),
    market_code: str = "SD",
    customer_id: str | None = None,
    data_version: str | None = None,
) -> dict[str, Any]:
    return unified_hourly_data_v1(market_date, market_code, customer_id, data_version)


@v1.post("/data-assets/prices", status_code=202)
def upload_prices_v1(request: PriceAssetUploadRequestV1) -> dict[str, Any]:
    response, _ = store_price_asset_v1(request)
    return response


@v1.post("/models/price-forecast/runs", status_code=202)
def create_price_forecast_run_v1(request: ModelRunCreateRequestV1) -> dict[str, Any]:
    response, replay = create_model_run(request, actor="api-client")
    response["idempotent_replay"] = replay
    if response["status"] == "PENDING_PROVIDER":
        response["provider_callback"] = f"/api/v1/model-runs/{response['run_id']}/results"
    return response


@v1.put("/model-runs/{run_id}/results")
def submit_model_run_results_v1(run_id: str, result: ForecastStrategyResultV1) -> dict[str, Any]:
    row = refresh_run_timeout(run_id)
    if row["status"] not in {"PENDING_PROVIDER", "RUNNING"}:
        raise api_error(409, "RUN_NOT_ACCEPTING_RESULTS", f"Run status {row['status']} cannot accept results")
    ready, available, missing = strategy_readiness(json.loads(row["input_summary_json"]))
    if result.strategy_ready != ready:
        raise api_error(409, "STRATEGY_READINESS_MISMATCH", "strategy_ready must be determined from the platform input snapshot", expected=ready)
    if sorted(result.data_snapshot.available_domains) != available or sorted(result.data_snapshot.missing_domains) != missing:
        raise api_error(409, "DATA_SNAPSHOT_MISMATCH", "Result data snapshot domains must match the stored input summary")
    return complete_model_run(run_id, result, "algorithm-provider")


@v1.get("/model-runs/{run_id}")
def model_run_status_v1(run_id: str) -> dict[str, Any]:
    return public_model_run(refresh_run_timeout(run_id))


@v1.get("/model-runs/{run_id}/results")
def model_run_results_v1(run_id: str) -> dict[str, Any]:
    row = refresh_run_timeout(run_id)
    if row["status"] != "SUCCEEDED" or row["result_json"] is None:
        raise api_error(409, "RUN_NOT_COMPLETE", f"Run status is {row['status']}; results are not available")
    return {
        "run": public_model_run(row),
        "result": json.loads(row["result_json"]),
        "review": {
            "required": True,
            "status": row["review_status"],
            "execution_allowed": False,
        },
    }


@v1.get("/model-runs")
def model_runs_v1(
    market_code: str | None = None,
    market_date: str | None = None,
    model_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    purge_expired_runs()
    clauses: list[str] = []
    parameters: list[Any] = []
    for column, value in [("market_code", market_code.upper() if market_code else None), ("market_date", market_date), ("model_id", model_id), ("status", status)]:
        if value:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    query = "SELECT run_id FROM model_runs_v1"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ?"
    parameters.append(limit)
    with closing(connect()) as connection, connection:
        run_ids = [row["run_id"] for row in connection.execute(query, parameters).fetchall()]
    rows = [public_model_run(refresh_run_timeout(run_id)) for run_id in run_ids]
    return {"count": len(rows), "retention_days": RUN_RETENTION_DAYS, "runs": rows}


@v1.post("/model-runs/{run_id}/cancel")
def cancel_model_run_v1(run_id: str, request: RunActionRequestV1) -> dict[str, Any]:
    row = refresh_run_timeout(run_id)
    if row["status"] not in {"QUEUED", "RUNNING", "PENDING_PROVIDER"}:
        raise api_error(409, "RUN_NOT_CANCELLABLE", f"Run status {row['status']} cannot be cancelled")
    now = utc_now()
    with closing(connect()) as connection, connection:
        connection.execute(
            "UPDATE model_runs_v1 SET status = 'CANCELLED', completed_at = ?, updated_at = ? WHERE run_id = ?",
            (now, now, run_id),
        )
    write_run_audit(run_id, "RUN_CANCELLED", request.actor, {"reason": request.reason})
    return public_model_run(get_model_run_row(run_id))


@v1.post("/model-runs/{run_id}/rerun", status_code=202)
def rerun_model_v1(run_id: str, request: RerunRequestV1) -> dict[str, Any]:
    row = get_model_run_row(run_id)
    original = json.loads(row["request_json"])
    original["request_id"] = request.request_id
    new_request = ModelRunCreateRequestV1(**original)
    response, replay = create_model_run(new_request, parent_run_id=run_id, actor=request.actor)
    response["idempotent_replay"] = replay
    response["rerun_reason"] = request.reason
    return response


@v1.post("/model-runs/{run_id}/review")
def review_strategy_v1(run_id: str, request: StrategyReviewRequestV1) -> dict[str, Any]:
    row = get_model_run_row(run_id)
    if row["status"] != "SUCCEEDED" or row["result_json"] is None:
        raise api_error(409, "RUN_NOT_REVIEWABLE", "Only a succeeded run can enter strategy review")
    transitions = {
        ("DRAFT", "SUBMIT"): "PENDING_REVIEW",
        ("PENDING_REVIEW", "APPROVE"): "APPROVED",
        ("PENDING_REVIEW", "MODIFY"): "MODIFIED",
        ("PENDING_REVIEW", "REJECT"): "REJECTED",
    }
    target = transitions.get((row["review_status"], request.action))
    if target is None:
        raise api_error(409, "INVALID_REVIEW_TRANSITION", f"Cannot apply {request.action} from {row['review_status']}")
    if request.action in {"MODIFY", "REJECT"} and not request.reason:
        raise api_error(422, "REVIEW_REASON_REQUIRED", f"{request.action} requires a reason")
    if request.action == "MODIFY" and not request.modified_suggestions:
        raise api_error(422, "MODIFIED_SUGGESTIONS_REQUIRED", "MODIFY requires modified_suggestions")

    original_strategy = json.loads(row["current_strategy_json"] or "[]")
    modified_strategy = original_strategy
    if request.action == "MODIFY":
        result = json.loads(row["result_json"])
        if not result.get("strategy_ready"):
            invalid = [item for item in request.modified_suggestions or [] if str(item.get("action", "")).upper() != "HOLD"]
            if invalid:
                raise api_error(409, "STRATEGY_INPUTS_INCOMPLETE", "Modified suggestions must remain HOLD while strategy inputs are incomplete")
        by_period = {int(item["period"]): dict(item) for item in original_strategy}
        for update in request.modified_suggestions or []:
            period = int(update.get("period", 0))
            if period not in by_period:
                raise api_error(422, "INVALID_PERIOD", f"Unknown modified suggestion period: {period}")
            by_period[period].update(update)
            by_period[period]["action"] = str(by_period[period].get("action", "HOLD")).upper()
        modified_strategy = [by_period[period] for period in sorted(by_period)]

    now = utc_now()
    with closing(connect()) as connection, connection:
        connection.execute(
            "UPDATE model_runs_v1 SET review_status = ?, current_strategy_json = ?, updated_at = ? WHERE run_id = ?",
            (target, json.dumps(modified_strategy, ensure_ascii=False), now, run_id),
        )
        connection.execute(
            """
            INSERT INTO strategy_reviews_v1
                (run_id, from_status, to_status, reviewer, reason, original_json, modified_json,
                 model_version, data_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["review_status"],
                target,
                request.reviewer,
                request.reason,
                json.dumps(original_strategy, ensure_ascii=False),
                json.dumps(modified_strategy, ensure_ascii=False),
                row["model_version"],
                row["data_version"],
                now,
            ),
        )
    write_run_audit(run_id, f"REVIEW_{request.action}", request.reviewer, {"from": row["review_status"], "to": target, "reason": request.reason})
    return {
        "run_id": run_id,
        "from_status": row["review_status"],
        "review_status": target,
        "reviewer": request.reviewer,
        "reviewed_at": now,
        "reason": request.reason,
        "model_version": row["model_version"],
        "data_version": row["data_version"],
        "execution_allowed": False,
    }


@v1.get("/model-runs/{run_id}/reviews")
def strategy_reviews_v1(run_id: str) -> dict[str, Any]:
    get_model_run_row(run_id)
    with closing(connect()) as connection, connection:
        rows = connection.execute("SELECT * FROM strategy_reviews_v1 WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
    return {
        "run_id": run_id,
        "count": len(rows),
        "reviews": [
            {
                "review_id": row["id"],
                "from_status": row["from_status"],
                "to_status": row["to_status"],
                "reviewer": row["reviewer"],
                "reason": row["reason"],
                "original_suggestions": json.loads(row["original_json"] or "[]"),
                "modified_suggestions": json.loads(row["modified_json"] or "[]"),
                "model_version": row["model_version"],
                "data_version": row["data_version"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
    }


@v1.get("/model-runs/{run_id}/audit-logs")
def run_audit_logs_v1(run_id: str) -> dict[str, Any]:
    get_model_run_row(run_id)
    with closing(connect()) as connection, connection:
        rows = connection.execute("SELECT * FROM run_audit_logs_v1 WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
    return {
        "run_id": run_id,
        "count": len(rows),
        "logs": [
            {
                "log_id": row["id"],
                "event_type": row["event_type"],
                "actor": row["actor"],
                "details": json.loads(row["details_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ],
    }


app.include_router(v1)



