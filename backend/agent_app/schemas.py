from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TrimmedTextModel(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def trim_strings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"
    NEEDS_ONBOARDING = "NEEDS_ONBOARDING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_RUN_STATUSES = {
    RunStatus.APPROVED.value,
    RunStatus.MODIFIED.value,
    RunStatus.REJECTED.value,
    RunStatus.NEEDS_ONBOARDING.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
}


class ReviewStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class AgentRunCreate(TrimmedTextModel):
    request_id: str = Field(min_length=1, max_length=128)
    market_code: str = Field(min_length=2, max_length=8)
    trading_subject: str = Field(min_length=1, max_length=64)
    business_date: date
    initiated_by: str = Field(min_length=1, max_length=128)
    strategy_version: Literal[
        "historical_cvar_v02", "regime_cvar_v03", "similar_day_cvar_v04",
        "advanced_cvar_v05", "joint_cvar_v06",
    ] = "historical_cvar_v02"
    risk_aversion: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("risk_aversion", mode="before")
    @classmethod
    def risk_is_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("risk_aversion 必须是数字而非布尔值")
        return value


class TradingDraftRunCreate(TrimmedTextModel):
    market_code: Literal["SD"] = "SD"
    trading_subject: Literal["retail"] = "retail"
    granularity: Literal["HOUR_24"] = "HOUR_24"
    run_type: Literal["DAY_AHEAD_DRAFT"] = "DAY_AHEAD_DRAFT"
    request_id: str = Field(min_length=1, max_length=128)
    business_date: date
    initiated_by: str = Field(min_length=1, max_length=128)
    forecast_version: str | None = Field(default=None, max_length=128)
    rule_version: str | None = Field(default=None, max_length=128)
    strategy_version: Literal[
        "historical_cvar_v02", "regime_cvar_v03", "legacy_spread_v01"
    ] = "historical_cvar_v02"
    risk_aversion: float = Field(default=0.3, ge=0.0, le=1.0)
    scenario_source: str | None = Field(default=None, max_length=255)
    scenario_version: str | None = Field(default=None, max_length=128)

    @field_validator("risk_aversion", mode="before")
    @classmethod
    def risk_is_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("risk_aversion 必须是数字而非布尔值")
        return value


class TradingResearchRunCreate(TradingDraftRunCreate):
    run_type: Literal["TRADING_RESEARCH"] = "TRADING_RESEARCH"
    strategy_version: Literal[
        "historical_cvar_v02", "regime_cvar_v03", "similar_day_cvar_v04",
        "advanced_cvar_v05", "joint_cvar_v06",
    ] = "historical_cvar_v02"
    forecast_version: None = None
    rule_version: None = None
    scenario_source: None = None
    scenario_version: None = None


class TradingDraftReview(TrimmedTextModel):
    expected_revision: int = Field(ge=0, strict=True)
    decision: Literal["APPROVE", "MODIFY", "REJECT"]
    reviewed_by: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)
    modifications: list[dict[str, Any]] | None = None


class RunActionRequest(TrimmedTextModel):
    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)


class RerunRequest(RunActionRequest):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class SubmitReviewRequest(TrimmedTextModel):
    submitted_by: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="提交人工复核", max_length=1000)


class ReviewDecision(str, Enum):
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"


class ReviewRequest(TrimmedTextModel):
    decision: ReviewDecision
    reviewed_by: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="", max_length=1000)
    modified_suggestions: list[dict[str, Any]] | None = None


class MessageCreate(TrimmedTextModel):
    content: str = Field(min_length=1, max_length=8000)
    created_by: str = Field(min_length=1, max_length=128)


class PolicyJsonUpload(TrimmedTextModel):
    filename: str = Field(min_length=1, max_length=255)
    market_code: str = Field(min_length=2, max_length=8)
    version: str = Field(min_length=1, max_length=128)
    effective_date: date | None = None
    source_authorized: bool = False
    content_base64: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=255)


class PolicySearchRequest(TrimmedTextModel):
    query: str = Field(min_length=1, max_length=2000)
    market_code: str | None = Field(default=None, min_length=2, max_length=8)
    business_date: date | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ReportFormat(str, Enum):
    json = "json"
    markdown = "markdown"
    html = "html"


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()


ContentType = Literal["application/json", "text/markdown", "text/html"]
