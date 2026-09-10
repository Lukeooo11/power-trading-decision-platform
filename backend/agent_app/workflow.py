from __future__ import annotations

import asyncio
import html as html_lib
import inspect
import json
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Callable

from .config import Settings
from .db import Database, utc_now
from .platform_client import PlatformClient, PlatformError
from .research_workflow import validate_plan as validate_research_plan


WORKFLOW_STEPS: tuple[tuple[int, str], ...] = (
    (1, "market_validation"),
    (2, "data_quality"),
    (3, "policy_rules"),
    (4, "price_forecast"),
    (5, "declaration_strategy"),
    (6, "risk_signals"),
    (7, "strategy_gate"),
    (8, "report_draft"),
    (9, "dual_review"),
    (10, "final_report"),
)


class WorkflowCancelled(Exception):
    pass


class WorkflowValidationError(RuntimeError):
    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        return converted if isinstance(converted, dict) else {"value": converted}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {"value": value}


def _p50(raw: Any) -> float | None:
    if isinstance(raw, dict):
        raw = raw.get("p50")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


class AgentWorkflow:
    def __init__(
        self,
        db: Database,
        platform: PlatformClient,
        settings: Settings,
    ) -> None:
        self.db = db
        self.platform = platform
        self.settings = settings

    def initialize_steps(self, run_id: str) -> None:
        for sequence, name in WORKFLOW_STEPS:
            self.db.upsert_step(run_id, sequence, name, "PENDING", {})

    def _check_cancel(self, run_id: str) -> None:
        if self.db.is_cancel_requested(run_id):
            raise WorkflowCancelled

    def _finish_remaining_steps(
        self, run_id: str, current_sequence: int, status: str, detail: dict[str, Any]
    ) -> None:
        steps = {step["name"]: step for step in self.db.list_steps(run_id)}
        for sequence, name in WORKFLOW_STEPS:
            if sequence <= current_sequence:
                continue
            if steps.get(name, {}).get("status") == "PENDING":
                self.db.upsert_step(run_id, sequence, name, status, detail)

    async def run(self, run_id: str) -> None:
        current_sequence = 0
        current_name = "workflow"
        try:
            run = self.db.get_run(run_id)
            if run["status"] == "CANCELLED":
                return
            self.db.update_run(
                run_id,
                status="RUNNING",
                started_at=run.get("started_at") or utc_now(),
                error_code=None,
                error_message=None,
            )
            self.db.add_audit(run_id, "RUN_STARTED", "agent-workflow", {})

            current_sequence, current_name = 1, "market_validation"
            self.db.upsert_step(run_id, 1, current_name, "RUNNING")
            self._check_cancel(run_id)
            if run["market_code"] != "SD" or run["trading_subject"] != "retail":
                gap = {
                    "code": "MARKET_NOT_ONBOARDED",
                    "market_code": run["market_code"],
                    "trading_subject": run["trading_subject"],
                    "required": [
                        "authorized_market_data",
                        "market_specific_model",
                        "effective_market_rules",
                    ],
                    "message": "该市场或交易主体尚未接入，未调用山东数据和模型。",
                }
                self.db.add_evidence(
                    run_id,
                    kind="onboarding_gap",
                    source="agent-market-registry",
                    title="市场接入缺口",
                    data=gap,
                )
                self.db.upsert_step(run_id, 1, current_name, "SUCCEEDED", gap)
                for sequence, name in WORKFLOW_STEPS[1:7]:
                    self.db.upsert_step(
                        run_id,
                        sequence,
                        name,
                        "SKIPPED",
                        {"reason": "MARKET_NOT_ONBOARDED"},
                    )
                self.db.update_run(
                    run_id,
                    missing_data_json=gap["required"],
                    strategy_ready=False,
                )
                await self.build_report_version(run_id, phase="ONBOARDING")
                self.db.upsert_step(
                    run_id, 8, "report_draft", "SUCCEEDED", {"phase": "ONBOARDING"}
                )
                self.db.upsert_step(
                    run_id, 9, "dual_review", "SKIPPED", {"reason": "MARKET_NOT_ONBOARDED"}
                )
                self.db.upsert_step(
                    run_id, 10, "final_report", "SKIPPED", {"reason": "MARKET_NOT_ONBOARDED"}
                )
                self.db.add_audit(
                    run_id, "RUN_NEEDS_ONBOARDING", "agent-workflow", gap
                )
                # Publish the terminal state only after its report and steps are
                # durable, so GET/SSE clients cannot observe a partial result.
                self.db.update_run(
                    run_id,
                    status="NEEDS_ONBOARDING",
                    completed_at=utc_now(),
                )
                return
            self.db.upsert_step(
                run_id,
                1,
                current_name,
                "SUCCEEDED",
                {"market_supported": True, "data_scope": "SD/retail"},
            )

            current_sequence, current_name = 2, "data_quality"
            self.db.upsert_step(run_id, 2, current_name, "RUNNING")
            self._check_cancel(run_id)
            hourly = await self.platform.hourly_data(
                "SD",
                run["business_date"],
                data_version=run.get("data_version"),
            )
            snapshot, data_ready, missing_data = self._data_snapshot(hourly, run)
            self.db.update_run(
                run_id,
                data_version=snapshot.get("data_version"),
                data_ready=data_ready,
                input_snapshot_json=snapshot,
                missing_data_json=missing_data,
            )
            self.db.add_evidence(
                run_id,
                kind="data_quality",
                source="platform:/api/v1/hourly-data",
                title="平台数据质量快照",
                citation=f"data-version:{snapshot.get('data_version') or 'unknown'}",
                data=snapshot,
            )
            self.db.upsert_step(
                run_id,
                2,
                current_name,
                "SUCCEEDED",
                {
                    "input_shape_valid": snapshot["input_shape_valid"],
                    "data_ready": data_ready,
                    "point_count": snapshot["point_count"],
                    "missing_data_count": len(missing_data),
                },
            )

            current_sequence, current_name = 3, "policy_rules"
            self.db.upsert_step(run_id, 3, current_name, "RUNNING")
            self._check_cancel(run_id)
            policy_bundle = await asyncio.to_thread(
                self._policy_bundle, run["market_code"], run["business_date"]
            )
            policy_ready = bool(policy_bundle.get("ready")) and bool(
                policy_bundle.get("citations")
            )
            if policy_bundle.get("ready") and not policy_bundle.get("citations"):
                policy_bundle["ready"] = False
                policy_bundle["status"] = "MISSING_CITATION"
            self.db.update_run(
                run_id,
                policy_ready=policy_ready,
                policies_json=policy_bundle,
            )
            citations = policy_bundle.get("citations", [])
            if citations:
                for citation in citations:
                    location = (
                        f"第{citation['page_number']}页"
                        if citation.get("page_number")
                        else citation.get("section") or "文档片段"
                    )
                    self.db.add_evidence(
                        run_id,
                        evidence_id=f"policy-{citation['citation_id']}",
                        kind="policy_citation",
                        source=str(citation.get("source_reference") or citation.get("filename") or "policy-store"),
                        title=str(citation.get("title") or "政策规则"),
                        citation=f"{citation.get('title')} {citation.get('version')} {location}: {citation.get('snippet')}",
                        data=citation,
                    )
            else:
                self.db.add_evidence(
                    run_id,
                    kind="policy_gap",
                    source="agent-policy-store",
                    title="政策门禁未通过",
                    data={
                        "status": policy_bundle.get("status", "MISSING_POLICY"),
                        "message": "没有可引用的已授权、有效政策原文。",
                    },
                )
            self.db.upsert_step(
                run_id,
                3,
                current_name,
                "SUCCEEDED",
                {
                    "policy_ready": policy_ready,
                    "status": policy_bundle.get("status"),
                    "citation_count": len(citations),
                },
            )

            current_sequence, current_name = 4, "price_forecast"
            self.db.upsert_step(run_id, 4, current_name, "RUNNING")
            self._check_cancel(run_id)
            platform_request_id = self._platform_request_id(run)
            platform_run = await self.platform.create_forecast_run(
                request_id=platform_request_id,
                market_code="SD",
                market_date=run["business_date"],
                model_version=self.settings.model_version,
                data_version=str(snapshot.get("data_version") or "unknown"),
                parameters={
                    "quantiles": [0.1, 0.5, 0.9],
                    "spread_attention_threshold_yuan_per_mwh": self.settings.spread_attention_threshold,
                },
                input_summary={
                    "available_domains": snapshot.get("available_domains", []),
                    "hourly_point_count": snapshot.get("point_count", 0),
                    "customer_scope": "portfolio",
                },
            )
            platform_run_id = str(platform_run.get("run_id") or "")
            if not platform_run_id:
                raise WorkflowValidationError(
                    "PLATFORM_RUN_ID_MISSING", "平台预测运行未返回 run_id", platform_run
                )
            self.db.update_run(run_id, platform_run_id=platform_run_id)
            result_envelope = await self.platform.wait_for_forecast_result(
                platform_run,
                cancel_requested=lambda: self.db.is_cancel_requested(run_id),
            )
            forecast = result_envelope.get("result", result_envelope)
            self._validate_forecast(forecast, run)
            platform_strategy_ready = bool(forecast.get("strategy_ready"))
            self.db.update_run(
                run_id,
                forecast_json=forecast,
                platform_strategy_ready=platform_strategy_ready,
            )
            self.db.add_evidence(
                run_id,
                kind="model_result",
                source=f"platform:/api/v1/model-runs/{platform_run_id}/results",
                title="价格预测模型结果",
                citation=f"model:{self.settings.model_version};run:{platform_run_id}",
                data={
                    "model": forecast.get("model", {}),
                    "data_snapshot": forecast.get("data_snapshot", {}),
                    "backtest": forecast.get("backtest", {}),
                    "point_count": len(forecast.get("periods", [])),
                    "strategy_ready": platform_strategy_ready,
                },
            )
            self.db.upsert_step(
                run_id,
                4,
                current_name,
                "SUCCEEDED",
                {
                    "platform_run_id": platform_run_id,
                    "model_version": self.settings.model_version,
                    "point_count": 24,
                    "platform_strategy_ready": platform_strategy_ready,
                },
            )

            current_sequence, current_name = 5, "declaration_strategy"
            self.db.upsert_step(run_id, 5, current_name, "RUNNING")
            self._check_cancel(run_id)
            declaration_ready = False
            strategy_run = {**run, "platform_run_id": platform_run_id}
            try:
                plan = await self.platform.research_plan(
                    run["business_date"],
                    run["strategy_version"],
                    run["risk_aversion"],
                    platform_run_id,
                )
                try:
                    validate_research_plan(plan, strategy_run)
                except (KeyError, TypeError, ValueError) as error:
                    raise WorkflowValidationError(
                        "INVALID_DECLARATION_STRATEGY",
                        "交易申报策略返回结果未通过只读安全校验",
                        str(error),
                    ) from error
                declaration_strategy = self._declaration_strategy_reference(
                    plan, strategy_run
                )
                declaration_ready = plan.get("research_status") == "READY"
                declaration_missing = [
                    str(item) for item in plan.get("missing_data", []) if item
                ]
                if declaration_missing:
                    missing_data = sorted(set([*missing_data, *declaration_missing]))
                self.db.update_run(
                    run_id,
                    declaration_strategy_ready=declaration_ready,
                    declaration_strategy_json=declaration_strategy,
                    missing_data_json=missing_data,
                )
                self.db.add_evidence(
                    run_id,
                    kind="declaration_strategy_reference",
                    source="platform:/api/strategy/research",
                    title="交易申报策略只读引用",
                    citation=(
                        f"strategy:{run['strategy_version']};"
                        f"source-sha256:{plan.get('source_sha256') or 'unknown'}"
                    ),
                    data=declaration_strategy,
                )
                self.db.upsert_step(
                    run_id,
                    5,
                    current_name,
                    "SUCCEEDED" if declaration_ready else "BLOCKED",
                    {
                        "research_status": plan.get("research_status"),
                        "strategy_version": run["strategy_version"],
                        "risk_aversion": run["risk_aversion"],
                        "source_sha256": plan.get("source_sha256"),
                        "execution_allowed": False,
                    },
                )
            except PlatformError as error:
                missing_data = sorted(
                    set([*missing_data, "DECLARATION_STRATEGY_UNAVAILABLE"])
                )
                declaration_strategy = self._blocked_declaration_strategy_reference(
                    strategy_run, error
                )
                self.db.update_run(
                    run_id,
                    declaration_strategy_ready=False,
                    declaration_strategy_json=declaration_strategy,
                    missing_data_json=missing_data,
                )
                self.db.add_evidence(
                    run_id,
                    kind="declaration_strategy_gap",
                    source="platform:/api/strategy/research",
                    title="交易申报策略调用未完成",
                    data=declaration_strategy,
                )
                self.db.upsert_step(
                    run_id,
                    5,
                    current_name,
                    "BLOCKED",
                    {
                        "code": error.code,
                        "message": error.message,
                        "execution_allowed": False,
                    },
                )

            current_sequence, current_name = 6, "risk_signals"
            self.db.upsert_step(run_id, 6, current_name, "RUNNING")
            self._check_cancel(run_id)
            signals = self._build_signals(forecast)
            self.db.update_run(run_id, signals_json=signals)
            attention_count = sum(bool(item["attention_required"]) for item in signals)
            self.db.add_evidence(
                run_id,
                kind="research_signals",
                source=f"derived:{self.settings.model_version}",
                title="24 时段研究性风险信号",
                citation=f"absolute-spread-threshold:{self.settings.spread_attention_threshold:g} CNY/MWh",
                data={
                    "threshold_yuan_per_mwh": self.settings.spread_attention_threshold,
                    "attention_periods": [
                        item["period"] for item in signals if item["attention_required"]
                    ],
                },
            )
            self.db.upsert_step(
                run_id,
                6,
                current_name,
                "SUCCEEDED",
                {"point_count": len(signals), "attention_count": attention_count},
            )

            current_sequence, current_name = 7, "strategy_gate"
            self.db.upsert_step(run_id, 7, current_name, "RUNNING")
            self._check_cancel(run_id)
            strategy_ready = bool(
                platform_strategy_ready
                and declaration_ready
                and data_ready
                and policy_ready
            )
            formal_strategy = self._build_formal_strategy(
                forecast,
                data_ready=data_ready,
                policy_ready=policy_ready,
                platform_strategy_ready=platform_strategy_ready,
                declaration_strategy_ready=declaration_ready,
            )
            self.db.update_run(
                run_id,
                strategy_ready=strategy_ready,
                formal_strategy_json=formal_strategy,
            )
            self.db.upsert_step(
                run_id,
                7,
                current_name,
                "SUCCEEDED",
                {
                    "strategy_ready": strategy_ready,
                    "declaration_strategy_ready": declaration_ready,
                    "formal_action": "HOLD",
                    "point_count": len(formal_strategy),
                    "execution_allowed": False,
                },
            )

            current_sequence, current_name = 8, "report_draft"
            self.db.upsert_step(run_id, 8, current_name, "RUNNING")
            self._check_cancel(run_id)
            report = await self.build_report_version(run_id, phase="DRAFT")
            self.db.upsert_step(
                run_id,
                8,
                current_name,
                "SUCCEEDED",
                {"version": report["version"], "phase": report["phase"]},
            )
            self.db.update_run(run_id, status="READY_FOR_REVIEW")
            self.db.add_audit(
                run_id,
                "RUN_READY_FOR_REVIEW",
                "agent-workflow",
                {"report_version": report["version"], "execution_allowed": False},
            )
        except (WorkflowCancelled, asyncio.CancelledError):
            self.db.update_run(
                run_id,
                status="CANCELLED",
                cancel_requested=True,
                completed_at=utc_now(),
            )
            if current_sequence:
                self.db.upsert_step(
                    run_id,
                    current_sequence,
                    current_name,
                    "CANCELLED",
                    {"reason": "CANCEL_REQUESTED"},
                )
            self._finish_remaining_steps(
                run_id, current_sequence, "CANCELLED", {"reason": "CANCEL_REQUESTED"}
            )
            self.db.add_audit(run_id, "RUN_CANCELLED", "agent-workflow", {})
        except (PlatformError, WorkflowValidationError) as error:
            code = error.code
            message = error.message
            details = getattr(error, "details", None)
            self._fail(run_id, current_sequence, current_name, code, message, details)
        except Exception as error:
            self._fail(
                run_id,
                current_sequence,
                current_name,
                "WORKFLOW_FAILED",
                str(error),
                None,
            )

    def _fail(
        self,
        run_id: str,
        sequence: int,
        step_name: str,
        code: str,
        message: str,
        details: Any,
    ) -> None:
        self.db.update_run(
            run_id,
            status="FAILED",
            error_code=code,
            error_message=message,
            completed_at=utc_now(),
        )
        if sequence:
            self.db.upsert_step(
                run_id,
                sequence,
                step_name,
                "FAILED",
                {"code": code, "message": message, "details": details},
            )
        self._finish_remaining_steps(
            run_id, sequence, "SKIPPED", {"reason": "UPSTREAM_FAILURE"}
        )
        self.db.add_audit(
            run_id,
            "RUN_FAILED",
            "agent-workflow",
            {"code": code, "message": message},
        )

    @staticmethod
    def _platform_request_id(run: dict[str, Any]) -> str:
        suffix = run["run_id"].removeprefix("agent-")[:12]
        prefix = str(run["request_id"])
        max_prefix = max(1, 128 - len(suffix) - 4)
        return f"{prefix[:max_prefix]}-pf-{suffix}"

    @staticmethod
    def _data_snapshot(
        hourly: dict[str, Any], run: dict[str, Any]
    ) -> tuple[dict[str, Any], bool, list[str]]:
        rows = hourly.get("rows")
        rows = rows if isinstance(rows, list) else []
        periods = [row.get("period") for row in rows if isinstance(row, dict)]
        input_shape_valid = (
            len(rows) == 24
            and len(periods) == 24
            and set(periods) == set(range(1, 25))
            and len(set(periods)) == 24
            and int(hourly.get("point_count", len(rows))) == 24
        )
        missing_counts = hourly.get("missing_field_counts") or {}
        missing_domains = [str(item) for item in hourly.get("strategy_missing_domains", [])]
        missing_fields = [
            f"{name}:{count}"
            for name, count in missing_counts.items()
            if isinstance(count, int) and count > 0
        ]
        missing_data = sorted(set([*missing_domains, *missing_fields]))
        data_ready = bool(input_shape_valid and hourly.get("strategy_ready", False))
        snapshot = {
            "schema_version": hourly.get("schema_version"),
            "market_code": run["market_code"],
            "trading_subject": run["trading_subject"],
            "business_date": run["business_date"],
            "timezone": hourly.get("timezone", "Asia/Shanghai"),
            "data_version": hourly.get("data_version"),
            "updated_at": hourly.get("updated_at"),
            "point_count": len(rows),
            "source_versions": hourly.get("source_versions", {}),
            "missing_field_counts": missing_counts,
            "available_domains": hourly.get("available_domains", []),
            "strategy_missing_domains": missing_domains,
            "platform_input_strategy_ready": bool(hourly.get("strategy_ready", False)),
            "input_shape_valid": input_shape_valid,
            "private_rows_persisted": False,
        }
        return snapshot, data_ready, missing_data

    def _policy_bundle(self, market_code: str, business_date: str) -> dict[str, Any]:
        try:
            from . import knowledge
        except ImportError:
            return {
                "ready": False,
                "status": "POLICY_MODULE_UNAVAILABLE",
                "documents": [],
                "citations": [],
            }
        try:
            with closing(self.db.connect()) as connection:
                gate = knowledge.policy_gate_status(connection, market_code, business_date)
                citations: list[dict[str, Any]] = []
                seen: set[str] = set()
                if gate.get("ready"):
                    search = getattr(knowledge, "search_policy", None) or getattr(
                        knowledge, "search_policy_chunks"
                    )
                    for query in ("交易", "规则", "申报", "结算", "价格"):
                        hits = search(
                            connection,
                            query,
                            market_code=market_code,
                            as_of_date=business_date,
                            limit=5,
                        )
                        for hit in hits:
                            item = _as_dict(hit)
                            citation_id = str(item.get("citation_id") or "")
                            if citation_id and citation_id not in seen:
                                seen.add(citation_id)
                                citations.append(item)
                        if len(citations) >= 8:
                            break
                return {**gate, "citations": citations[:8]}
        except Exception as error:
            return {
                "ready": False,
                "status": "POLICY_STORE_ERROR",
                "documents": [],
                "citations": [],
                "error": str(error),
            }

    @staticmethod
    def _validate_forecast(forecast: Any, run: dict[str, Any]) -> None:
        if not isinstance(forecast, dict):
            raise WorkflowValidationError(
                "INVALID_FORECAST", "平台预测结果必须是 JSON 对象"
            )
        if str(forecast.get("market_code", "")).upper() != run["market_code"]:
            raise WorkflowValidationError(
                "FORECAST_MARKET_MISMATCH", "预测市场与 Agent 运行市场不一致"
            )
        if forecast.get("market_date") != run["business_date"]:
            raise WorkflowValidationError(
                "FORECAST_DATE_MISMATCH", "预测日期与 Agent 业务日期不一致"
            )
        periods = forecast.get("periods")
        if not isinstance(periods, list) or len(periods) != 24:
            raise WorkflowValidationError(
                "INCOMPLETE_FORECAST", "预测结果必须包含 24 个时段"
            )
        period_numbers = [
            item.get("period") for item in periods if isinstance(item, dict)
        ]
        if len(period_numbers) != 24 or set(period_numbers) != set(range(1, 25)):
            raise WorkflowValidationError(
                "INCOMPLETE_FORECAST", "预测结果必须且只能包含时段 1 至 24"
            )

    @staticmethod
    def _declaration_strategy_reference(
        plan: dict[str, Any], run: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist only the strategy module's immutable research reference."""

        return {
            "strategy_version": run["strategy_version"],
            "resolved_strategy_version": plan.get("strategy_version"),
            "risk_aversion": run["risk_aversion"],
            "research_status": plan.get("research_status"),
            "summary": plan.get("summary") or {},
            "method": plan.get("method") or {},
            "top_attention_periods": plan.get("top_attention_periods") or [],
            "missing_data": plan.get("missing_data") or [],
            "assumptions": plan.get("assumptions") or [],
            "source": {
                "endpoint": "/api/strategy/research",
                "market_code": "SD",
                "business_date": run["business_date"],
                "source_sha256": plan.get("source_sha256"),
                "forecast_version": plan.get("forecast_version"),
                "forecast_run_id": plan.get("forecast_run_id"),
                "position_data_version": plan.get("position_data_version"),
                "captured_at": utc_now(),
            },
            "run_reference": {
                "agent_run_id": run["run_id"],
                "preceding_forecast_run_id": run.get("platform_run_id"),
                "bound_forecast_run_id": plan.get("forecast_run_id"),
            },
            "formal_gate": "BLOCKED",
            "read_only": True,
            "execution_allowed": False,
        }

    @staticmethod
    def _blocked_declaration_strategy_reference(
        run: dict[str, Any], error: PlatformError
    ) -> dict[str, Any]:
        return {
            "strategy_version": run["strategy_version"],
            "resolved_strategy_version": None,
            "risk_aversion": run["risk_aversion"],
            "research_status": "BLOCKED",
            "summary": {},
            "missing_data": ["DECLARATION_STRATEGY_UNAVAILABLE"],
            "source": {
                "endpoint": "/api/strategy/research",
                "market_code": "SD",
                "business_date": run["business_date"],
                "captured_at": utc_now(),
            },
            "run_reference": {
                "agent_run_id": run["run_id"],
                "preceding_forecast_run_id": run.get("platform_run_id"),
            },
            "error": error.as_dict(),
            "formal_gate": "BLOCKED",
            "read_only": True,
            "execution_allowed": False,
        }

    def _build_signals(self, forecast: dict[str, Any]) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        for point in sorted(forecast["periods"], key=lambda item: item["period"]):
            day_ahead = _p50(point.get("day_ahead_price_yuan_per_mwh"))
            real_time = _p50(point.get("real_time_price_yuan_per_mwh"))
            provided_spread = point.get(
                "spread_day_ahead_minus_real_time_yuan_per_mwh"
            )
            spread = (
                float(provided_spread)
                if isinstance(provided_spread, (int, float))
                and not isinstance(provided_spread, bool)
                else day_ahead - real_time
                if day_ahead is not None and real_time is not None
                else None
            )
            negative_risk = point.get("negative_price_risk") or {
                "probability": None,
                "level": "LOW",
            }
            high_risk = point.get("high_price_risk") or {
                "probability": None,
                "level": "LOW",
            }
            reasons = list(point.get("risk_reason_codes") or [])
            spread_attention = spread is not None and abs(spread) >= self.settings.spread_attention_threshold
            if spread_attention and "ABSOLUTE_SPREAD_ATTENTION" not in reasons:
                reasons.append("ABSOLUTE_SPREAD_ATTENTION")
            attention = bool(
                spread_attention
                or str(negative_risk.get("level", "LOW")).upper() != "LOW"
                or str(high_risk.get("level", "LOW")).upper() != "LOW"
            )
            signals.append(
                {
                    "period": int(point["period"]),
                    "datetime": point.get("datetime"),
                    "day_ahead_p50_yuan_per_mwh": day_ahead,
                    "real_time_p50_yuan_per_mwh": real_time,
                    "forecast_price": day_ahead,
                    "reference_price": real_time,
                    "spread_yuan_per_mwh": spread,
                    "absolute_spread_yuan_per_mwh": abs(spread) if spread is not None else None,
                    "absolute_spread": abs(spread) if spread is not None else None,
                    "spread_attention_threshold_yuan_per_mwh": self.settings.spread_attention_threshold,
                    "spread_attention": spread_attention,
                    "negative_price_risk": negative_risk,
                    "high_price_risk": high_risk,
                    "negative_price_level": str(negative_risk.get("level", "LOW")).upper(),
                    "high_price_level": str(high_risk.get("level", "LOW")).upper(),
                    "risk_level": (
                        "HIGH"
                        if "HIGH" in {
                            str(negative_risk.get("level", "LOW")).upper(),
                            str(high_risk.get("level", "LOW")).upper(),
                        }
                        else "MEDIUM"
                        if "MEDIUM" in {
                            str(negative_risk.get("level", "LOW")).upper(),
                            str(high_risk.get("level", "LOW")).upper(),
                        }
                        else "LOW"
                    ),
                    "attention_required": attention,
                    "reason_codes": reasons,
                    "research_only": True,
                    "execution_allowed": False,
                }
            )
        return signals

    @staticmethod
    def _build_formal_strategy(
        forecast: dict[str, Any],
        *,
        data_ready: bool,
        policy_ready: bool,
        platform_strategy_ready: bool,
        declaration_strategy_ready: bool,
    ) -> list[dict[str, Any]]:
        reason_codes: list[str] = []
        if not data_ready:
            reason_codes.append("DATA_GATE_NOT_READY")
        if not policy_ready:
            reason_codes.append("POLICY_GATE_NOT_READY")
        if not platform_strategy_ready:
            reason_codes.append("PLATFORM_STRATEGY_NOT_READY")
        if not declaration_strategy_ready:
            reason_codes.append("DECLARATION_STRATEGY_NOT_READY")
        if not reason_codes:
            reason_codes.append("MANUAL_REVIEW_ONLY")
        strategy = []
        for point in sorted(forecast["periods"], key=lambda item: item["period"]):
            strategy.append(
                {
                    "period": int(point["period"]),
                    "datetime": point.get("datetime"),
                    "action": "HOLD",
                    "volume_mwh": 0,
                    "quantity_mwh": 0,
                    "price_yuan_per_mwh": None,
                    "confidence": point.get("confidence"),
                    "reason_codes": reason_codes,
                    "execution_allowed": False,
                }
            )
        return strategy

    async def build_report_version(
        self,
        run_id: str,
        *,
        phase: str,
        review: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.db.get_run(run_id)
        evidence = run.get("evidence", [])
        forecast = run.get("forecast", {})
        signals = run.get("signals", [])
        policies = run.get("policies", {})
        report_version = len(run.get("reports", [])) + 1
        payload, markdown, html = await asyncio.to_thread(
            self._render_report,
            run,
            evidence,
            forecast,
            signals,
            policies,
            review,
            phase,
            report_version,
        )
        payload["execution_allowed"] = False
        payload["phase"] = phase
        return self.db.save_report(
            run_id,
            phase=phase,
            payload=payload,
            markdown=markdown,
            html=html,
        )

    def _render_report(
        self,
        run: dict[str, Any],
        evidence: list[dict[str, Any]],
        forecast: dict[str, Any],
        signals: list[dict[str, Any]],
        policies: Any,
        review: dict[str, Any] | None,
        phase: str,
        report_version: int,
    ) -> tuple[dict[str, Any], str, str]:
        try:
            from . import reporting

            policy_citations = (
                policies.get("citations", []) if isinstance(policies, dict) else []
            )
            snapshot = {
                **run,
                "data_quality": run.get("input_snapshot", {}),
                "policy": policies if isinstance(policies, dict) else {},
                "policy_citations": policy_citations,
                "forecast": forecast,
                "research_signals": signals,
                "formal_strategy": run.get("formal_strategy", []),
                "evidence": evidence,
                "review": review or run.get("review", {}),
            }
            narrative = None
            try:
                from . import llm

                narrative = llm.generate_narrative(
                    snapshot, policy_citations, settings=self.settings
                )
            except Exception:
                narrative = None
            parameters = inspect.signature(reporting.build_report).parameters
            if "report_version" in parameters:
                built = reporting.build_report(
                    snapshot,
                    report_version=report_version,
                    phase=phase,
                    narrative=narrative,
                )
            else:
                built = reporting.build_report(
                    run,
                    evidence,
                    forecast,
                    signals,
                    policies,
                    review=review,
                )
            bundle = _as_dict(built)
            payload = (
                bundle["payload"]
                if isinstance(bundle.get("payload"), dict)
                else bundle
            )
            markdown = str(bundle.get("markdown") or "")
            html = str(bundle.get("html") or "")
            render_md = getattr(reporting, "render_report_markdown", None)
            render_html = getattr(reporting, "render_report_html", None)
            if not markdown and callable(render_md):
                markdown = str(render_md(payload))
            if not html and callable(render_html):
                html = str(render_html(markdown))
            if markdown and html:
                return payload, markdown, html
        except Exception:
            pass
        return self._fallback_report(run, evidence, forecast, signals, policies, review, phase)

    @staticmethod
    def _fallback_report(
        run: dict[str, Any],
        evidence: list[dict[str, Any]],
        forecast: dict[str, Any],
        signals: list[dict[str, Any]],
        policies: Any,
        review: dict[str, Any] | None,
        phase: str,
    ) -> tuple[dict[str, Any], str, str]:
        attention = [item["period"] for item in signals if item.get("attention_required")]
        payload = {
            "title": f"{run['market_code']} {run['business_date']} 交易决策编排报告",
            "phase": phase,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run": {
                "run_id": run["run_id"],
                "request_id": run["request_id"],
                "market_code": run["market_code"],
                "trading_subject": run["trading_subject"],
                "business_date": run["business_date"],
                "model_version": run["model_version"],
                "data_version": run.get("data_version"),
            },
            "data_quality": run.get("input_snapshot", {}),
            "policy_basis": policies,
            "model": forecast.get("model", {}),
            "backtest": forecast.get("backtest", {}),
            "risk_attention_periods": attention,
            "signals": signals,
            "declaration_strategy": run.get("declaration_strategy", {}),
            "formal_strategy": run.get("formal_strategy", []),
            "gates": run.get("gates", {}),
            "missing_data": run.get("missing_data", []),
            "evidence": evidence,
            "review": review or run.get("review", {}),
            "execution_allowed": False,
            "disclaimer": "本报告仅用于目标日研究、历史复盘和人工复核，不连接交易终端，不构成自动下单指令。",
        }
        lines = [
            f"# {payload['title']}",
            "",
            f"- 报告阶段：{phase}",
            f"- 模型版本：{run['model_version']}",
            f"- 数据版本：{run.get('data_version') or '未取得'}",
            f"- 申报策略版本：{run.get('strategy_version') or '未取得'}",
            f"- 申报策略研究状态：{(run.get('declaration_strategy') or {}).get('research_status') or '未取得'}",
            f"- 策略门禁：{'通过' if run.get('strategy_ready') else '未通过'}",
            "- 执行安全占位：24 时段 HOLD，电量 0 MWh",
            "- 自动执行：关闭",
            "",
            "## 研究性风险时段",
            "",
            ", ".join(str(item) for item in attention) if attention else "无阈值关注时段",
            "",
            "## 缺失数据",
            "",
            ", ".join(run.get("missing_data", [])) or "无已记录缺口",
            "",
            "## 声明",
            "",
            payload["disclaimer"],
        ]
        markdown = "\n".join(lines)
        html = (
            "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
            f"<title>{html_lib.escape(payload['title'])}</title>"
            "<body><main><pre style=\"white-space:pre-wrap;font-family:system-ui\">"
            f"{html_lib.escape(markdown)}</pre></main></body></html>"
        )
        return payload, markdown, html
