from __future__ import annotations

import asyncio
import base64
import binascii
import json
import csv
import io
import math
import hashlib
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .config import Settings, get_settings
from .db import Database, DraftConflict, content_sha256, utc_now
from .platform_client import PlatformClient, PlatformError
from .schemas import (
    AgentRunCreate,
    MessageCreate,
    ReportFormat,
    RerunRequest,
    ReviewDecision,
    ReviewRequest,
    RunActionRequest,
    SubmitReviewRequest,
    TradingDraftRunCreate,
    TradingDraftReview,
    TradingResearchRunCreate,
    model_to_dict,
)
from .workflow import AgentWorkflow
from .research_workflow import ResearchWorkflow


MARKETS: tuple[tuple[str, str], ...] = (
    ("BJ", "北京"),
    ("TJ", "天津"),
    ("HE", "河北"),
    ("SX", "山西"),
    ("NM", "内蒙古"),
    ("LN", "辽宁"),
    ("JL", "吉林"),
    ("HL", "黑龙江"),
    ("SH", "上海"),
    ("JS", "江苏"),
    ("ZJ", "浙江"),
    ("AH", "安徽"),
    ("FJ", "福建"),
    ("JX", "江西"),
    ("SD", "山东"),
    ("HA", "河南"),
    ("HB", "湖北"),
    ("HN", "湖南"),
    ("GD", "广东"),
    ("GX", "广西"),
    ("HI", "海南"),
    ("CQ", "重庆"),
    ("SC", "四川"),
    ("GZ", "贵州"),
    ("YN", "云南"),
    ("XZ", "西藏"),
    ("SN", "陕西"),
    ("GS", "甘肃"),
    ("QH", "青海"),
    ("NX", "宁夏"),
    ("XJ", "新疆"),
)

TRADING_SUBJECTS = frozenset({"retail", "generation", "station", "storage"})

STABLE_SSE_STATUSES = {
    "READY_FOR_REVIEW",
    "PENDING_REVIEW",
    "APPROVED",
    "MODIFIED",
    "REJECTED",
    "NEEDS_ONBOARDING",
    "FAILED",
    "CANCELLED",
}


def api_error(status_code: int, code: str, message: str, **details: Any) -> HTTPException:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return HTTPException(status_code=status_code, detail=payload)


def _run_or_404(db: Database, run_id: str) -> dict[str, Any]:
    try:
        return db.get_run(run_id)
    except KeyError as error:
        raise api_error(404, "RUN_NOT_FOUND", f"未知 Agent 运行：{run_id}") from error


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalized_identity(value: str) -> str:
    return value.strip().casefold()


def _validate_run_scope(market_code: str, trading_subject: str) -> tuple[str, str]:
    normalized_market = market_code.strip().upper()
    normalized_subject = trading_subject.strip().lower()
    if normalized_market not in dict(MARKETS):
        raise api_error(404, "MARKET_NOT_FOUND", f"未知市场代码：{normalized_market}")
    if normalized_subject not in TRADING_SUBJECTS:
        raise api_error(
            422,
            "TRADING_SUBJECT_NOT_FOUND",
            f"未知交易主体：{normalized_subject}",
            supported=sorted(TRADING_SUBJECTS),
        )
    return normalized_market, normalized_subject


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        result = converter()
        return result if isinstance(result, dict) else {"value": result}
    return {"value": value}


def _schedule_workflow(app: FastAPI, run_id: str) -> None:
    tasks: dict[str, asyncio.Task[None]] = app.state.workflow_tasks
    existing = tasks.get(run_id)
    if existing and not existing.done():
        return
    task = asyncio.create_task(app.state.workflow.run(run_id), name=f"agent-run:{run_id}")
    tasks[run_id] = task

    def cleanup(done: asyncio.Task[None]) -> None:
        if tasks.get(run_id) is done:
            tasks.pop(run_id, None)
        if not done.cancelled():
            try:
                done.exception()
            except Exception:
                pass

    task.add_done_callback(cleanup)


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    settings = custom_settings or get_settings()
    database = Database(settings.database_path, settings.retention_days)
    platform = PlatformClient(settings)
    workflow = AgentWorkflow(database, platform, settings)
    research_workflow = ResearchWorkflow(database, platform, workflow._policy_bundle)

    def schedule_research(run_id: str) -> None:
        tasks = application.state.workflow_tasks
        key = f"research:{run_id}"
        if key in tasks and not tasks[key].done():
            return
        task = asyncio.create_task(research_workflow.run(run_id))
        tasks[key] = task
        task.add_done_callback(lambda completed: tasks.pop(key, None) if tasks.get(key) is completed else None)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database.initialize()
        database.prune_expired_runs()
        try:
            from .knowledge import init_policy_store

            with closing(database.connect()) as connection:
                init_policy_store(connection)
        except Exception:
            pass
        for pending_id in database.pending_research_runs():
            schedule_research(pending_id)
        yield
        tasks = list(application.state.workflow_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    application = FastAPI(
        title="全国电力交易 AI Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.db = database
    application.state.platform = platform
    application.state.workflow = workflow
    application.state.research_workflow = research_workflow
    application.state.workflow_tasks = {}
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "VALIDATION_ERROR",
                    "message": "请求参数校验失败",
                    "errors": jsonable_encoder(exc.errors(), custom_encoder={ValueError: str}),
                }
            },
        )

    @application.exception_handler(DraftConflict)
    async def draft_conflict_handler(_: Request, exc: DraftConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": {"code": "DRAFT_CONFLICT", "message": str(exc)}})

    @application.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        try:
            platform_health = await platform.health()
            platform_state = {
                "status": "available",
                "base_url": settings.platform_base_url,
                "service": platform_health.get("service"),
                "api_version": platform_health.get("api_version"),
                "execution_allowed": False,
            }
        except PlatformError as error:
            platform_state = {
                "status": "unavailable",
                "base_url": settings.platform_base_url,
                "error": error.as_dict(),
                "execution_allowed": False,
            }
        try:
            from .llm import LLMAdapter

            llm_adapter = LLMAdapter(settings)
            llm_state = llm_adapter.health()
            llm_adapter.close()
        except Exception as error:
            llm_state = {
                "enabled": False,
                "mode": "deterministic_fallback",
                "error": str(error),
            }
        return {
            "status": "ok" if platform_state["status"] == "available" else "degraded",
            "service": settings.service_name,
            "environment": settings.environment,
            "api_version": "1.0.0",
            "timezone": "Asia/Shanghai",
            "platform": platform_state,
            "llm": llm_state,
            "retention_days": settings.retention_days,
            "execution_allowed": False,
        }

    @application.get("/api/v1/readiness")
    async def readiness(
        market_code: str | None = Query(default=None, min_length=2, max_length=8),
        trading_subject: str = Query(default="retail", min_length=1, max_length=64),
    ) -> dict[str, Any]:
        normalized_market = market_code.strip().upper() if market_code else None
        market_lookup = dict(MARKETS)
        if normalized_market and normalized_market not in market_lookup:
            raise api_error(404, "MARKET_NOT_FOUND", f"未知市场代码：{normalized_market}")
        normalized_subject = trading_subject.strip().lower()
        if normalized_subject not in TRADING_SUBJECTS:
            raise api_error(
                422,
                "TRADING_SUBJECT_NOT_FOUND",
                f"未知交易主体：{normalized_subject}",
                supported=sorted(TRADING_SUBJECTS),
            )
        selected = [item for item in MARKETS if not normalized_market or item[0] == normalized_market]

        try:
            platform_health = await platform.health()
            platform_online = str(platform_health.get("status", "ok")).lower() in {"ok", "healthy"}
            platform_error: dict[str, Any] | None = None
        except PlatformError as error:
            platform_online = False
            platform_error = error.as_dict()

        models_payload: dict[str, Any] = {}
        assets_payload: dict[str, Any] = {}
        if any(code == "SD" for code, _ in selected) and platform_online:
            model_result, asset_result = await asyncio.gather(
                platform.models(), platform.data_assets("SD"), return_exceptions=True
            )
            if isinstance(model_result, dict):
                models_payload = model_result
            if isinstance(asset_result, dict):
                assets_payload = asset_result

        model_ready = any(
            item.get("id") == settings.model_id
            and settings.model_version in (item.get("versions") or [])
            for item in models_payload.get("models", [])
            if isinstance(item, dict)
        )
        asset_status = {
            str(item.get("domain")): str(item.get("status"))
            for item in assets_payload.get("assets", [])
            if isinstance(item, dict)
        }
        data_connected = all(
            asset_status.get(domain) in {"REAL_CONNECTED", "REAL_PARTIAL"}
            for domain in ("prices", "load_actual", "weather", "market_supply_history")
        )

        def policy_gate(code: str) -> dict[str, Any]:
            try:
                from .knowledge import policy_gate_status

                with closing(database.connect()) as connection:
                    return policy_gate_status(
                        connection, code, datetime.now(timezone.utc).date().isoformat()
                    )
            except Exception as error:
                return {
                    "ready": False,
                    "status": "POLICY_STORE_ERROR",
                    "documents": [],
                    "error": str(error),
                }

        matrices = []
        for code, name in selected:
            gate = await asyncio.to_thread(policy_gate, code)
            supported_scope = code == "SD" and normalized_subject == "retail"
            if supported_scope:
                missing = []
                if not platform_online:
                    missing.append("platform_connection")
                if not data_connected:
                    missing.append("authorized_data_assets")
                if not model_ready:
                    missing.append(settings.model_version)
                if not gate.get("ready"):
                    missing.append("effective_policy_with_citation")
                state = "READY" if not missing else "PARTIAL"
            else:
                missing = [
                    "authorized_market_data",
                    "market_specific_model",
                    "effective_market_rules",
                ]
                state = "NEEDS_ONBOARDING"
            matrices.append(
                {
                    "market_code": code,
                    "market_name": name,
                    "trading_subject": normalized_subject,
                    "status": state,
                    "run_supported": supported_scope,
                    "data": {
                        "ready": bool(supported_scope and platform_online and data_connected),
                        "status": "CONNECTED" if supported_scope and data_connected else "NOT_ONBOARDED",
                        "domains": asset_status if supported_scope else {},
                    },
                    "model": {
                        "ready": bool(supported_scope and model_ready),
                        "id": settings.model_id if supported_scope else None,
                        "version": settings.model_version if supported_scope else None,
                    },
                    "policy": {
                        "ready": bool(gate.get("ready")),
                        "status": gate.get("status", "MISSING_POLICY"),
                        "document_count": len(gate.get("documents", [])),
                    },
                    "missing": missing,
                    "execution_allowed": False,
                }
            )
        return {
            "generated_at": utc_now(),
            "market_count": len(matrices),
            "total_market_count": len(MARKETS),
            "supported_scope": {
                "market_code": "SD",
                "trading_subject": "retail",
                "model_version": settings.model_version,
            },
            "platform": {
                "status": "available" if platform_online else "unavailable",
                "error": platform_error,
            },
            "markets": matrices,
            "execution_allowed": False,
        }

    @application.post("/api/v1/agent-runs", status_code=202)
    async def create_agent_run(request: AgentRunCreate) -> JSONResponse:
        payload = model_to_dict(request)
        market_code, trading_subject = _validate_run_scope(
            request.market_code, request.trading_subject
        )
        run, replay = database.create_run(
            request_id=request.request_id.strip(),
            market_code=market_code,
            trading_subject=trading_subject,
            business_date=str(payload["business_date"]),
            initiated_by=request.initiated_by.strip(),
            model_id=settings.model_id,
            model_version=settings.model_version,
            strategy_version=request.strategy_version,
            risk_aversion=request.risk_aversion,
        )
        if not replay:
            workflow.initialize_steps(run["run_id"])
            database.add_audit(run["run_id"], "RUN_QUEUED", request.initiated_by, {})
            run = database.get_run(run["run_id"])
            _schedule_workflow(application, run["run_id"])
        elif run["status"] == "QUEUED":
            _schedule_workflow(application, run["run_id"])
        run["idempotent_replay"] = replay
        return JSONResponse(status_code=202, content=run)

    @application.get("/api/v1/agent-runs/{run_id}")
    async def get_agent_run(run_id: str) -> dict[str, Any]:
        return _run_or_404(database, run_id)

    @application.get("/api/v1/agent-runs/{run_id}/events")
    async def run_events(
        request: Request,
        run_id: str,
        after_id: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        _run_or_404(database, run_id)
        header_id = request.headers.get("Last-Event-ID")
        if header_id and header_id.isdigit():
            after_id = max(after_id, int(header_id))

        async def stream() -> AsyncIterator[str]:
            cursor = after_id
            idle_ticks = 0
            yield "retry: 1500\n\n"
            while True:
                if await request.is_disconnected():
                    return
                events = database.list_audit(run_id, cursor)
                for event in events:
                    cursor = int(event["id"])
                    event_name = str(event["event_type"]).lower().replace("_", "-")
                    yield (
                        f"id: {cursor}\n"
                        f"event: {event_name}\n"
                        f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                    )
                state = database.get_run(run_id, include_details=False)
                if state["status"] in STABLE_SSE_STATUSES and not database.list_audit(run_id, cursor):
                    yield (
                        "event: stream-complete\n"
                        f"data: {json.dumps({'run_id': run_id, 'status': state['status']}, ensure_ascii=False)}\n\n"
                    )
                    return
                idle_ticks += 1
                if idle_ticks >= 40:
                    idle_ticks = 0
                    yield f": heartbeat {utc_now()}\n\n"
                await asyncio.sleep(0.35)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @application.post("/api/v1/agent-runs/{run_id}/cancel")
    async def cancel_agent_run(run_id: str, request: RunActionRequest) -> dict[str, Any]:
        run = _run_or_404(database, run_id)
        if run["status"] == "CANCELLED":
            run["idempotent_replay"] = True
            return run
        if run["status"] not in {"QUEUED", "RUNNING"}:
            raise api_error(
                409,
                "RUN_NOT_CANCELLABLE",
                f"状态 {run['status']} 的运行不可取消",
            )
        database.update_run(
            run_id,
            cancel_requested=True,
            status="CANCELLED",
            completed_at=utc_now(),
        )
        database.add_audit(
            run_id,
            "CANCEL_REQUESTED",
            request.actor,
            {"reason": request.reason},
        )
        if run.get("platform_run_id"):
            try:
                await platform.cancel_run(
                    run["platform_run_id"], request.actor, request.reason
                )
            except PlatformError as error:
                database.add_audit(
                    run_id,
                    "PLATFORM_CANCEL_DEGRADED",
                    request.actor,
                    error.as_dict(),
                )
        task = application.state.workflow_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        for step in database.list_steps(run_id):
            if step["status"] in {"PENDING", "RUNNING"}:
                database.upsert_step(
                    run_id,
                    int(step["sequence"]),
                    str(step["name"]),
                    "CANCELLED",
                    {"reason": "CANCEL_REQUESTED"},
                )
        result = database.get_run(run_id)
        result["idempotent_replay"] = False
        return result

    @application.post("/api/v1/agent-runs/{run_id}/rerun", status_code=202)
    async def rerun_agent_run(run_id: str, request: RerunRequest) -> JSONResponse:
        original = _run_or_404(database, run_id)
        if original["status"] in {"QUEUED", "RUNNING"}:
            raise api_error(409, "RUN_STILL_ACTIVE", "活动中的运行不能重跑")
        rerun_request_id = request.request_id or (
            f"{original['request_id']}:rerun:{run_id.removeprefix('agent-')[:12]}"
        )
        rerun_request_id = rerun_request_id[:128]
        rerun, replay = database.create_run(
            request_id=rerun_request_id,
            market_code=original["market_code"],
            trading_subject=original["trading_subject"],
            business_date=original["business_date"],
            initiated_by=request.actor,
            model_id=original["model_id"],
            model_version=original["model_version"],
            strategy_version=original.get("strategy_version") or "historical_cvar_v02",
            risk_aversion=float(original.get("risk_aversion", 0.3)),
            parent_run_id=run_id,
            data_version=original.get("data_version"),
            input_snapshot={
                "rerun_of": run_id,
                "frozen_data_version": original.get("data_version"),
            },
        )
        if not replay:
            workflow.initialize_steps(rerun["run_id"])
            database.add_audit(
                rerun["run_id"],
                "RUN_QUEUED",
                request.actor,
                {"reason": request.reason, "rerun_of": run_id},
            )
            rerun = database.get_run(rerun["run_id"])
            _schedule_workflow(application, rerun["run_id"])
        rerun["idempotent_replay"] = replay
        return JSONResponse(status_code=202, content=rerun)

    @application.post("/api/v1/agent-runs/{run_id}/submit-review")
    async def submit_review(run_id: str, request: SubmitReviewRequest) -> dict[str, Any]:
        run = _run_or_404(database, run_id)
        if run["status"] == "PENDING_REVIEW":
            review = run.get("review") or {}
            if _normalized_identity(str(review.get("submitted_by") or "")) == _normalized_identity(
                request.submitted_by
            ):
                run["idempotent_replay"] = True
                return run
        if run["status"] != "READY_FOR_REVIEW" or run["review_status"] != "DRAFT":
            raise api_error(
                409,
                "RUN_NOT_SUBMITTABLE",
                f"状态 {run['status']}/{run['review_status']} 不可送审",
            )
        if not run.get("platform_run_id"):
            raise api_error(409, "PLATFORM_RUN_MISSING", "运行没有可回写的平台注册记录")
        try:
            platform_response = await platform.review_run(
                run["platform_run_id"],
                action="SUBMIT",
                reviewer=request.submitted_by,
                reason=request.reason,
            )
        except PlatformError as error:
            raise api_error(
                502,
                "PLATFORM_REVIEW_FAILED",
                "平台送审失败，本地状态未改变",
                platform_error=error.as_dict(),
            ) from error
        submitted_at = utc_now()
        review = {
            "status": "PENDING_REVIEW",
            "submitted_by": request.submitted_by,
            "submitted_at": submitted_at,
            "reason": request.reason,
            "platform_response": platform_response,
            "execution_allowed": False,
        }
        database.add_review(
            run_id,
            from_status="DRAFT",
            to_status="PENDING_REVIEW",
            action="SUBMIT",
            reviewer=request.submitted_by,
            reason=request.reason,
            original=run.get("formal_strategy", []),
            platform_response=platform_response,
        )
        database.update_run(
            run_id,
            status="PENDING_REVIEW",
            review_status="PENDING_REVIEW",
            review_json=review,
        )
        database.upsert_step(
            run_id,
            9,
            "dual_review",
            "SUCCEEDED",
            {"submitted_by": request.submitted_by, "submitted_at": submitted_at},
        )
        result = database.get_run(run_id)
        result["idempotent_replay"] = False
        return result

    @application.post("/api/v1/agent-runs/{run_id}/review")
    async def decide_review(run_id: str, request: ReviewRequest) -> dict[str, Any]:
        run = _run_or_404(database, run_id)
        if run["status"] != "PENDING_REVIEW" or run["review_status"] != "PENDING_REVIEW":
            raise api_error(
                409,
                "RUN_NOT_REVIEWABLE",
                f"状态 {run['status']}/{run['review_status']} 不可审核",
            )
        submitted_by = str((run.get("review") or {}).get("submitted_by") or "")
        reviewer_identity = _normalized_identity(request.reviewed_by)
        separated_identities = {
            _normalized_identity(run["initiated_by"]),
            _normalized_identity(submitted_by),
        }
        if reviewer_identity in separated_identities:
            raise api_error(
                409,
                "REVIEWER_SEPARATION_REQUIRED",
                "批准人与运行发起人/送审人必须不同",
            )
        if request.decision in {ReviewDecision.MODIFIED, ReviewDecision.REJECTED} and not request.reason.strip():
            raise api_error(
                422,
                "REVIEW_REASON_REQUIRED",
                f"{request.decision.value} 必须填写原因",
            )
        modifications = request.modified_suggestions
        if request.decision == ReviewDecision.MODIFIED:
            if not modifications:
                raise api_error(
                    422,
                    "MODIFIED_SUGGESTIONS_REQUIRED",
                    "MODIFIED 必须提供 modified_suggestions",
                )
            _validate_hold_modifications(modifications)
        elif modifications:
            raise api_error(
                422,
                "UNEXPECTED_MODIFICATIONS",
                "仅 MODIFIED 决策可携带 modified_suggestions",
            )
        if not run.get("platform_run_id"):
            raise api_error(409, "PLATFORM_RUN_MISSING", "运行没有可回写的平台注册记录")
        action = {
            ReviewDecision.APPROVED: "APPROVE",
            ReviewDecision.MODIFIED: "MODIFY",
            ReviewDecision.REJECTED: "REJECT",
        }[request.decision]
        try:
            platform_response = await platform.review_run(
                run["platform_run_id"],
                action=action,
                reviewer=request.reviewed_by,
                reason=request.reason,
                modified_suggestions=modifications,
            )
        except PlatformError as error:
            raise api_error(
                502,
                "PLATFORM_REVIEW_FAILED",
                "平台审核回写失败，本地状态未改变",
                platform_error=error.as_dict(),
            ) from error
        formal_strategy = run.get("formal_strategy", [])
        if request.decision == ReviewDecision.MODIFIED:
            formal_strategy = _merge_hold_modifications(formal_strategy, modifications or [])
        reviewed_at = utc_now()
        review = {
            **(run.get("review") or {}),
            "status": request.decision.value,
            "decision": request.decision.value,
            "reviewed_by": request.reviewed_by,
            "reviewed_at": reviewed_at,
            "review_reason": request.reason,
            "modified_suggestions": modifications or [],
            "platform_response": platform_response,
            "execution_allowed": False,
        }
        database.add_review(
            run_id,
            from_status="PENDING_REVIEW",
            to_status=request.decision.value,
            action=action,
            reviewer=request.reviewed_by,
            reason=request.reason,
            original=run.get("formal_strategy", []),
            modified=formal_strategy,
            platform_response=platform_response,
        )
        database.update_run(
            run_id,
            status=request.decision.value,
            review_status=request.decision.value,
            review_json=review,
            formal_strategy_json=formal_strategy,
            completed_at=reviewed_at,
        )
        database.upsert_step(run_id, 10, "final_report", "RUNNING")
        report = await workflow.build_report_version(
            run_id, phase="FINAL", review=review
        )
        database.upsert_step(
            run_id,
            10,
            "final_report",
            "SUCCEEDED",
            {"version": report["version"], "decision": request.decision.value},
        )
        result = database.get_run(run_id)
        result["review_history"] = database.list_reviews(run_id)
        return result

    @application.get("/api/v1/agent-runs/{run_id}/report")
    async def get_report(
        run_id: str,
        format: ReportFormat = ReportFormat.json,
        version: int | None = Query(default=None, ge=1),
    ) -> Any:
        _run_or_404(database, run_id)
        report = database.get_report(run_id, version)
        if report is None:
            raise api_error(409, "REPORT_NOT_READY", "该运行尚未生成报告")
        headers = {
            "X-Report-Version": str(report["version"]),
            "X-Report-Phase": str(report["phase"]),
        }
        if format == ReportFormat.markdown:
            return PlainTextResponse(
                report["markdown"], media_type="text/markdown; charset=utf-8", headers=headers
            )
        if format == ReportFormat.html:
            return HTMLResponse(report["html"], headers=headers)
        return JSONResponse(
            {
                "report_id": report["report_id"],
                "run_id": run_id,
                "version": report["version"],
                "phase": report["phase"],
                "created_at": report["created_at"],
                "report": report["payload"],
                "execution_allowed": False,
            },
            headers=headers,
        )

    @application.get("/api/v1/agent-runs/{run_id}/messages")
    async def list_messages(run_id: str) -> dict[str, Any]:
        _run_or_404(database, run_id)
        messages = database.list_messages(run_id)
        return {
            "run_id": run_id,
            "count": len(messages),
            "messages": messages,
            "execution_allowed": False,
        }

    @application.post("/api/v1/agent-runs/{run_id}/messages", status_code=201)
    async def create_message(run_id: str, request: MessageCreate) -> JSONResponse:
        run = _run_or_404(database, run_id)
        user_message = database.add_message(
            run_id,
            role="user",
            created_by=request.created_by,
            content=request.content.strip(),
        )
        policies = run.get("policies") or {}
        policy_citations = policies.get("citations", []) if isinstance(policies, dict) else []
        citations = [*policy_citations]
        if not citations:
            citations = [
                {
                    "citation_id": item["evidence_id"],
                    "title": item["title"],
                    "source_reference": item["source"],
                    "snippet": item.get("citation") or item["title"],
                }
                for item in run.get("evidence", [])[:8]
            ]
        try:
            from .llm import answer_question

            answer = await asyncio.to_thread(
                answer_question,
                request.content,
                run,
                citations,
                settings=settings,
            )
        except Exception as error:
            answer = {
                "kind": "answer",
                "content": "当前无法调用文字解释模块。请以运行门禁、证据和 HOLD 执行安全占位为准。",
                "citation_ids": [item.get("citation_id") for item in citations if item.get("citation_id")],
                "citations": citations,
                "fallback_used": True,
                "fallback_reason": str(error),
                "model": None,
            }
        answer_citations = answer.get("citations") or [
            item for item in citations if item.get("citation_id") in set(answer.get("citation_ids") or [])
        ]
        assistant_message = database.add_message(
            run_id,
            role="assistant",
            created_by="power-trading-ai-agent",
            content=str(answer.get("content") or answer.get("text") or ""),
            citations=answer_citations,
            metadata={
                key: value
                for key, value in answer.items()
                if key not in {"content", "text", "citations"}
            },
        )
        return JSONResponse(
            status_code=201,
            content={
                "run_id": run_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "execution_allowed": False,
            },
        )

    @application.post("/api/v1/trading-draft-runs", status_code=202)
    async def create_trading_draft(request: TradingDraftRunCreate) -> dict[str, Any]:
        if request.business_date is None:
            raise api_error(422, "BUSINESS_DATE_REQUIRED", "交易日不能为空")
        run, replay = database.create_trading_draft(
            request_id=request.request_id,
            business_date=str(request.business_date),
            initiated_by=request.initiated_by,
            forecast_version=request.forecast_version,
            rule_version=request.rule_version,
            strategy_version=request.strategy_version,
            risk_aversion=request.risk_aversion,
            scenario_source=request.scenario_source,
            scenario_version=request.scenario_version,
        )
        run["run_type"] = "DAY_AHEAD_DRAFT"
        run["idempotent_replay"] = replay
        return run

    @application.post("/api/v1/trading-research-runs", status_code=202)
    async def create_research_run(request: TradingResearchRunCreate) -> dict[str, Any]:
        run, replay = database.create_trading_draft(
            request_id=request.request_id, business_date=str(request.business_date),
            initiated_by=request.initiated_by, strategy_version=request.strategy_version,
            risk_aversion=request.risk_aversion, input_source="PLATFORM",
        )
        if run["status"] in {"DRAFT", "RUNNING"}:
            schedule_research(run["run_id"])
        return {**run, "run_type": "TRADING_RESEARCH", "idempotent_replay": replay}

    def _draft_or_404(run_id: str) -> dict[str, Any]:
        try:
            run = database.get_trading_draft(run_id)
        except KeyError:
            raise api_error(404, "RUN_NOT_FOUND", f"未知日前草稿运行：{run_id}")
        run["run_type"] = "TRADING_RESEARCH" if run.get("input_source") == "PLATFORM" else "DAY_AHEAD_DRAFT"
        return run

    def _require_draft_integrity(run: dict[str, Any]) -> None:
        if (
            not run.get("input_sha256") or not run.get("draft_sha256")
            or content_sha256(run.get("input_snapshot")) != run["input_sha256"]
            or content_sha256(run.get("draft")) != run["draft_sha256"]
        ):
            raise api_error(409, "DRAFT_SNAPSHOT_UNVERIFIED", "快照未锁定或内容已变化，请创建新任务重新复核")

    def _update_draft(run: dict[str, Any], **changes: Any) -> dict[str, Any]:
        return database.update_trading_draft(run["run_id"], expected_revision=run["revision"], **changes)

    def _draft_strategy_metadata(run: dict[str, Any]) -> dict[str, Any]:
        return {
            "strategy_version": run.get("strategy_version") or "historical_cvar_v02",
            "risk_aversion": float(run.get("risk_aversion", 0.3)),
            "scenario_source": run.get("scenario_source"),
            "scenario_version": run.get("scenario_version"),
            "execution_allowed": False,
        }

    def _uploaded_snapshot_value(
        payload: dict[str, Any] | None,
        items: list[dict[str, Any]] | None,
        field: str,
    ) -> str | None:
        values = {
            str(value).strip()
            for value in [
                payload.get(field) if payload else None,
                *(
                    item.get(field)
                    for item in (items or [])
                    if isinstance(item, dict)
                ),
            ]
            if value is not None and str(value).strip()
        }
        if len(values) > 1:
            raise api_error(
                422,
                "INCONSISTENT_INPUT_VERSION",
                f"上传内容包含多个 {field}",
                field=field,
                values=sorted(values),
            )
        return next(iter(values), None)

    def _lock_uploaded_value(
        run: dict[str, Any], field: str, uploaded_value: str | None
    ) -> str | None:
        locked_value = run.get(field)
        if locked_value and uploaded_value and locked_value != uploaded_value:
            raise api_error(
                409,
                "INPUT_SNAPSHOT_MISMATCH",
                f"上传的 {field} 与任务锁定值不一致",
                field=field,
                locked_value=locked_value,
                uploaded_value=uploaded_value,
            )
        return locked_value or uploaded_value

    @application.post("/api/v1/trading-draft-runs/{run_id}/inputs")
    async def upload_trading_draft_inputs(run_id: str, request: Request) -> dict[str, Any]:
        run = _draft_or_404(run_id)
        if run.get("input_source") == "PLATFORM":
            raise api_error(409, "INPUT_SOURCE_LOCKED", "该任务从平台读取数据，不能再用手工文件覆盖")
        if run.get("input_sha256") or run.get("inputs") or run["status"] != "DRAFT":
            raise api_error(409, "INPUT_SNAPSHOT_LOCKED", "任务输入已冻结；更换数据请创建新任务，原任务不会被覆盖")
        content_type = request.headers.get("content-type", "")
        forecasts = None; rules = None; payload = None
        if "multipart/form-data" in content_type:
            form = await request.form(); upload = form.get("file")
            if upload is None: raise api_error(422, "FILE_REQUIRED", "请上传 CSV/JSON 文件")
            raw = await upload.read(); parsed_type = getattr(upload, "content_type", "")
        else:
            raw = await request.body(); parsed_type = content_type
        try:
            from .trading_draft import parse_input, validate_rows, validate_forecasts, build_draft, InputValidationError
            text = raw.decode("utf-8-sig")
            payload = json.loads(text) if text.lstrip().startswith("{") else None
            if payload is not None:
                rows = payload.get("records", payload.get("inputs", [])); forecasts = payload.get("forecasts"); rules = payload.get("rules")
            else:
                rows = parse_input(raw, parsed_type)
            errors = validate_rows(rows, run["business_date"], "SD")
            errors.extend(validate_forecasts(forecasts, run["business_date"]))
            if rules is not None and not isinstance(rules, dict):
                errors.append("rules 必须是对象")
            if errors:
                raise InputValidationError(errors)
            content_sha256({"records": rows, "forecasts": forecasts, "rules": rules})
            uploaded_scenario_source = _uploaded_snapshot_value(
                payload, forecasts, "scenario_source"
            )
            uploaded_scenario_version = _uploaded_snapshot_value(
                payload, forecasts, "scenario_version"
            )
            uploaded_forecast_version = _uploaded_snapshot_value(
                payload, forecasts, "forecast_version"
            )
            uploaded_rule_version = _uploaded_snapshot_value(
                payload, [rules] if isinstance(rules, dict) else [], "rule_version"
            )
            scenario_source = _lock_uploaded_value(
                run, "scenario_source", uploaded_scenario_source
            )
            scenario_version = _lock_uploaded_value(
                run, "scenario_version", uploaded_scenario_version
            )
            forecast_version = _lock_uploaded_value(
                run, "forecast_version", uploaded_forecast_version
            )
            rule_version = _lock_uploaded_value(
                run, "rule_version", uploaded_rule_version
            )
            if forecasts is not None and forecast_version:
                forecasts = [{**item, "forecast_version": item.get("forecast_version") or forecast_version} for item in forecasts]
            draft = build_draft(
                rows,
                forecasts,
                rules or {"rule_version": rule_version},
                business_date=run["business_date"],
                strategy_version=run.get("strategy_version") or "historical_cvar_v02",
                risk_aversion=float(run.get("risk_aversion", 0.3)),
                scenario_source=scenario_source,
                scenario_version=scenario_version,
            )
        except HTTPException:
            raise
        except InputValidationError as error:
            raise api_error(422, "INVALID_INPUT", "输入校验失败，未生成草稿", errors=error.errors) from error
        except Exception as error:
            raise api_error(422, "INVALID_INPUT", str(error)) from error
        strategy_metadata = {
            **_draft_strategy_metadata(run),
            "scenario_source": scenario_source,
            "scenario_version": scenario_version,
        }
        input_snapshot = {
            "records": rows,
            "forecasts": forecasts or [],
            "rules": rules or {},
            "forecast_version": forecast_version,
            "rule_version": rule_version,
            "strategy": strategy_metadata,
            "captured_at": utc_now(),
            "source_file_sha256": hashlib.sha256(raw).hexdigest(),
            "energy_unit": "MWh",
            "execution_allowed": False,
        }
        audit = list(run.get("audit") or [])
        audit.append({"action": "INPUT_INGESTED", "row_count": len(rows), "errors": errors,
                      "input_sha256": content_sha256(input_snapshot), "draft_sha256": content_sha256(draft),
                      **strategy_metadata, "execution_allowed": False, "at": utc_now()})
        draft_status = "READY_FOR_REVIEW" if not errors and draft.get("formal_gate") == "PASSED" else "BLOCKED"
        _update_draft(
            run, input_json=rows, input_snapshot_json=input_snapshot,
            draft_json=draft, error_json=errors,
            forecast_version=forecast_version, rule_version=rule_version,
            scenario_source=scenario_source, scenario_version=scenario_version,
            status=draft_status,
            review_status="DRAFT", audit_json=audit,
        )
        run = _draft_or_404(run_id)
        return {**run, "row_count": len(rows), "errors": errors}

    @application.get("/api/v1/trading-draft-runs/{run_id}")
    async def get_trading_draft(run_id: str) -> dict[str, Any]:
        return _draft_or_404(run_id)

    @application.get("/api/v1/trading-draft-runs/{run_id}/draft")
    async def get_trading_draft_detail(run_id: str) -> dict[str, Any]:
        run = _draft_or_404(run_id)
        return {"run_id": run_id, "draft": run.get("draft"), "status": run.get("status"), "execution_allowed": False}

    @application.post("/api/v1/trading-draft-runs/{run_id}/submit-review")
    async def submit_trading_draft(run_id: str, request: SubmitReviewRequest) -> dict[str, Any]:
        run = _draft_or_404(run_id)
        is_research = run.get("input_source") == "PLATFORM"
        ready = run.get("research", {}).get("research_status") == "READY" if is_research else (run.get("draft") or {}).get("formal_gate") == "PASSED"
        if run["status"] not in {"READY_FOR_REVIEW", "RESEARCH_READY", "MODIFIED"} or not ready:
            raise api_error(409, "DRAFT_BLOCKED", "仅待复核或修改后待重审的有效草稿可以送审")
        _require_draft_integrity(run)
        review = {"status": "PENDING_REVIEW", "submitted_by": request.submitted_by,
                  "submitted_at": utc_now(), "reason": request.reason,
                  "input_sha256": run["input_sha256"], "draft_sha256": run["draft_sha256"],
                  "review_scope": "RESEARCH_REPORT" if is_research else "MANUAL_DRAFT",
                  "research_sha256": content_sha256(run.get("research", {})),
                  **_draft_strategy_metadata(run), "execution_allowed": False}
        audit = list(run.get("audit") or [])
        audit.append({"action": "SUBMIT", "submitted_by": request.submitted_by,
                      "input_sha256": run["input_sha256"], "draft_sha256": run["draft_sha256"],
                      "reason": request.reason, **_draft_strategy_metadata(run),
                      "at": utc_now()})
        _update_draft(run, status="PENDING_REVIEW",
                                      review_status="PENDING_REVIEW",
                                      review_json=review, audit_json=audit,
                                      **({"steps_json": [{**item, "status": "RUNNING"} if item["sequence"] == 10 else item for item in run["steps"]]} if is_research else {}))
        run = _draft_or_404(run_id)
        return run

    @application.post("/api/v1/trading-draft-runs/{run_id}/review")
    async def review_trading_draft(run_id: str, request: TradingDraftReview) -> dict[str, Any]:
        run = _draft_or_404(run_id)
        if run.get("status") != "PENDING_REVIEW": raise api_error(409, "RUN_NOT_REVIEWABLE", "草稿尚未送审")
        if request.expected_revision != run["revision"]:
            raise api_error(409, "DRAFT_REVISION_MISMATCH", "审核页面版本已过期，请重新读取草稿后审核")
        _require_draft_integrity(run)
        submitted_review = run.get("review") or {}
        if any(submitted_review.get(key) != run[key] for key in ("input_sha256", "draft_sha256")):
            raise api_error(409, "REVIEW_SNAPSHOT_MISMATCH", "送审版本与当前草稿不一致，不能审核")
        is_research = run.get("input_source") == "PLATFORM"
        if is_research and submitted_review.get("research_sha256") != content_sha256(run["research"]):
            raise api_error(409, "REVIEW_SNAPSHOT_MISMATCH", "研究方案与送审版本不一致")
        if is_research and request.decision == "MODIFY":
            raise api_error(409, "RESEARCH_RECALCULATION_REQUIRED", "平台研究方案不能手改算法输出；更换策略参数请重新生成并复核")
        submitted_by = (run.get("review") or {}).get("submitted_by", "")
        excluded_reviewers = {run.get("initiated_by", ""), submitted_by}
        excluded_reviewers.update(event.get("reviewed_by", "") for event in run["audit"] if event.get("action") == "MODIFY")
        if _normalized_identity(request.reviewed_by) in {_normalized_identity(actor) for actor in excluded_reviewers}:
            raise api_error(409, "REVIEWER_SEPARATION_REQUIRED", "审核人与发起人、送审人及本草稿修改人必须不同")
        if request.decision in {"MODIFY", "REJECT"} and not request.reason.strip(): raise api_error(422, "REVIEW_REASON_REQUIRED", "修改或驳回必须填写原因")
        if request.decision == "MODIFY" and not request.modifications: raise api_error(422, "MODIFICATIONS_REQUIRED", "修改必须指定时段")
        if request.decision != "MODIFY" and request.modifications:
            raise api_error(422, "INVALID_MODIFICATION", "APPROVE/REJECT 不得携带修改内容，请使用 MODIFY 后重新送审")
        draft = json.loads(json.dumps(run.get("draft") or {}))
        modifications = []
        if request.decision == "MODIFY":
            from .trading_draft import apply_draft_modifications
            try:
                draft, modifications = apply_draft_modifications(
                    draft, run["input_snapshot"], request.modifications, reason=request.reason,
                )
            except (ValueError, TypeError, KeyError) as error:
                raise api_error(422, "INVALID_MODIFICATION", str(error)) from error
        next_status = {"APPROVE": "APPROVED", "MODIFY": "MODIFIED", "REJECT": "REJECTED"}[request.decision]
        review = {"status": next_status, "submitted_by": (run.get("review") or {}).get("submitted_by"),
                  "reviewed_by": request.reviewed_by, "reviewed_at": utc_now(),
                  "reason": request.reason, "modifications": modifications,
                  "input_sha256": run["input_sha256"], "draft_sha256": content_sha256(draft),
                  "review_scope": "RESEARCH_REPORT" if is_research else "MANUAL_DRAFT",
                  "research_sha256": content_sha256(run.get("research", {})),
                  **_draft_strategy_metadata(run), "execution_allowed": False}
        audit = list(run.get("audit") or [])
        audit.append({"action": request.decision, "reviewed_by": request.reviewed_by,
                      "modifications": modifications, "reason": request.reason,
                      "input_sha256": run["input_sha256"],
                      "old_draft_sha256": run["draft_sha256"], "new_draft_sha256": content_sha256(draft),
                      **_draft_strategy_metadata(run), "at": utc_now()})
        _update_draft(run, status=next_status, review_status=next_status,
                                      draft_json=draft, review_json=review, audit_json=audit,
                                      **({"steps_json": [{**item, "status": "SUCCEEDED" if request.decision == "APPROVE" else "BLOCKED"} if item["sequence"] == 10 else item for item in run["steps"]]} if is_research else {}))
        run = _draft_or_404(run_id)
        return run

    @application.get("/api/v1/trading-draft-runs/{run_id}/export")
    async def export_trading_draft(run_id: str, format: str = Query(default="json")) -> Any:
        run = _draft_or_404(run_id)
        normalized_format = format.lower()
        if normalized_format not in {"csv", "json"}:
            raise api_error(422, "INVALID_EXPORT_FORMAT", "format 仅支持 csv 或 json")
        if run.get("status") != "APPROVED":
            raise api_error(409, "EXPORT_NOT_ALLOWED", "仅独立审核批准的草稿可以导出；修改后必须重新送审")
        _require_draft_integrity(run)
        if any((run.get("review") or {}).get(key) != run[key] for key in ("input_sha256", "draft_sha256")):
            raise api_error(409, "REVIEW_SNAPSHOT_MISMATCH", "批准版本与当前草稿不一致，禁止导出")
        is_research = run.get("input_source") == "PLATFORM"
        if is_research and run["review"].get("research_sha256") != content_sha256(run["research"]):
            raise api_error(409, "REVIEW_SNAPSHOT_MISMATCH", "研究方案不属于已批准版本")
        draft = run.get("draft") or {}
        title = "交易研究方案 - 人工复核材料 - 不可自动提交" if is_research else "人工复核申报草稿 - 不可自动提交"
        strategy_metadata = _draft_strategy_metadata(run)
        audit = list(run.get("audit") or [])
        audit.append({"action": "EXPORTED", "format": normalized_format,
                      "input_sha256": run["input_sha256"], "draft_sha256": run["draft_sha256"],
                      **strategy_metadata, "at": utc_now()})
        _update_draft(run, audit_json=audit,
                      **({"steps_json": [{**item, "status": "SUCCEEDED"} if item["sequence"] == 11 else item for item in run["steps"]]} if is_research else {}))
        if normalized_format == "csv":
            rows = [
                {
                    "document_title": title,
                    **strategy_metadata,
                    "review_status": run.get("review_status"),
                    "business_date": run["business_date"],
                    "review_scope": run["review"].get("review_scope"),
                    "submitted_by": run["review"].get("submitted_by"),
                    "reviewed_by": run["review"].get("reviewed_by"),
                    "reviewed_at": run["review"].get("reviewed_at"),
                    "research_mode": run.get("research", {}).get("mode"),
                    "source_sha256": run.get("research", {}).get("source_sha256"),
                    "research_assumptions": "；".join(run.get("research", {}).get("assumptions", [])),
                    "input_sha256": run["input_sha256"], "draft_sha256": run["draft_sha256"],
                    **row,
                }
                for row in draft.get("records", [])
            ]
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["document_title", "period"])
            writer.writeheader()
            # Protect user-authored text from spreadsheet formula interpretation;
            # numeric negative prices remain numeric, not escaped text.
            writer.writerows({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for key, value in row.items()} for row in rows)
            return PlainTextResponse(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="manual-review-draft-{run_id}.csv"'})
        return JSONResponse({"title": title, "run_id": run_id, "strategy": strategy_metadata,
                             "input_sha256": run["input_sha256"], "draft_sha256": run["draft_sha256"],
                             "review": run.get("review"), "audit": audit,
                             "research": run.get("research", {}),
                             "draft": draft, "execution_allowed": False})

    @application.post("/api/v1/policies", status_code=201)
    async def upload_policy(request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type", "").lower()
        metadata: dict[str, Any]
        content: bytes
        if content_type.startswith("multipart/form-data"):
            try:
                form = await request.form()
            except Exception as error:
                raise api_error(
                    422,
                    "INVALID_MULTIPART",
                    "无法解析政策上传表单；请确认已安装 multipart 依赖",
                ) from error
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise api_error(422, "FILE_REQUIRED", "multipart 表单必须包含 file")
            content = await upload.read()
            metadata = {
                "filename": getattr(upload, "filename", None) or form.get("filename"),
                "market_code": form.get("market_code"),
                "title": form.get("title"),
                "version": form.get("version"),
                "effective_date": form.get("effective_date"),
                "expires_on": form.get("expires_on") or None,
                "source_authorized": _bool_value(form.get("source_authorized")),
                "source_reference": form.get("source_reference") or getattr(upload, "filename", "uploaded-file"),
            }
        elif "application/json" in content_type:
            try:
                metadata = await request.json()
                content = base64.b64decode(
                    str(metadata.get("content_base64") or ""), validate=True
                )
            except (ValueError, binascii.Error, json.JSONDecodeError) as error:
                raise api_error(422, "INVALID_BASE64", "content_base64 不是有效 Base64") from error
            metadata.setdefault("source_reference", metadata.get("filename"))
        else:
            raise api_error(
                415,
                "UNSUPPORTED_MEDIA_TYPE",
                "请使用 multipart/form-data 或 application/json",
            )
        required = [
            name
            for name in ("filename", "market_code", "title", "version", "effective_date", "source_reference")
            if not str(metadata.get(name) or "").strip()
        ]
        if required:
            raise api_error(422, "POLICY_METADATA_REQUIRED", "政策元数据不完整", missing=required)
        normalized_policy_market = str(metadata["market_code"]).strip().upper()
        if normalized_policy_market not in dict(MARKETS):
            raise api_error(
                404,
                "MARKET_NOT_FOUND",
                f"未知市场代码：{normalized_policy_market}",
            )
        metadata["market_code"] = normalized_policy_market

        def ingest() -> dict[str, Any]:
            from .knowledge import ingest_policy

            with closing(database.connect()) as connection:
                document = ingest_policy(
                    connection,
                    filename=str(metadata["filename"]),
                    content=content,
                    market_code=str(metadata["market_code"]),
                    title=str(metadata["title"]),
                    version=str(metadata["version"]),
                    effective_date=str(metadata["effective_date"]),
                    expires_on=metadata.get("expires_on"),
                    source_authorized=_bool_value(metadata.get("source_authorized")),
                    source_reference=str(metadata["source_reference"]),
                )
                return _to_dict(document)

        try:
            document = await asyncio.to_thread(ingest)
        except Exception as error:
            code = str(getattr(error, "code", "POLICY_INGESTION_FAILED"))
            status_code = 403 if code == "UNAUTHORIZED_SOURCE" else 422
            raise api_error(status_code, code, str(error)) from error
        database.add_audit(
            None,
            "POLICY_INGESTED",
            "policy-uploader",
            {
                "document_id": document.get("id"),
                "market_code": document.get("market_code"),
                "status": document.get("status"),
                "sha256": document.get("sha256"),
            },
        )
        return JSONResponse(
            status_code=201,
            content={"policy": document, "execution_allowed": False},
        )

    @application.get("/api/v1/policies")
    async def policies(
        market_code: str | None = Query(default=None, min_length=2, max_length=8),
        status: str | None = Query(default=None),
        q: str | None = Query(default=None, min_length=1, max_length=2000),
        business_date: str | None = Query(default=None),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        normalized_market = market_code.strip().upper() if market_code else None
        if normalized_market and normalized_market not in dict(MARKETS):
            raise api_error(404, "MARKET_NOT_FOUND", f"未知市场代码：{normalized_market}")

        def read() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            from .knowledge import list_policies, search_policy_chunks

            with closing(database.connect()) as connection:
                documents = [
                    _to_dict(item)
                    for item in list_policies(
                        connection, market_code=normalized_market, status=status
                    )
                ]
                hits = (
                    [
                        _to_dict(item)
                        for item in search_policy_chunks(
                            connection,
                            q,
                            market_code=normalized_market,
                            as_of_date=business_date,
                            limit=limit,
                        )
                    ]
                    if q
                    else []
                )
                return documents, hits

        try:
            documents, hits = await asyncio.to_thread(read)
        except Exception as error:
            code = str(getattr(error, "code", "POLICY_QUERY_FAILED"))
            raise api_error(422, code, str(error)) from error
        return {
            "count": len(documents),
            "policies": documents,
            "query": q,
            "citation_count": len(hits),
            "citations": hits,
            "execution_allowed": False,
        }

    return application


def _validate_hold_modifications(modifications: list[dict[str, Any]]) -> None:
    seen: set[int] = set()
    for item in modifications:
        try:
            period = int(item.get("period"))
        except (TypeError, ValueError) as error:
            raise api_error(422, "INVALID_PERIOD", "修改建议必须包含 1 至 24 的 period") from error
        if period not in range(1, 25) or period in seen:
            raise api_error(422, "INVALID_PERIOD", f"无效或重复的 period：{period}")
        seen.add(period)
        if str(item.get("action", "HOLD")).upper() != "HOLD":
            raise api_error(
                409,
                "STRATEGY_GATE_ENFORCED",
                "审核修改仍必须保持 HOLD",
                period=period,
            )
        volume = item.get("volume_mwh", item.get("quantity_mwh", 0))
        if volume not in {0, 0.0, None}:
            raise api_error(
                409,
                "STRATEGY_GATE_ENFORCED",
                "审核修改的电量必须为 0",
                period=period,
            )
        if item.get("price_yuan_per_mwh") is not None:
            raise api_error(
                409,
                "STRATEGY_GATE_ENFORCED",
                "HOLD 建议不得包含可执行价格",
                period=period,
            )


def _merge_hold_modifications(
    strategy: list[dict[str, Any]], modifications: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_period = {int(item["period"]): dict(item) for item in strategy}
    for modification in modifications:
        period = int(modification["period"])
        merged = {**by_period.get(period, {"period": period}), **modification}
        merged.update(
            {
                "action": "HOLD",
                "volume_mwh": 0,
                "price_yuan_per_mwh": None,
                "execution_allowed": False,
            }
        )
        by_period[period] = merged
    return [by_period[period] for period in sorted(by_period)]


app = create_app()
