from __future__ import annotations

import asyncio
import math
from typing import Any

from .db import Database, content_sha256, utc_now
from .platform_client import PlatformClient, PlatformError


STEP_NAMES = (
    "市场与交易日校验", "读取平台数据与策略", "锁定来源快照", "政策证据检查",
    "价格预测检查", "剩余敞口检查", "研究分配方案", "正式申报门禁",
    "研究报告", "独立人工复核", "审核后导出",
)


def validate_plan(plan: dict[str, Any], run: dict[str, Any]) -> None:
    if plan.get("business_date") != run["business_date"] or plan.get("market_code") != "SD":
        raise ValueError("平台研究结果日期或市场与任务不一致")
    if plan.get("execution_allowed") is not False or plan.get("formal_gate") != "BLOCKED":
        raise ValueError("平台研究结果不得开放执行或正式申报")
    expected_strategy = {
        "historical_cvar_v02": "da-rt-split-historical-cvar-v0.2",
        "regime_cvar_v03": "da-rt-split-regime-cvar-v0.3",
        "similar_day_cvar_v04": "da-rt-split-similar-day-cvar-v0.4",
        "advanced_cvar_v05": "da-rt-split-hybrid-calibrated-newsvendor-v0.5",
        "joint_cvar_v06": "da-rt-split-joint-hybrid-cvar-v0.6",
    }[run["strategy_version"]]
    if plan.get("strategy_version") != expected_strategy or plan.get("risk_aversion") != run["risk_aversion"]:
        raise ValueError("平台策略版本或风险参数与任务不一致")
    if plan.get("research_status") not in {"READY", "BLOCKED"}:
        raise ValueError("平台研究状态无效")
    expected_forecast_run_id = run.get("platform_run_id")
    if expected_forecast_run_id and plan.get("forecast_run_id") != expected_forecast_run_id:
        raise ValueError("交易申报策略未绑定本次价格预测运行")
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != 24 or any(not isinstance(row, dict) or type(row.get("period")) is not int for row in records) or {row["period"] for row in records} != set(range(1, 25)):
        raise ValueError("平台研究结果必须包含完整24个时段")
    for row in records:
        strategy = row.get("strategy") or {}
        if strategy.get("execution_allowed") is not False or strategy.get("action") not in {"HOLD", "BUY_SPLIT"}:
            raise ValueError("平台研究时段状态无效")
        if plan["research_status"] == "READY" and (strategy["action"] != "BUY_SPLIT" or strategy.get("gate_status") != "RESEARCH_ONLY"):
            raise ValueError("研究状态与时段门禁不一致")
        if strategy["action"] == "BUY_SPLIT":
            values = [strategy.get(key) for key in ("remaining_exposure_mwh", "day_ahead_quantity_mwh", "real_time_reserved_mwh")]
            if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("平台研究量必须为非负有限数值")
            exposure, da, rt = values
            if abs(da + rt - exposure) > 0.00001:
                raise ValueError("日前研究量与实时保留敞口之和不等于剩余敞口")
    if plan["research_status"] == "READY":
        for summary_key, row_key in (("day_ahead_quantity_mwh", "day_ahead_quantity_mwh"), ("reserved_realtime_mwh", "real_time_reserved_mwh")):
            total = (plan.get("summary") or {}).get(summary_key)
            if type(total) not in (int, float) or not math.isfinite(total) or abs(total - sum(row["strategy"][row_key] for row in records)) > 0.00001:
                raise ValueError("平台研究汇总与24点明细不一致")
    content_sha256(plan)


def research_draft(plan: dict[str, Any]) -> dict[str, Any]:
    records = []
    for item in plan["records"]:
        strategy = item["strategy"]
        forecast = item.get("forecast") or {}
        da = forecast.get("day_ahead_price_yuan_per_mwh") or {}
        rt = forecast.get("real_time_price_yuan_per_mwh") or {}
        records.append({
            "period": item["period"], "load_forecast_mwh": item["load_mwh"] if item.get("load_basis") == "FORECAST" else None,
            "research_load_mwh": item.get("load_mwh"), "load_basis": item.get("load_basis"),
            "medium_position_mwh": item.get("medium_position_allocated_mwh"), "position_basis": item.get("position_basis"),
            "cleared_energy_mwh": item.get("cleared_energy_mwh"), "remaining_exposure_mwh": strategy.get("remaining_exposure_mwh"),
            **{f"day_ahead_price_{level}": da.get(level) for level in ("p10", "p50", "p90")},
            **{f"real_time_price_{level}": rt.get(level) for level in ("p10", "p50", "p90")},
            "research_quantity_mwh": strategy.get("day_ahead_quantity_mwh"),
            "research_real_time_reserved_mwh": strategy.get("real_time_reserved_mwh"),
            "research_lock_ratio": strategy.get("lock_ratio"), "research_action": strategy.get("action"),
            "research_price_lower": strategy.get("suggested_price_lower_yuan_per_mwh"),
            "research_price_upper": strategy.get("suggested_price_upper_yuan_per_mwh"),
            "research_optimization": strategy.get("optimization"),
            "research_scenario_audit": strategy.get("scenario_audit"),
            "research_quantile_calibration": strategy.get("quantile_calibration"),
            "research_spread_prediction": strategy.get("spread_prediction"),
            "research_confidence": strategy.get("confidence"),
            "research_raw_lock_ratio": strategy.get("lock_ratio_raw"),
            "research_lock_ratio_floor": strategy.get("lock_ratio_floor"),
            "research_lock_ratio_ceiling": strategy.get("lock_ratio_ceiling"),
            "suggested_quantity_mwh": 0, "suggested_price_lower": None, "suggested_price_upper": None,
            "action": "HOLD", "gate_status": "BLOCKED", "risk_flags": strategy.get("risk_flags") or [],
            "reason": strategy.get("reason"), "forecast_version": plan.get("forecast_version"),
            "rule_version": None, "execution_allowed": False,
        })
    return {"business_date": plan["business_date"], "market_code": "SD", "records": records,
            "formal_gate": "BLOCKED", "execution_allowed": False, "strategy_gate_status": "RESEARCH_ONLY",
            "strategy_version": plan["strategy_version"], "summary": {"suggested_day_ahead_quantity_mwh": 0,
            "research_day_ahead_quantity_mwh": plan["summary"].get("day_ahead_quantity_mwh")}}


class ResearchWorkflow:
    def __init__(self, db: Database, platform: PlatformClient, policy_provider) -> None:
        self.db, self.platform, self.policy_provider = db, platform, policy_provider

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        current = self.db.get_trading_draft(run_id)
        return self.db.update_trading_draft(run_id, expected_revision=current["revision"], **changes)

    async def run(self, run_id: str) -> None:
        steps = [{"sequence": index + 1, "name": name, "status": "PENDING", "detail": ""} for index, name in enumerate(STEP_NAMES)]
        def step(index, status, detail=""):
            steps[index].update(status=status, detail=detail, updated_at=utc_now())
        try:
            run = self.db.get_trading_draft(run_id)
            step(0, "SUCCEEDED", f"SD / retail / {run['business_date']}")
            step(1, "RUNNING")
            self.update(run_id, status="RUNNING", steps_json=steps)
            snapshot = run.get("input_snapshot") or {}
            if snapshot.get("platform_research"):
                if content_sha256(snapshot) != run.get("input_sha256"):
                    raise ValueError("冻结的平台研究快照校验失败")
                plan = snapshot["platform_research"]
                validate_plan(plan, run)
            else:
                plan = await self.platform.research_plan(run["business_date"], run["strategy_version"], run["risk_aversion"])
                validate_plan(plan, run)
                snapshot = {"platform_research": plan, "source": "platform:/api/strategy/research", "captured_at": utc_now(), "execution_allowed": False}
                self.update(run_id, input_snapshot_json=snapshot, forecast_version=plan.get("forecast_version"))
            step(1, "SUCCEEDED", "已读取目标日期的平台数据与确定性策略")
            step(2, "SUCCEEDED", content_sha256(snapshot))
            step(3, "RUNNING")
            self.update(run_id, steps_json=steps)
            try:
                policy = await asyncio.to_thread(self.policy_provider, "SD", run["business_date"])
            except Exception as error:
                policy = {"ready": False, "status": "UNAVAILABLE", "message": str(error), "citations": []}
            step(3, "SUCCEEDED" if policy.get("ready") else "BLOCKED", "政策引用不等于正式申报规则已确认")
            forecast_count = sum(all((row.get("forecast") or {}).get(key) for key in ("day_ahead_price_yuan_per_mwh", "real_time_price_yuan_per_mwh")) for row in plan["records"])
            step(4, "SUCCEEDED" if forecast_count == 24 else "BLOCKED", f"预测覆盖 {forecast_count}/24 时段")
            exposure_count = sum(row["strategy"].get("remaining_exposure_mwh") is not None for row in plan["records"])
            step(5, "SUCCEEDED" if exposure_count == 24 else "BLOCKED", f"敞口可计算 {exposure_count}/24 时段")
            ready = plan.get("research_status") == "READY"
            step(6, "SUCCEEDED" if ready else "BLOCKED", "复用平台分配方案，未重复优化或改写数量")
            step(7, "BLOCKED", "研究方案不可直接申报；正式规则和分时持仓口径仍需核验")
            step(8, "SUCCEEDED", "研究量、数据来源、风险与缺口已汇总")
            if not ready:
                step(9, "BLOCKED", "研究输入不完整，不能批准为完整研究方案")
                step(10, "BLOCKED", "待研究数据补齐后新建任务")
            research = {**plan, "policy_evidence": policy, "review_scope": "RESEARCH_REPORT", "execution_allowed": False}
            draft = research_draft(plan)
            current = self.db.get_trading_draft(run_id)
            audit = [*current["audit"], {"action": "PLATFORM_RESEARCH_GENERATED", "at": utc_now(),
                "input_sha256": current["input_sha256"], "draft_sha256": content_sha256(draft),
                "source_sha256": plan.get("source_sha256"), "execution_allowed": False}]
            self.update(run_id, status="RESEARCH_READY" if ready else "BLOCKED", draft_json=draft,
                        research_json=research, steps_json=steps, error_json=plan.get("missing_data", []), audit_json=audit)
        except asyncio.CancelledError:
            # Leave the frozen snapshot intact; service startup resumes pending research.
            raise
        except Exception as error:
            for entry in steps:
                if entry["status"] == "RUNNING":
                    entry.update(status="FAILED", detail=str(error))
            self.update(run_id, status="FAILED", steps_json=steps, error_json=[error.as_dict() if isinstance(error, PlatformError) else str(error)])
