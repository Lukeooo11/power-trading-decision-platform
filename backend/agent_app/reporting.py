"""Versioned, deterministic JSON/Markdown reports for agent run snapshots."""

from __future__ import annotations

import copy
import hashlib
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ReportBundle:
    version: int
    phase: str
    generated_at: str
    payload: dict[str, Any]
    json_text: str
    markdown: str
    html: str
    sha256: str

    def __getitem__(self, key: str) -> Any:
        """Expose the serialized bundle like a mapping for route adapters."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "phase": self.phase,
            "generated_at": self.generated_at,
            "payload": copy.deepcopy(self.payload),
            "json": self.json_text,
            "markdown": self.markdown,
            "html": self.html,
            "sha256": self.sha256,
        }


def build_report(
    run_snapshot: Mapping[str, Any],
    *,
    report_version: int = 1,
    phase: str | None = None,
    generated_at: str | None = None,
    narrative: Mapping[str, Any] | None = None,
    version: int | None = None,
    finalized: bool | None = None,
) -> ReportBundle:
    """Build immutable report representations without persisting anything."""

    if version is not None:
        if report_version != 1 and report_version != version:
            raise ValueError("report_version and version disagree")
        report_version = version
    if finalized is not None and phase is None:
        phase = "FINAL" if finalized else "DRAFT"
    payload = build_report_payload(
        run_snapshot,
        report_version=report_version,
        phase=phase,
        generated_at=generated_at,
        narrative=narrative,
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["content_sha256"] = digest
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    markdown = render_report_markdown(payload)
    return ReportBundle(
        version=report_version,
        phase=str(payload["phase"]),
        generated_at=str(payload["generated_at"]),
        payload=payload,
        json_text=json_text,
        markdown=markdown,
        html=render_report_html(markdown, title="电力交易 AI Agent 陪跑报告"),
        sha256=digest,
    )


def build_report_payload(
    run_snapshot: Mapping[str, Any],
    *,
    report_version: int = 1,
    phase: str | None = None,
    generated_at: str | None = None,
    narrative: Mapping[str, Any] | None = None,
    version: int | None = None,
    finalized: bool | None = None,
) -> dict[str, Any]:
    if version is not None:
        if report_version != 1 and report_version != version:
            raise ValueError("report_version and version disagree")
        report_version = version
    if finalized is not None and phase is None:
        phase = "FINAL" if finalized else "DRAFT"
    if report_version < 1:
        raise ValueError("report_version must be at least 1")
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    report_phase = (phase or _infer_phase(run_snapshot)).upper()

    gates = _mapping(run_snapshot.get("gates"))
    data_ready = _boolean_gate(run_snapshot, gates, "data_ready")
    policy_ready = _boolean_gate(run_snapshot, gates, "policy_ready")
    strategy_ready = _boolean_gate(run_snapshot, gates, "strategy_ready")
    citations = _report_citations(run_snapshot)
    research_signals = _safe_research_signals(
        _as_list(run_snapshot.get("research_signals", run_snapshot.get("signals", [])))
    )
    forecast = copy.deepcopy(run_snapshot.get("forecast") or run_snapshot.get("model_result") or {})
    formal_periods = _formal_hold_periods(run_snapshot, forecast)

    data_quality_details = copy.deepcopy(
        run_snapshot.get("data_quality")
        or run_snapshot.get("data_quality_result")
        or {}
    )
    policy_status = "READY" if policy_ready and citations else (
        "NEEDS_OCR" if _contains_policy_status(run_snapshot, "NEEDS_OCR") else "MISSING_POLICY"
    )
    model_section = {
        "model_id": run_snapshot.get("model_id") or _nested(forecast, "model_id"),
        "model_version": run_snapshot.get("model_version") or _nested(forecast, "model_version"),
        "data_version": run_snapshot.get("data_version") or _nested(forecast, "data_version"),
        "platform_run_id": run_snapshot.get("platform_run_id") or _nested(forecast, "run_id"),
        "forecast": forecast,
        "backtest": _backtest(run_snapshot, forecast),
    }

    payload: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_version": report_version,
        "phase": report_phase,
        "generated_at": timestamp,
        "run": {
            key: copy.deepcopy(run_snapshot.get(key))
            for key in (
                "run_id",
                "request_id",
                "parent_run_id",
                "market_code",
                "trading_subject",
                "business_date",
                "initiated_by",
                "status",
                "review_status",
                "created_at",
                "started_at",
                "completed_at",
            )
            if key in run_snapshot
        },
        "data_quality": {
            "ready": data_ready,
            "status": "READY" if data_ready else "NOT_READY",
            "details": data_quality_details,
        },
        "policy_basis": {
            "ready": bool(policy_ready and citations),
            "status": policy_status,
            "citations": citations,
        },
        "model": model_section,
        "research_signals": copy.deepcopy(research_signals),
        "formal_strategy": {
            "strategy_ready": strategy_ready,
            "action": "HOLD",
            "periods": formal_periods,
            "note": "正式策略不包含可执行买卖量价",
        },
        "review": copy.deepcopy(run_snapshot.get("review") or {}),
        "gaps": _report_gaps(run_snapshot, data_ready=data_ready, policy_ready=policy_ready),
        "execution_allowed": False,
        "disclaimer": "本报告为历史数据陪跑分析，不构成交易申报或下单指令。",
    }
    if narrative:
        payload["narrative"] = {
            key: copy.deepcopy(narrative.get(key))
            for key in (
                "content",
                "citation_ids",
                "citations",
                "fallback_used",
                "fallback_reason",
                "model",
            )
            if key in narrative
        }
    return payload


def render_report_markdown(report: Mapping[str, Any]) -> str:
    run = _mapping(report.get("run"))
    data_quality = _mapping(report.get("data_quality"))
    policy = _mapping(report.get("policy_basis"))
    model = _mapping(report.get("model"))
    formal = _mapping(report.get("formal_strategy"))
    review = _mapping(report.get("review"))
    lines = [
        "# 电力交易 AI Agent 陪跑报告",
        "",
        f"- 报告版本：{_md(report.get('report_version'))}",
        f"- 报告阶段：{_md(report.get('phase'))}",
        f"- 生成时间：{_md(report.get('generated_at'))}",
        f"- 运行编号：{_md(run.get('run_id'))}",
        f"- 市场/主体：{_md(run.get('market_code'))} / {_md(run.get('trading_subject'))}",
        f"- 业务日期：{_md(run.get('business_date'))}",
        "",
        "## 核心结论",
        "",
        "- `execution_allowed=false`；本服务不接入交易终端。",
        "- 正式策略为 `HOLD`，所有时段电量为 `0 MWh`。",
        f"- 数据质量门禁：{_ready_label(data_quality.get('ready'))}。",
        f"- 政策门禁：{_ready_label(policy.get('ready'))}（{_md(policy.get('status'))}）。",
        "",
    ]

    narrative = _mapping(report.get("narrative"))
    if narrative.get("content"):
        lines.extend(["## 解读", "", str(narrative["content"]), ""])

    lines.extend(["## 数据质量", ""])
    details = data_quality.get("details")
    if details:
        lines.extend(["```json", _pretty_json(details), "```", ""])
    else:
        lines.extend(["未提供额外数据质量明细。", ""])

    lines.extend(["## 政策依据", ""])
    citations = _as_list(policy.get("citations"))
    if citations:
        for citation in citations:
            item = _mapping(citation)
            location = _citation_location(item)
            lines.append(
                f"- [{_md(item.get('citation_id'))}] {_md(item.get('title'))} "
                f"{location}：{_md(item.get('snippet'))}"
            )
        lines.append("")
    else:
        lines.extend(["当前没有可用且已授权的政策原文引用。", ""])

    lines.extend(
        [
            "## 模型与回测",
            "",
            f"- 模型：{_md(model.get('model_id'))}",
            f"- 模型版本：{_md(model.get('model_version'))}",
            f"- 数据版本：{_md(model.get('data_version'))}",
            f"- 平台运行编号：{_md(model.get('platform_run_id'))}",
            "",
        ]
    )
    if model.get("backtest"):
        lines.extend(["```json", _pretty_json(model["backtest"]), "```", ""])
    else:
        lines.extend(["未提供回测指标。", ""])

    lines.extend(["## 研究性风险信号", ""])
    signals = _as_list(report.get("research_signals"))
    if signals:
        lines.extend(
            [
                "| 时段 | 预测价 | 参考价 | 绝对价差 | 风险等级 |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for index, raw_signal in enumerate(signals):
            signal = _mapping(raw_signal)
            lines.append(
                "| "
                + " | ".join(
                    _table(
                        value
                    )
                    for value in (
                        _period_label(signal, index),
                        _pick(
                            signal,
                            "forecast_price",
                            "predicted_price",
                            "prediction",
                            "day_ahead_p50_yuan_per_mwh",
                        ),
                        _pick(
                            signal,
                            "reference_price",
                            "actual_price",
                            "baseline_price",
                            "real_time_p50_yuan_per_mwh",
                        ),
                        _pick(
                            signal,
                            "absolute_spread",
                            "absolute_spread_yuan_per_mwh",
                            "spread",
                            "price_spread",
                            "spread_yuan_per_mwh",
                        ),
                        _signal_risk_label(signal),
                    )
                )
                + " |"
            )
        lines.append("")
    else:
        lines.extend(["未生成研究性风险信号。", ""])

    lines.extend(
        [
            "## 正式策略（HOLD）",
            "",
            "| 时段 | 动作 | 电量 (MWh) |",
            "| --- | --- | ---: |",
        ]
    )
    for index, raw_period in enumerate(_as_list(formal.get("periods"))):
        period = _mapping(raw_period)
        lines.append(
            f"| {_table(_period_label(period, index))} | HOLD | 0 |"
        )
    lines.append("")

    lines.extend(["## 审核记录", ""])
    if review:
        lines.extend(["```json", _pretty_json(review), "```", ""])
    else:
        lines.extend(["当前尚无审核记录。", ""])

    lines.extend(["## 缺口与待办", ""])
    gaps = _as_list(report.get("gaps"))
    if gaps:
        for gap in gaps:
            if isinstance(gap, Mapping):
                label = gap.get("message") or gap.get("detail") or gap.get("code") or gap
            else:
                label = gap
            lines.append(f"- {_md(label)}")
    else:
        lines.append("-无已记录缺口。")
    lines.extend(["", "---", "", str(report.get("disclaimer") or "")])
    return "\n".join(lines).rstrip() + "\n"


def render_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)


def serialize_report_json(report: Mapping[str, Any]) -> str:
    """Compatibility alias used by lightweight API clients."""
    return render_report_json(report)


def render_report_html(markdown: str | Mapping[str, Any], *, title: str = "Agent Report") -> str:
    """Render a dependency-free, safe web view preserving Markdown layout."""

    if isinstance(markdown, Mapping):
        markdown = render_report_markdown(markdown)
    escaped = html.escape(str(markdown))
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; color: #17212b; background: #f5f7f8; }}
    main {{ box-sizing: border-box; width: min(1100px, 100%); margin: 0 auto; padding: 24px; }}
    pre {{ box-sizing: border-box; margin: 0; padding: 24px; overflow-wrap: anywhere;
      white-space: pre-wrap; line-height: 1.65; background: #fff; border: 1px solid #d9e0e4;
      border-radius: 6px; font: 14px/1.65 system-ui, "Microsoft YaHei", sans-serif; }}
    @media (max-width: 640px) {{ main {{ padding: 0; }} pre {{ border-width: 0; border-radius: 0; padding: 16px; }} }}
  </style>
</head>
<body><main><pre>{escaped}</pre></main></body>
</html>"""


def _infer_phase(snapshot: Mapping[str, Any]) -> str:
    status = str(snapshot.get("review_status") or snapshot.get("status") or "").upper()
    return "FINAL" if status in {"APPROVED", "MODIFIED", "REJECTED"} else "DRAFT"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _boolean_gate(snapshot: Mapping[str, Any], gates: Mapping[str, Any], name: str) -> bool:
    if name in snapshot:
        return bool(snapshot[name])
    if name in gates:
        return bool(gates[name])
    return False


def _report_citations(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = snapshot.get("policy_citations")
    if not isinstance(candidates, (list, tuple)):
        candidates = snapshot.get("policies") or _mapping(snapshot.get("policy")).get("citations") or []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_as_list(candidates), start=1):
        item = _mapping(raw)
        snippet = item.get("snippet") or item.get("text") or item.get("citation")
        if not snippet:
            continue
        citation_id = str(item.get("citation_id") or item.get("id") or f"policy-{index}")
        if citation_id in seen:
            continue
        seen.add(citation_id)
        result.append(
            {
                "citation_id": citation_id,
                "document_id": item.get("document_id"),
                "title": item.get("title") or item.get("document_title") or "政策文档",
                "filename": item.get("filename"),
                "version": item.get("version"),
                "effective_date": item.get("effective_date"),
                "source_reference": item.get("source_reference") or item.get("source"),
                "page_number": item.get("page_number") or item.get("page"),
                "section": item.get("section"),
                "snippet": str(snippet),
            }
        )
    return result


def _safe_research_signals(signals: Sequence[Any]) -> list[dict[str, Any]]:
    """Copy only research metadata and force the non-executable contract."""
    safe: list[dict[str, Any]] = []
    for raw in signals:
        if not isinstance(raw, Mapping):
            continue
        item = copy.deepcopy(dict(raw))
        item["research_only"] = True
        item["execution_allowed"] = False
        # Research signals may describe a spread or probability, but never a
        # quantity/price order that a downstream client could mistake as an
        # executable suggestion.
        for key in (
            "action",
            "side",
            "quantity_mwh",
            "volume_mwh",
            "suggested_quantity_mwh",
            "suggested_volume_mwh",
            "price_yuan_per_mwh",
            "suggested_price_yuan_per_mwh",
            "bid_price",
            "bid_volume",
            "order_price",
            "order_quantity",
        ):
            item.pop(key, None)
        safe.append(item)
    return safe


def _contains_policy_status(snapshot: Mapping[str, Any], expected: str) -> bool:
    policy = snapshot.get("policy")
    if isinstance(policy, Mapping) and str(policy.get("status", "")).upper() == expected:
        return True
    for item in _as_list(snapshot.get("policies")):
        if isinstance(item, Mapping) and str(item.get("status", "")).upper() == expected:
            return True
    return False


def _formal_hold_periods(snapshot: Mapping[str, Any], forecast: Any) -> list[dict[str, Any]]:
    original = snapshot.get("formal_strategy") or snapshot.get("strategy") or []
    if isinstance(original, Mapping):
        original = original.get("periods") or original.get("suggestions") or []
    original_periods = _as_list(original)
    forecast_points = _forecast_points(forecast)
    periods: list[dict[str, Any]] = []
    for index in range(24):
        source = _mapping(original_periods[index]) if index < len(original_periods) else {}
        forecast_source = _mapping(forecast_points[index]) if index < len(forecast_points) else {}
        label = _period_label(source, index, fallback=forecast_source)
        item: dict[str, Any] = {
            "period": label,
            "action": "HOLD",
            "quantity_mwh": 0,
        }
        for key in ("datetime", "confidence", "reason", "reason_codes"):
            if source.get(key) is not None:
                item[key] = copy.deepcopy(source[key])
        periods.append(item)
    return periods


def _forecast_points(forecast: Any) -> list[Any]:
    if isinstance(forecast, list):
        return forecast
    if not isinstance(forecast, Mapping):
        return []
    for key in ("points", "predictions", "hourly_predictions", "forecast"):
        value = forecast.get(key)
        if isinstance(value, list):
            return value
    return []


def _period_label(value: Mapping[str, Any], index: int, *, fallback: Mapping[str, Any] | None = None) -> str:
    fallback = fallback or {}
    for source in (value, fallback):
        candidate = _pick(source, "period", "hour_label", "timestamp", "time", "hour")
        if candidate is not None:
            if isinstance(candidate, int):
                return f"{candidate:02d}:00"
            return str(candidate)
    return f"{index:02d}:00"


def _backtest(snapshot: Mapping[str, Any], forecast: Any) -> Any:
    for key in ("backtest", "backtest_metrics"):
        if key in snapshot and snapshot[key] is not None:
            return copy.deepcopy(snapshot[key])
    if isinstance(forecast, Mapping):
        for key in ("backtest", "backtest_metrics", "metrics"):
            if forecast.get(key) is not None:
                return copy.deepcopy(forecast[key])
    return {}


def _report_gaps(
    snapshot: Mapping[str, Any],
    *,
    data_ready: bool,
    policy_ready: bool,
) -> list[Any]:
    raw = snapshot.get("gaps", snapshot.get("missing_data", []))
    gaps = copy.deepcopy(_as_list(raw))
    codes = {
        str(item.get("code")) for item in gaps if isinstance(item, Mapping) and item.get("code")
    }
    if not data_ready and "DATA_GATE_NOT_READY" not in codes:
        gaps.append({"code": "DATA_GATE_NOT_READY", "message": "数据质量门禁未通过"})
    if not policy_ready and "POLICY_GATE_NOT_READY" not in codes:
        gaps.append({"code": "POLICY_GATE_NOT_READY", "message": "政策门禁未通过"})
    error = snapshot.get("error")
    if error and "RUN_ERROR" not in codes:
        gaps.append({"code": "RUN_ERROR", "message": copy.deepcopy(error)})
    return gaps


def _nested(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _pick(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _signal_risk_label(signal: Mapping[str, Any]) -> Any:
    direct = _pick(signal, "risk_level", "level", "signal")
    if direct is not None:
        return direct
    high = _mapping(signal.get("high_price_risk")).get("level")
    negative = _mapping(signal.get("negative_price_risk")).get("level")
    labels = []
    if high is not None:
        labels.append(f"高价:{high}")
    if negative is not None:
        labels.append(f"负价:{negative}")
    return " / ".join(labels) if labels else None


def _md(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("\n", " ")


def _table(value: Any) -> str:
    return _md(value).replace("|", "\\|")


def _ready_label(value: Any) -> str:
    return "通过" if bool(value) else "未通过"


def _citation_location(citation: Mapping[str, Any]) -> str:
    parts = []
    if citation.get("version"):
        parts.append(f"版本 {_md(citation['version'])}")
    if citation.get("page_number"):
        parts.append(f"第 {_md(citation['page_number'])} 页")
    if citation.get("section"):
        parts.append(f"章节 {_md(citation['section'])}")
    return f"（{'，'.join(parts)}）" if parts else ""


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportBundle",
    "build_report",
    "build_report_payload",
    "render_report_html",
    "render_report_json",
    "render_report_markdown",
    "serialize_report_json",
]
