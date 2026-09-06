"""Constrained OpenAI-compatible explanations with deterministic fallbacks.

The language model is never treated as a source of trading facts. Numeric
results, gates, formal actions, and execution permission remain authoritative
data supplied by the deterministic workflow.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 30.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip() and self.api_key.strip() and self.model.strip())

    @classmethod
    def from_env(cls) -> "LLMSettings":
        raw_timeout = os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "30")
        try:
            timeout = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            timeout = 30.0
        return cls(
            base_url=os.getenv("AGENT_LLM_BASE_URL", "").rstrip("/"),
            api_key=os.getenv("AGENT_LLM_API_KEY", ""),
            model=os.getenv("AGENT_LLM_MODEL", ""),
            timeout_seconds=timeout,
        )

    @classmethod
    def from_object(cls, settings: Any) -> "LLMSettings":
        if isinstance(settings, cls):
            return settings
        return cls(
            base_url=str(getattr(settings, "llm_base_url", "")).rstrip("/"),
            api_key=str(getattr(settings, "llm_api_key", "")),
            model=str(getattr(settings, "llm_model", "")),
            timeout_seconds=float(getattr(settings, "llm_timeout_seconds", 30.0)),
        )


@dataclass(frozen=True)
class LLMResult:
    kind: str
    content: str
    citation_ids: tuple[str, ...]
    citations: tuple[dict[str, Any], ...]
    fallback_used: bool
    fallback_reason: str | None
    model: str | None

    @property
    def text(self) -> str:
        """Read-only alias used by generic chat/message serializers."""
        return self.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.content,
            # ``text`` is retained as a read-only compatibility alias for
            # clients that used the earlier message response shape.
            "text": self.content,
            "citation_ids": list(self.citation_ids),
            "citations": [dict(citation) for citation in self.citations],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "model": self.model,
            "authoritative_fields": [
                "numbers",
                "gates",
                "formal_strategy",
                "execution_allowed",
            ],
        }


class LLMAdapter:
    """Synchronous adapter for ``/chat/completions`` with strict validation."""

    def __init__(
        self,
        settings: LLMSettings | Any | None = None,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = (
            LLMSettings.from_env()
            if settings is None
            else LLMSettings.from_object(settings)
        )
        if client is not None and transport is not None:
            raise ValueError("pass either client or transport, not both")
        self._client = client or (
            httpx.Client(transport=transport, timeout=self.settings.timeout_seconds)
            if transport is not None
            else None
        )
        self._owns_client = client is None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": "llm" if self.enabled else "deterministic_fallback",
            "model": self.settings.model or None,
            "base_url_configured": bool(self.settings.base_url),
        }

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None

    def __enter__(self) -> "LLMAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def generate_narrative(
        self,
        run_snapshot: Mapping[str, Any],
        citations: Sequence[Mapping[str, Any]] = (),
    ) -> LLMResult:
        normalized_citations = _normalize_citations(citations)
        if not self.enabled:
            return _fallback_narrative(run_snapshot, normalized_citations, "LLM_NOT_CONFIGURED")

        authoritative = _authoritative_context(run_snapshot)
        schema = {
            "summary": "中文总结",
            "risk_explanation": "中文研究风险解释",
            "policy_conclusion": "中文政策结论",
            "citation_ids": ["必须来自已提供的 citation_id"],
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请仅根据下列确定性事实生成陪跑报告解读。"
                    "严禁新增数字、改变门禁、改变 HOLD 动作或声称可下单。"
                    "只输出 JSON，结构为："
                    f"{json.dumps(schema, ensure_ascii=False)}\n"
                    f"确定性事实：{_json(authoritative)}\n"
                    f"可用引用：{_json(normalized_citations)}"
                ),
            },
        ]
        try:
            payload = self._complete(messages)
            content, citation_ids = _validate_narrative_payload(
                payload,
                authoritative,
                normalized_citations,
            )
            return _success_result("narrative", content, citation_ids, normalized_citations, self.settings.model)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return _fallback_narrative(run_snapshot, normalized_citations, _reason(exc))

    def answer_question(
        self,
        question: str,
        run_snapshot: Mapping[str, Any],
        citations: Sequence[Mapping[str, Any]] = (),
    ) -> LLMResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        normalized_citations = _normalize_citations(citations)
        if not self.enabled:
            return _fallback_answer(
                normalized_question,
                run_snapshot,
                normalized_citations,
                "LLM_NOT_CONFIGURED",
            )

        authoritative = _authoritative_context(run_snapshot)
        schema = {
            "answer": "中文回答",
            "citation_ids": ["必须来自已提供的 citation_id"],
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：{normalized_question}\n"
                    "仅使用下列确定性事实和政策引用回答。"
                    "严禁新增数字、改变门禁、改变 HOLD 动作或声称可下单。"
                    "只输出 JSON，结构为："
                    f"{json.dumps(schema, ensure_ascii=False)}\n"
                    f"确定性事实：{_json(authoritative)}\n"
                    f"可用引用：{_json(normalized_citations)}"
                ),
            },
        ]
        try:
            payload = self._complete(messages)
            answer, citation_ids = _validate_answer_payload(
                payload,
                authoritative,
                normalized_citations,
                question=normalized_question,
            )
            return _success_result("answer", answer, citation_ids, normalized_citations, self.settings.model)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return _fallback_answer(
                normalized_question,
                run_snapshot,
                normalized_citations,
                _reason(exc),
            )

    def _complete(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        client = self._client
        if client is None:
            client = httpx.Client(timeout=self.settings.timeout_seconds)
            self._client = client
        response = client.post(
            _chat_completions_url(self.settings.base_url),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": list(messages),
            },
        )
        response.raise_for_status()
        body = response.json()
        raw_content = body["choices"][0]["message"]["content"]
        if not isinstance(raw_content, str):
            raise ValueError("LLM_CONTENT_NOT_TEXT")
        decoded = json.loads(_strip_json_fence(raw_content))
        if not isinstance(decoded, Mapping):
            raise ValueError("LLM_JSON_NOT_OBJECT")
        return decoded


def generate_narrative(
    run_snapshot: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]] = (),
    *,
    adapter: LLMAdapter | None = None,
    settings: LLMSettings | Any | None = None,
) -> dict[str, Any]:
    """Convenience entry point returning a JSON-serializable result."""

    owned_adapter = adapter is None
    active_adapter = adapter or LLMAdapter(settings or _application_settings())
    try:
        return active_adapter.generate_narrative(run_snapshot, citations).to_dict()
    finally:
        if owned_adapter:
            active_adapter.close()


def answer_question(
    question: str,
    run_snapshot: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]] = (),
    *,
    adapter: LLMAdapter | None = None,
    settings: LLMSettings | Any | None = None,
) -> dict[str, Any]:
    """Answer from run evidence; invalid or unavailable LLM output is replaced."""

    owned_adapter = adapter is None
    active_adapter = adapter or LLMAdapter(settings or _application_settings())
    try:
        return active_adapter.answer_question(question, run_snapshot, citations).to_dict()
    finally:
        if owned_adapter:
            active_adapter.close()


_SYSTEM_PROMPT = (
    "你是电力交易陪跑分析的文字解释器，不是数值模型或交易决策器。"
    "所有数字、数据/政策门禁、正式动作和 execution_allowed 由输入确定，"
    "不得补齐、重算、修改或推测。不得调用审核、交易或下单接口。"
)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_CHINESE_NUMBER_RE = re.compile(
    r"[零一二三四五六七八九十百千万亿两]+(?:元|兆瓦时|兆瓦|千瓦时|百分比|%|个时段|条信号|页)"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_FORBIDDEN_ACTION_RE = re.compile(
    r"(?:execution_allowed\s*[:=]\s*true|\b(?:BUY|SELL)\b|建议(?:立即)?(?:买入|卖出|下单)|"
    r"(?:正式(?:动作|策略)|执行动作|策略动作)\s*(?:为|是|:|：)?\s*(?:买入|卖出|下单)|"
    r"可执行交易|允许下单|策略可下单|门禁已通过)",
    re.IGNORECASE,
)


def _application_settings() -> Any:
    try:
        from .config import settings

        return settings
    except ImportError:
        return LLMSettings.from_env()


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def _normalize_citations(citations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(citations, start=1):
        if not isinstance(raw, Mapping):
            converter = getattr(raw, "to_dict", None)
            raw = converter() if callable(converter) else {}
        if not isinstance(raw, Mapping):
            raw = {}
        citation_id = str(raw.get("citation_id") or raw.get("id") or f"citation-{index}")
        if citation_id in seen:
            continue
        seen.add(citation_id)
        normalized.append(
            {
                "citation_id": citation_id,
                "title": str(raw.get("title") or raw.get("document_title") or "政策文档"),
                "version": raw.get("version"),
                "page_number": raw.get("page_number") or raw.get("page"),
                "section": raw.get("section"),
                "snippet": str(raw.get("snippet") or raw.get("text") or raw.get("citation") or ""),
                "source_reference": raw.get("source_reference") or raw.get("source"),
                "required": bool(raw.get("required", False)),
            }
        )
    return normalized


def _authoritative_context(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Allowlist evidence sent to the LLM, excluding private/raw platform data."""

    identity_keys = (
        "run_id",
        "request_id",
        "market_code",
        "trading_subject",
        "business_date",
        "model_id",
        "model_version",
        "data_version",
        "status",
        "review_status",
    )
    context: dict[str, Any] = {
        key: snapshot.get(key) for key in identity_keys if key in snapshot
    }
    aliases = {
        "gates": ("gates",),
        "data_quality": ("data_quality", "data_quality_result"),
        "policy": ("policy", "policies", "policy_result"),
        "model": ("model", "forecast", "model_result"),
        "backtest": ("backtest", "backtest_metrics"),
        "research_signals": ("research_signals", "signals"),
        "formal_strategy": ("formal_strategy", "strategy"),
        "review": ("review",),
        "gaps": ("gaps", "missing_data"),
    }
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in snapshot:
                context[target] = snapshot[candidate]
                break
    # This value is immutable regardless of what an upstream snapshot claims.
    context["execution_allowed"] = False
    return context


def _validate_narrative_payload(
    payload: Mapping[str, Any],
    authoritative: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    fields = []
    for key in ("summary", "risk_explanation", "policy_conclusion"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"MISSING_{key.upper()}")
        fields.append(value.strip())
    content = "\n\n".join(fields)
    if not citations:
        raise ValueError("MISSING_REQUIRED_CITATION")
    citation_ids = _validate_common(
        content,
        payload.get("citation_ids"),
        authoritative,
        citations,
    )
    return content, citation_ids


def _validate_answer_payload(
    payload: Mapping[str, Any],
    authoritative: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    *,
    question: str,
) -> tuple[str, tuple[str, ...]]:
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("MISSING_ANSWER")
    content = answer.strip()
    citation_ids = _validate_common(
        content,
        payload.get("citation_ids"),
        authoritative,
        citations,
        extra_numeric_source=question,
    )
    return content, citation_ids


def _validate_common(
    content: str,
    raw_citation_ids: Any,
    authoritative: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    *,
    extra_numeric_source: str = "",
) -> tuple[str, ...]:
    if not _CJK_RE.search(content):
        raise ValueError("RESPONSE_NOT_CHINESE")
    if _FORBIDDEN_ACTION_RE.search(content):
        raise ValueError("ACTION_OR_GATE_CHANGED")

    allowed_numbers = set(_NUMBER_RE.findall(_json(authoritative)))
    allowed_numbers.update(_NUMBER_RE.findall(_json(citations)))
    allowed_numbers.update(_NUMBER_RE.findall(extra_numeric_source))
    response_numbers = set(_NUMBER_RE.findall(content))
    if not response_numbers.issubset(allowed_numbers):
        raise ValueError("UNSUPPORTED_NUMBER")
    allowed_chinese_numbers = set(_CHINESE_NUMBER_RE.findall(_json(authoritative)))
    allowed_chinese_numbers.update(_CHINESE_NUMBER_RE.findall(_json(citations)))
    allowed_chinese_numbers.update(_CHINESE_NUMBER_RE.findall(extra_numeric_source))
    response_chinese_numbers = set(_CHINESE_NUMBER_RE.findall(content))
    if not response_chinese_numbers.issubset(allowed_chinese_numbers):
        raise ValueError("UNSUPPORTED_NUMBER")

    if not isinstance(raw_citation_ids, list) or not all(
        isinstance(item, str) for item in raw_citation_ids
    ):
        raise ValueError("INVALID_CITATIONS")
    citation_ids = tuple(dict.fromkeys(raw_citation_ids))
    known_ids = {str(item["citation_id"]) for item in citations}
    if not set(citation_ids).issubset(known_ids):
        raise ValueError("UNKNOWN_CITATION")
    explicit_required = {
        str(item["citation_id"]) for item in citations if bool(item.get("required"))
    }
    if explicit_required and not explicit_required.issubset(citation_ids):
        raise ValueError("MISSING_REQUIRED_CITATION")
    if citations and not citation_ids:
        raise ValueError("MISSING_REQUIRED_CITATION")
    return citation_ids


def _success_result(
    kind: str,
    content: str,
    citation_ids: Sequence[str],
    citations: Sequence[Mapping[str, Any]],
    model: str,
) -> LLMResult:
    selected = tuple(dict(item) for item in citations if item["citation_id"] in citation_ids)
    return LLMResult(
        kind=kind,
        content=content,
        citation_ids=tuple(citation_ids),
        citations=selected,
        fallback_used=False,
        fallback_reason=None,
        model=model,
    )


def _fallback_narrative(
    snapshot: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    reason: str,
) -> LLMResult:
    context = _authoritative_context(snapshot)
    market = context.get("market_code") or "未知市场"
    subject = context.get("trading_subject") or "未知主体"
    business_date = context.get("business_date") or "未知日期"
    gates = context.get("gates") or {}
    data_gate = _gate_label(gates, "data_ready", context.get("data_quality"))
    policy_gate = _gate_label(gates, "policy_ready", context.get("policy"))
    signals = context.get("research_signals")
    signal_count = len(signals) if isinstance(signals, list) else 0
    content = (
        f"本次陪跑对象为 {market}/{subject}，业务日期为 {business_date}。"
        f"数据质量门禁为{data_gate}，政策门禁为{policy_gate}。\n\n"
        f"共记录 {signal_count} 条研究性风险信号，仅用于分析。"
        "正式策略维持 HOLD，execution_allowed=false，不构成申报或交易指令。\n\n"
        + _fallback_policy_sentence(citations)
    )
    return _fallback_result("narrative", content, citations, reason)


def _fallback_answer(
    question: str,
    snapshot: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    reason: str,
) -> LLMResult:
    context = _authoritative_context(snapshot)
    gates = context.get("gates") or {}
    data_gate = _gate_label(gates, "data_ready", context.get("data_quality"))
    policy_gate = _gate_label(gates, "policy_ready", context.get("policy"))
    content = (
        f"针对“{question}”，当前可确认的结果是：数据质量门禁为{data_gate}，"
        f"政策门禁为{policy_gate}。正式策略维持 HOLD，"
        "execution_allowed=false。大模型输出未通过可验证性检查，因此使用确定性回答。"
        + _fallback_policy_sentence(citations)
    )
    return _fallback_result("answer", content, citations, reason)


def _fallback_policy_sentence(citations: Sequence[Mapping[str, Any]]) -> str:
    if not citations:
        return "当前没有可用且已授权的政策引用，政策结论不得通过门禁。"
    labels = []
    for citation in citations[:3]:
        location = citation.get("section") or (
            f"第 {citation['page_number']} 页" if citation.get("page_number") else "未标注位置"
        )
        labels.append(f"[{citation['citation_id']}] {citation['title']}（{location}）")
    return "可核验政策依据：" + "；".join(labels) + "。"


def _fallback_result(
    kind: str,
    content: str,
    citations: Sequence[Mapping[str, Any]],
    reason: str,
) -> LLMResult:
    selected = tuple(dict(item) for item in citations[:3])
    return LLMResult(
        kind=kind,
        content=content,
        citation_ids=tuple(str(item["citation_id"]) for item in selected),
        citations=selected,
        fallback_used=True,
        fallback_reason=reason,
        model=None,
    )


def _gate_label(gates: Any, key: str, fallback: Any) -> str:
    if isinstance(gates, Mapping) and key in gates:
        return "通过" if bool(gates[key]) else "未通过"
    if isinstance(fallback, Mapping):
        candidate = fallback.get("ready")
        if candidate is not None:
            return "通过" if bool(candidate) else "未通过"
        status = str(fallback.get("status", "")).upper()
        if status:
            return "通过" if status in {"READY", "PASSED", "SUCCEEDED"} else "未通过"
    return "未确认"


def _reason(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "LLM_TIMEOUT"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"LLM_HTTP_{exc.response.status_code}"
    if isinstance(exc, httpx.HTTPError):
        return "LLM_REQUEST_FAILED"
    message = str(exc).strip()
    return message[:120] if message else exc.__class__.__name__


__all__ = [
    "LLMAdapter",
    "LLMConfig",
    "LLMResult",
    "LLMSettings",
    "answer_question",
    "generate_narrative",
]


# Naming alias for callers that use the configuration terminology from the
# service settings documentation.
LLMConfig = LLMSettings
