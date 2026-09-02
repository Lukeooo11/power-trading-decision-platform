from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PolicyAgentError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, unavailable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.unavailable = unavailable


class PolicyAgentClient:
    """Read-only adapter for the separately deployed policy QA service."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float | None = None) -> None:
        self.base_url = (base_url or os.getenv("POWER_QA_ANSWER_API_BASE_URL", "http://127.0.0.1:8022")).rstrip("/")
        configured_timeout = timeout_seconds or float(os.getenv("POWER_QA_ANSWER_TIMEOUT_SECONDS", "90"))
        self.timeout_seconds = max(1.0, configured_timeout)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/answers/capabilities")

    def active_release(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/answers/active-release")

    def supported_scope(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/answers/supported-scope")

    def query(
        self,
        question: str,
        *,
        region_hint: str,
        market_hint: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "tenant_id": "power-trading-platform",
            "source_system": "power-trading-decision-platform",
            "region_hint": region_hint,
            "market_hint": market_hint,
            "require_strict_citations": True,
            "allow_instant_web_evidence": False,
            "allow_auto_evidence_intake": False,
            "allow_traceable_web_references": False,
            "conversation_history": conversation_history or [],
        }
        return self._request("POST", "/api/v1/answers/query", payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw)
                message = detail.get("detail") or detail.get("message") or raw
                if isinstance(message, dict):
                    message = message.get("message") or message.get("code") or json.dumps(message, ensure_ascii=False)
            except json.JSONDecodeError:
                message = raw or f"HTTP {error.code}"
            raise PolicyAgentError(str(message), status_code=error.code) from error
        except (URLError, TimeoutError, OSError) as error:
            raise PolicyAgentError("政策问答服务不可达，请确认独立 Agent 服务已启动", unavailable=True) from error

        try:
            result = json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            raise PolicyAgentError("政策问答服务返回了无法解析的响应") from error
        if not isinstance(result, dict):
            raise PolicyAgentError("政策问答服务返回格式不符合接口约定")
        return result
