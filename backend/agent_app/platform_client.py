from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any, Callable

import httpx

from .config import Settings


class PlatformError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 502,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


class PlatformClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        base = settings.platform_base_url.rstrip("/")
        self.api_base = base if base.endswith("/api/v1") else f"{base}/api/v1"
        self.timeout = settings.platform_timeout_seconds
        self.poll_interval = settings.platform_poll_interval_seconds
        self.run_timeout = settings.platform_run_timeout_seconds
        self.headers = (
            {"X-API-Key": settings.platform_api_key}
            if settings.platform_api_key
            else {}
        )
        self._client = client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.request(
                method,
                f"{self.api_base}/{path.lstrip('/')}",
                params=params,
                json=json,
                headers=self.headers,
                timeout=timeout or self.timeout,
            )
        except httpx.TimeoutException as error:
            raise PlatformError(
                "PLATFORM_TIMEOUT", "平台请求超时", status_code=504, details=str(error)
            ) from error
        except httpx.RequestError as error:
            raise PlatformError(
                "PLATFORM_UNAVAILABLE", "无法连接现有交易平台", details=str(error)
            ) from error
        finally:
            if owns_client:
                await client.aclose()
        if response.is_error:
            try:
                body: Any = response.json()
            except ValueError:
                body = {"message": response.text[:1000]}
            detail = body.get("detail", body) if isinstance(body, dict) else body
            if isinstance(detail, dict):
                code = str(detail.get("code") or f"PLATFORM_HTTP_{response.status_code}")
                message = str(detail.get("message") or detail)
            else:
                code = f"PLATFORM_HTTP_{response.status_code}"
                message = str(detail)
            raise PlatformError(
                code,
                message,
                status_code=response.status_code,
                details=detail,
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise PlatformError(
                "PLATFORM_INVALID_JSON", "平台返回了无效 JSON", details=response.text[:1000]
            ) from error
        if not isinstance(payload, dict):
            raise PlatformError(
                "PLATFORM_INVALID_RESPONSE", "平台响应必须是 JSON 对象", details=payload
            )
        return payload

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health", timeout=min(self.timeout, 5.0))

    async def models(self) -> dict[str, Any]:
        return await self._request("GET", "/models")

    async def data_assets(self, market_code: str) -> dict[str, Any]:
        return await self._request(
            "GET", "/data-assets", params={"market_code": market_code.upper()}
        )

    async def hourly_data(
        self,
        market_code: str,
        market_date: str,
        *,
        data_version: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "market_code": market_code.upper(),
            "market_date": market_date,
        }
        if data_version:
            params["data_version"] = data_version
        return await self._request("GET", "/hourly-data", params=params)

    async def create_forecast_run(
        self,
        *,
        request_id: str,
        market_code: str,
        market_date: str,
        model_version: str,
        data_version: str,
        input_summary: dict[str, Any],
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/models/price-forecast/runs",
            json={
                "request_id": request_id,
                "market_code": market_code.upper(),
                "market_date": market_date,
                "model_id": "price-forecast",
                "model_version": model_version,
                "data_version": data_version,
                "parameters": parameters or {},
                "input_summary": input_summary,
                "timeout_seconds": self.run_timeout,
            },
            timeout=max(self.timeout, float(self.run_timeout) + 15.0),
        )

    async def get_model_run(self, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/model-runs/{run_id}")

    async def get_forecast_result(self, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/model-runs/{run_id}/results")

    async def wait_for_forecast_result(
        self,
        run: dict[str, Any],
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise PlatformError("PLATFORM_RUN_ID_MISSING", "平台未返回 run_id")
        deadline = monotonic() + self.run_timeout
        current = run
        while str(current.get("status", "")).upper() not in {
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "TIMED_OUT",
        }:
            if cancel_requested and cancel_requested():
                raise asyncio.CancelledError
            if monotonic() >= deadline:
                raise PlatformError(
                    "PLATFORM_RUN_TIMEOUT",
                    f"平台模型运行超过 {self.run_timeout} 秒",
                    status_code=504,
                )
            await asyncio.sleep(self.poll_interval)
            current = await self.get_model_run(run_id)
        status = str(current.get("status", "")).upper()
        if status != "SUCCEEDED":
            error = current.get("error") or {}
            raise PlatformError(
                str(error.get("code") or f"PLATFORM_RUN_{status}"),
                str(error.get("message") or f"平台模型运行状态为 {status}"),
                details=current,
            )
        return await self.get_forecast_result(run_id)

    async def cancel_run(self, run_id: str, actor: str, reason: str = "") -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/model-runs/{run_id}/cancel",
            json={"actor": actor, "reason": reason},
        )

    async def rerun(
        self, run_id: str, *, request_id: str, actor: str, reason: str = ""
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/model-runs/{run_id}/rerun",
            json={"request_id": request_id, "actor": actor, "reason": reason},
        )

    async def review_run(
        self,
        run_id: str,
        *,
        action: str,
        reviewer: str,
        reason: str = "",
        modified_suggestions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": action,
            "reviewer": reviewer,
            "reason": reason or None,
        }
        if modified_suggestions is not None:
            payload["modified_suggestions"] = modified_suggestions
        return await self._request(
            "POST", f"/model-runs/{run_id}/review", json=payload
        )
