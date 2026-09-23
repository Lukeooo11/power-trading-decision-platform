# -*- coding: utf-8 -*-
"""电源滞后来源策略（supply lag source policy）。

背景
----
平台的电源滞后特征必须来自「申报前可得」的信息。原实现只接受
``sourceType == "FORECAST"``，导致 2026-07 起（源表只提供实际出力）
无法构造任何目标日的完整特征。

本模块把合格来源从单一 FORECAST 扩展为一组**预测类**来源，并按优先级取用：

==========  ==========================================================
优先级      来源与依据
==========  ==========================================================
1           ``FORECAST``                  源表标注的预测出力
2           ``PROXY_FORECAST``            基于 D-2 截止构造的目标日出力代理
3           ``SEASONAL_PROXY_FORECAST``   2025 同日历日风光 + D-2 其他字段
==========  ==========================================================

三者均为事前可得：代理资产的 ``modelVersion`` 为 ``sd-target-supply-proxy-d2-v1``，
``audit.train_cutoff`` 为 ``D-2``，``audit.minimum_observation_lag_hours`` 为 48，
``audit.target_day_actual_supply_used`` 为 false。

``ACTUAL`` 默认被排除
---------------------
日前申报发生在 D-1，此时 D-1 全天的实际出力尚不可知；把它作为滞后特征会引入
未来信息，使回测指标虚高。如需在研究中显式评估该口径，可设置环境变量
``POWER_TRADING_ALLOW_ACTUAL_LAG=1``；此时结果中会带 ``lag_leakage_risk`` 标记。
"""
from __future__ import annotations

import os
from typing import Any, Iterable

SOURCE_TYPE_FORECAST = "FORECAST"
SOURCE_TYPE_PROXY_FORECAST = "PROXY_FORECAST"
SOURCE_TYPE_SEASONAL_PROXY_FORECAST = "SEASONAL_PROXY_FORECAST"
SOURCE_TYPE_ACTUAL = "ACTUAL"

#: 预测类来源，按优先级从高到低。索引即优先级，越小越优先。
FORECAST_SOURCES: tuple[str, ...] = (
    SOURCE_TYPE_FORECAST,
    SOURCE_TYPE_PROXY_FORECAST,
    SOURCE_TYPE_SEASONAL_PROXY_FORECAST,
)

#: 允许以 null 出现、按声明默认值处理的电源字段。
#: testUnitMw（试验机组）在已入库的全部数据中恒为 0.0（7224 行唯一值为 0.0），
#: 代理资产中该字段为 null，语义等同「无试验机组出力」。
OPTIONAL_SUPPLY_COLUMNS: frozenset[str] = frozenset({"testUnitMw"})
SUPPLY_COLUMN_DEFAULTS: dict[str, float] = {"testUnitMw": 0.0}


def actual_lag_allowed() -> bool:
    """研究口径开关：是否允许把 ACTUAL 用作滞后特征（默认否）。"""
    return os.environ.get("POWER_TRADING_ALLOW_ACTUAL_LAG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def source_rank(source_type: Any, allow_actual: bool | None = None) -> int | None:
    """返回来源的优先级；不可用时返回 None。"""
    if allow_actual is None:
        allow_actual = actual_lag_allowed()
    value = str(source_type or "").strip().upper()
    if value in FORECAST_SOURCES:
        return FORECAST_SOURCES.index(value)
    if value == SOURCE_TYPE_ACTUAL and allow_actual:
        return len(FORECAST_SOURCES)
    return None


def is_eligible(source_type: Any, allow_actual: bool | None = None) -> bool:
    return source_rank(source_type, allow_actual) is not None


def select_supply_rows(rows: Iterable[dict[str, Any]],
                       allow_actual: bool | None = None,
                       ) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """按来源优先级为每个 (marketDate, period) 选出一行。

    返回 ``(lookup, audit)``；``audit`` 记录实际用到的来源分布与泄漏风险。
    """
    if allow_actual is None:
        allow_actual = actual_lag_allowed()
    best: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for row in rows:
        rank = source_rank(row.get("sourceType"), allow_actual)
        if rank is None:
            continue
        try:
            key = (str(row["marketDate"]), int(row["period"]))
        except (KeyError, TypeError, ValueError):
            continue
        current = best.get(key)
        if current is None or rank < current[0]:
            best[key] = (rank, row)
    lookup = {key: row for key, (_, row) in best.items()}
    used: dict[str, int] = {}
    for _, row in best.values():
        name = str(row.get("sourceType") or "").strip().upper()
        used[name] = used.get(name, 0) + 1
    audit = {
        "eligible_sources_by_priority": list(FORECAST_SOURCES),
        "actual_lag_allowed": bool(allow_actual),
        "lag_leakage_risk": bool(allow_actual and SOURCE_TYPE_ACTUAL in used),
        "selected_row_source_counts": used,
        "selected_row_count": len(lookup),
    }
    return lookup, audit


def supply_column_values(row: dict[str, Any], columns: Iterable[str],
                         ) -> tuple[dict[str, float] | None, list[str]]:
    """取出一行的电源字段值。

    必需字段缺失返回 ``(None, [])``；可选字段缺失按 ``SUPPLY_COLUMN_DEFAULTS``
    取值，并在第二个返回值中列出被默认化的字段名。
    """
    values: dict[str, float] = {}
    defaulted: list[str] = []
    for name in columns:
        raw = row.get(name)
        if raw is None:
            if name in SUPPLY_COLUMN_DEFAULTS:
                values[name] = float(SUPPLY_COLUMN_DEFAULTS[name])
                defaulted.append(name)
                continue
            return None, []
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None, []
        if number != number or number in (float("inf"), float("-inf")):
            return None, []
        values[name] = number
    return values, defaulted
