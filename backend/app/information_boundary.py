"""可审计的信息可得性边界。

业务确认：目标交易任务启动前，上一交易日（D-1）的日前结算价和实时
结算价都已经可以读取。因此价格历史特征和历史价格场景可以使用 D-1。

这不等于 D-1 的实际负荷、风光出力或目标日出清结果可用；电源/实际运行
数据仍由各自的发布时间审计和 ``supply_source_policy`` 控制。
"""
from __future__ import annotations

from datetime import date, timedelta


PRICE_SETTLEMENT_LAG_DAYS = 1
PRICE_SETTLEMENT_LAG_HOURS = 24
PRICE_SETTLEMENT_AVAILABLE_BEFORE_TRADE = True
SUPPLY_MINIMUM_LAG_DAYS = 2


def price_history_cutoff(target_date: str | date) -> date:
    """返回目标交易日前可用于价格历史的最后结算日期。"""
    target = target_date if isinstance(target_date, date) else date.fromisoformat(target_date)
    return target - timedelta(days=PRICE_SETTLEMENT_LAG_DAYS)


def price_boundary_metadata() -> dict[str, object]:
    return {
        "price_settlement_lag_days": PRICE_SETTLEMENT_LAG_DAYS,
        "price_settlement_lag_hours": PRICE_SETTLEMENT_LAG_HOURS,
        "price_settlement_available_before_trade": PRICE_SETTLEMENT_AVAILABLE_BEFORE_TRADE,
        "price_history_cutoff": "D-1",
        "supply_actual_minimum_lag_days": SUPPLY_MINIMUM_LAG_DAYS,
        "target_day_actual_price_used": False,
        "target_day_actual_supply_used": False,
    }
