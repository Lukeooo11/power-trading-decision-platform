"""Append September 2026 Shandong spot prices to the public price asset.

The model feature store already contains these observed labels. This creates a
small anonymized display asset; missing settlement labels remain null.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend" / "model_assets" / "sd-gfs24-d1-price-actual-supply-mixed-v6" / "shandong_feature_store_20260101_20260930.csv.gz"
TARGET = ROOT / "data" / "spot-prices.json"


def numeric(value):
    return None if pd.isna(value) else round(float(value), 6)


def main() -> None:
    existing = json.loads(TARGET.read_text(encoding="utf-8"))
    existing = [row for row in existing if not str(row.get("date", "")).startswith("2026-09-")]
    frame = pd.read_csv(SOURCE, compression="gzip", low_memory=False)
    frame = frame[frame["date"].astype(str).str.startswith("2026-09-")].copy()
    if len(frame) != 720 or frame[["date", "period"]].duplicated().any():
        raise ValueError("September feature-store price labels are not a complete unique 24-point-per-day set")
    rows = []
    for row in frame.sort_values(["date", "period"]).itertuples(index=False):
        da = numeric(getattr(row, "日前价格"))
        rt = numeric(getattr(row, "实时价格"))
        spread = round(da - rt, 6) if da is not None and rt is not None else None
        rows.append({
            "marketCode": "SD", "date": str(row.date), "time": f"{int(row.period):02d}:00",
            "dayAheadPriceYuanMwh": da, "realtimePriceYuanMwh": rt,
            "spreadYuanMwh": spread, "realtimeSpreadYuanMwh": spread,
            "intervalMinutes": 60, "sourceBatchId": "sd-september-prices-v1",
        })
    combined = sorted(existing + rows, key=lambda item: (item["date"], item["time"]))
    TARGET.write_text(json.dumps(combined, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "total_rows": len(combined), "september_rows": len(rows),
        "september_dates": sorted({row["date"] for row in rows}),
        "september_missing_day_ahead": sum(row["dayAheadPriceYuanMwh"] is None for row in rows),
        "september_missing_realtime": sum(row["realtimePriceYuanMwh"] is None for row in rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
