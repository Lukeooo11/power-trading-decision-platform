from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


RUN_JSON_COLUMNS = {
    "input_snapshot_json": "input_snapshot",
    "forecast_json": "forecast",
    "signals_json": "signals",
    "formal_strategy_json": "formal_strategy",
    "policies_json": "policies",
    "review_json": "review",
    "missing_data_json": "missing_data",
}

RUN_MUTABLE_COLUMNS = {
    "status",
    "review_status",
    "data_version",
    "platform_run_id",
    "platform_strategy_ready",
    "data_ready",
    "policy_ready",
    "strategy_ready",
    "input_snapshot_json",
    "forecast_json",
    "signals_json",
    "formal_strategy_json",
    "policies_json",
    "review_json",
    "missing_data_json",
    "error_code",
    "error_message",
    "cancel_requested",
    "started_at",
    "completed_at",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class Database:
    def __init__(self, path: str | Path, retention_days: int = 180) -> None:
        self.path = Path(path)
        self.retention_days = retention_days
        self._schema_lock = threading.Lock()
        self._initialized = False

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._schema_lock:
            if self._initialized:
                return
            with closing(self.connect()) as connection, connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        parent_run_id TEXT,
                        market_code TEXT NOT NULL,
                        trading_subject TEXT NOT NULL,
                        business_date TEXT NOT NULL,
                        initiated_by TEXT NOT NULL,
                        status TEXT NOT NULL,
                        review_status TEXT NOT NULL DEFAULT 'DRAFT',
                        model_id TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        data_version TEXT,
                        platform_run_id TEXT,
                        platform_strategy_ready INTEGER NOT NULL DEFAULT 0,
                        data_ready INTEGER NOT NULL DEFAULT 0,
                        policy_ready INTEGER NOT NULL DEFAULT 0,
                        strategy_ready INTEGER NOT NULL DEFAULT 0,
                        execution_allowed INTEGER NOT NULL DEFAULT 0 CHECK (execution_allowed = 0),
                        input_snapshot_json TEXT NOT NULL DEFAULT '{}',
                        forecast_json TEXT NOT NULL DEFAULT '{}',
                        signals_json TEXT NOT NULL DEFAULT '[]',
                        formal_strategy_json TEXT NOT NULL DEFAULT '[]',
                        policies_json TEXT NOT NULL DEFAULT '{}',
                        review_json TEXT NOT NULL DEFAULT '{}',
                        missing_data_json TEXT NOT NULL DEFAULT '[]',
                        error_code TEXT,
                        error_message TEXT,
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(parent_run_id) REFERENCES agent_runs(run_id),
                        UNIQUE(request_id, market_code, trading_subject, business_date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_agent_runs_created
                        ON agent_runs(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_status
                        ON agent_runs(status, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS run_steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        detail_json TEXT NOT NULL DEFAULT '{}',
                        started_at TEXT,
                        completed_at TEXT,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                        UNIQUE(run_id, name)
                    );

                    CREATE TABLE IF NOT EXISTS evidence (
                        evidence_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        source TEXT NOT NULL,
                        title TEXT NOT NULL,
                        citation TEXT,
                        data_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id, created_at);

                    CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        phase TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        markdown TEXT NOT NULL,
                        html TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                        UNIQUE(run_id, version)
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        created_by TEXT NOT NULL,
                        content TEXT NOT NULL,
                        citations_json TEXT NOT NULL DEFAULT '[]',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id, created_at);

                    CREATE TABLE IF NOT EXISTS reviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        from_status TEXT NOT NULL,
                        to_status TEXT NOT NULL,
                        action TEXT NOT NULL,
                        reviewer TEXT NOT NULL,
                        reason TEXT,
                        original_json TEXT NOT NULL DEFAULT '[]',
                        modified_json TEXT NOT NULL DEFAULT '[]',
                        platform_response_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT,
                        event_type TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_logs(run_id, id);

                    CREATE TABLE IF NOT EXISTS trading_draft_runs (
                        run_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        market_code TEXT NOT NULL,
                        trading_subject TEXT NOT NULL,
                        granularity TEXT NOT NULL,
                        business_date TEXT NOT NULL,
                        initiated_by TEXT NOT NULL,
                        forecast_version TEXT,
                        rule_version TEXT,
                        strategy_version TEXT NOT NULL DEFAULT 'historical_cvar_v02',
                        risk_aversion REAL NOT NULL DEFAULT 0.3,
                        scenario_source TEXT,
                        scenario_version TEXT,
                        status TEXT NOT NULL,
                        review_status TEXT NOT NULL DEFAULT 'DRAFT',
                        execution_allowed INTEGER NOT NULL DEFAULT 0 CHECK (execution_allowed = 0),
                        input_json TEXT NOT NULL DEFAULT '[]',
                        input_snapshot_json TEXT NOT NULL DEFAULT '{}',
                        draft_json TEXT NOT NULL DEFAULT '{}',
                        error_json TEXT NOT NULL DEFAULT '[]',
                        review_json TEXT NOT NULL DEFAULT '{}',
                        audit_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(request_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_trading_drafts_date
                        ON trading_draft_runs(business_date, updated_at DESC);
                    """
                )
                # SQLite's CREATE TABLE IF NOT EXISTS does not add columns to an
                # existing installation, so keep this migration additive.
                existing_columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(trading_draft_runs)")
                }
                draft_migrations = {
                    "strategy_version": (
                        "TEXT NOT NULL DEFAULT 'historical_cvar_v02'"
                    ),
                    "risk_aversion": "REAL NOT NULL DEFAULT 0.3",
                    "scenario_source": "TEXT",
                    "scenario_version": "TEXT",
                    "input_snapshot_json": "TEXT NOT NULL DEFAULT '{}'",
                }
                for column, definition in draft_migrations.items():
                    if column not in existing_columns:
                        connection.execute(
                            f"ALTER TABLE trading_draft_runs ADD COLUMN {column} {definition}"
                        )
                connection.execute(
                    """UPDATE trading_draft_runs
                    SET strategy_version = 'historical_cvar_v02'
                    WHERE strategy_version IS NULL OR TRIM(strategy_version) = ''"""
                )
                connection.execute(
                    """UPDATE trading_draft_runs
                    SET risk_aversion = 0.3
                    WHERE risk_aversion IS NULL"""
                )
            self._initialized = True

    def prune_expired_runs(self) -> int:
        self.initialize()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM agent_runs WHERE created_at < ?", (cutoff,)
            )
            return max(0, cursor.rowcount)

    def create_trading_draft(
        self, *, request_id: str, business_date: str, initiated_by: str,
        forecast_version: str | None = None, rule_version: str | None = None,
        strategy_version: str = "historical_cvar_v02", risk_aversion: float = 0.3,
        scenario_source: str | None = None, scenario_version: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        self.initialize()
        strategy_version = str(strategy_version or "").strip()
        if not strategy_version:
            raise ValueError("strategy_version must not be empty")
        risk_aversion = float(risk_aversion)
        if not 0.0 <= risk_aversion <= 1.0:
            raise ValueError("risk_aversion must be between 0 and 1")
        with closing(self.connect()) as connection, connection:
            existing = connection.execute(
                "SELECT run_id FROM trading_draft_runs WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing:
                return self.get_trading_draft(existing["run_id"]), True
            run_id = f"draft-{uuid.uuid4().hex}"
            now = utc_now()
            strategy_snapshot = {
                "strategy_version": strategy_version,
                "risk_aversion": risk_aversion,
                "scenario_source": scenario_source,
                "scenario_version": scenario_version,
                "execution_allowed": False,
            }
            audit = [{
                "action": "CREATED",
                "initiated_by": initiated_by,
                **strategy_snapshot,
                "at": now,
            }]
            connection.execute(
                """INSERT INTO trading_draft_runs
                (run_id, request_id, market_code, trading_subject, granularity,
                 business_date, initiated_by, forecast_version, rule_version,
                 strategy_version, risk_aversion, scenario_source, scenario_version,
                 status, review_status, input_snapshot_json, audit_json,
                 created_at, updated_at)
                VALUES (?, ?, 'SD', 'retail', 'HOUR_24', ?, ?, ?, ?, ?, ?, ?, ?,
                        'DRAFT', 'DRAFT', ?, ?, ?, ?)""",
                (
                    run_id, request_id, business_date, initiated_by,
                    forecast_version, rule_version, strategy_version,
                    risk_aversion, scenario_source, scenario_version,
                    json_dumps({"strategy": strategy_snapshot}), json_dumps(audit),
                    now, now,
                ),
            )
        self.add_audit(None, "TRADING_DRAFT_CREATED", initiated_by,
                       {"run_id": run_id, "business_date": business_date,
                        **strategy_snapshot})
        return self.get_trading_draft(run_id), False

    def get_trading_draft(self, run_id: str) -> dict[str, Any]:
        self.initialize()
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT * FROM trading_draft_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        item = dict(row)
        for column, public_name, default in (
            ("input_json", "inputs", []), ("draft_json", "draft", None),
            ("input_snapshot_json", "input_snapshot", {}),
            ("error_json", "input_errors", []), ("review_json", "review", {}),
            ("audit_json", "audit", []),
        ):
            item[public_name] = json_loads(item.pop(column), default)
        item["execution_allowed"] = False
        return item

    def update_trading_draft(self, run_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"status", "review_status", "forecast_version", "rule_version",
                   "strategy_version", "risk_aversion", "scenario_source",
                   "scenario_version", "input_json", "input_snapshot_json",
                   "draft_json", "error_json", "review_json", "audit_json"}
        invalid = set(changes) - allowed
        if invalid:
            raise ValueError(f"Unsupported trading draft columns: {sorted(invalid)}")
        if not changes:
            return self.get_trading_draft(run_id)
        if "strategy_version" in changes and not str(changes["strategy_version"] or "").strip():
            raise ValueError("strategy_version must not be empty")
        if "risk_aversion" in changes:
            risk_aversion = float(changes["risk_aversion"])
            if not 0.0 <= risk_aversion <= 1.0:
                raise ValueError("risk_aversion must be between 0 and 1")
            changes["risk_aversion"] = risk_aversion
        normalized = {k: (json_dumps(v) if k.endswith("_json") and not isinstance(v, str) else v)
                      for k, v in changes.items()}
        normalized["updated_at"] = utc_now()
        assignments = ", ".join(f"{k} = ?" for k in normalized)
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(f"UPDATE trading_draft_runs SET {assignments} WHERE run_id = ?",
                                         [*normalized.values(), run_id])
            if cursor.rowcount == 0:
                raise KeyError(run_id)
        return self.get_trading_draft(run_id)

    def create_run(
        self,
        *,
        request_id: str,
        market_code: str,
        trading_subject: str,
        business_date: str,
        initiated_by: str,
        model_id: str,
        model_version: str,
        parent_run_id: str | None = None,
        data_version: str | None = None,
        input_snapshot: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        self.initialize()
        self.prune_expired_runs()
        market_code = market_code.upper()
        trading_subject = trading_subject.lower()
        with closing(self.connect()) as connection, connection:
            existing = connection.execute(
                """
                SELECT run_id FROM agent_runs
                WHERE request_id = ? AND market_code = ? AND trading_subject = ?
                  AND business_date = ?
                """,
                (request_id, market_code, trading_subject, business_date),
            ).fetchone()
            if existing:
                return self.get_run(existing["run_id"]), True
            run_id = f"agent-{uuid.uuid4().hex}"
            now = utc_now()
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, request_id, parent_run_id, market_code, trading_subject,
                    business_date, initiated_by, status, review_status, model_id,
                    model_version, data_version, input_snapshot_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', 'DRAFT', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    request_id,
                    parent_run_id,
                    market_code,
                    trading_subject,
                    business_date,
                    initiated_by,
                    model_id,
                    model_version,
                    data_version,
                    json_dumps(input_snapshot or {}),
                    now,
                    now,
                ),
            )
        self.add_audit(
            run_id,
            "RUN_CREATED",
            initiated_by,
            {"status": "QUEUED", "parent_run_id": parent_run_id},
        )
        return self.get_run(run_id), False

    def _row_to_run(self, row: sqlite3.Row) -> dict[str, Any]:
        run = dict(row)
        for column, public_name in RUN_JSON_COLUMNS.items():
            default: Any = [] if public_name in {
                "signals",
                "formal_strategy",
                "missing_data",
            } else {}
            run[public_name] = json_loads(run.pop(column, None), default)
        for name in (
            "platform_strategy_ready",
            "data_ready",
            "policy_ready",
            "strategy_ready",
            "execution_allowed",
            "cancel_requested",
        ):
            run[name] = bool(run[name])
        run["error"] = (
            {"code": run.pop("error_code"), "message": run.pop("error_message")}
            if run.get("error_code")
            else None
        )
        run.pop("error_code", None)
        run.pop("error_message", None)
        run["gates"] = {
            "market_supported": run["market_code"] == "SD"
            and run["trading_subject"] == "retail",
            "data_ready": run["data_ready"],
            "policy_ready": run["policy_ready"],
            "platform_strategy_ready": run["platform_strategy_ready"],
            "strategy_ready": run["strategy_ready"],
        }
        run["urls"] = {
            "self": f"/api/v1/agent-runs/{run['run_id']}",
            "events": f"/api/v1/agent-runs/{run['run_id']}/events",
            "report": f"/api/v1/agent-runs/{run['run_id']}/report",
            "messages": f"/api/v1/agent-runs/{run['run_id']}/messages",
        }
        return run

    def get_run(self, run_id: str, *, include_details: bool = True) -> dict[str, Any]:
        self.initialize()
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        run = self._row_to_run(row)
        if include_details:
            run["steps"] = self.list_steps(run_id)
            run["evidence"] = self.list_evidence(run_id)
            run["reports"] = self.list_report_versions(run_id)
        return run

    def find_run_by_idempotency_key(
        self, request_id: str, market_code: str, trading_subject: str, business_date: str
    ) -> dict[str, Any] | None:
        self.initialize()
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT run_id FROM agent_runs WHERE request_id = ? AND market_code = ?
                    AND trading_subject = ? AND business_date = ?
                """,
                (request_id, market_code.upper(), trading_subject.lower(), business_date),
            ).fetchone()
        return self.get_run(row["run_id"]) if row else None

    def update_run(self, run_id: str, **changes: Any) -> dict[str, Any]:
        self.initialize()
        invalid = set(changes) - RUN_MUTABLE_COLUMNS
        if invalid:
            raise ValueError(f"Unsupported run columns: {sorted(invalid)}")
        if not changes:
            return self.get_run(run_id)
        normalized: dict[str, Any] = {}
        for key, value in changes.items():
            if key in RUN_JSON_COLUMNS and not isinstance(value, str):
                normalized[key] = json_dumps(value)
            elif key in {
                "platform_strategy_ready",
                "data_ready",
                "policy_ready",
                "strategy_ready",
                "cancel_requested",
            }:
                normalized[key] = int(bool(value))
            else:
                normalized[key] = value
        normalized["updated_at"] = utc_now()
        assignments = ", ".join(f"{column} = ?" for column in normalized)
        values = [*normalized.values(), run_id]
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                f"UPDATE agent_runs SET {assignments} WHERE run_id = ?", values
            )
            if cursor.rowcount == 0:
                raise KeyError(run_id)
        return self.get_run(run_id)

    def is_cancel_requested(self, run_id: str) -> bool:
        self.initialize()
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return bool(row["cancel_requested"])

    def upsert_step(
        self,
        run_id: str,
        sequence: int,
        name: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        now = utc_now()
        started_at = now if status == "RUNNING" else None
        completed_at = now if status in {"SUCCEEDED", "FAILED", "SKIPPED", "CANCELLED"} else None
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO run_steps
                    (run_id, sequence, name, status, detail_json, started_at, completed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, name) DO UPDATE SET
                    sequence = excluded.sequence,
                    status = excluded.status,
                    detail_json = excluded.detail_json,
                    started_at = COALESCE(run_steps.started_at, excluded.started_at),
                    completed_at = excluded.completed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    sequence,
                    name,
                    status,
                    json_dumps(detail or {}),
                    started_at,
                    completed_at,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM run_steps WHERE run_id = ? AND name = ?", (run_id, name)
            ).fetchone()
        self.add_audit(
            run_id,
            "STEP_UPDATED",
            "agent-workflow",
            {"sequence": sequence, "name": name, "status": status, "detail": detail or {}},
        )
        return self._step_to_dict(row)

    @staticmethod
    def _step_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item.pop("id", None)
        item["detail"] = json_loads(item.pop("detail_json", None), {})
        return item

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM run_steps WHERE run_id = ? ORDER BY sequence, id", (run_id,)
            ).fetchall()
        return [self._step_to_dict(row) for row in rows]

    def add_evidence(
        self,
        run_id: str,
        *,
        kind: str,
        source: str,
        title: str,
        citation: str | None = None,
        data: dict[str, Any] | list[Any] | None = None,
        evidence_id: str | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        evidence_id = evidence_id or f"ev-{uuid.uuid4().hex}"
        now = utc_now()
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence
                    (evidence_id, run_id, kind, source, title, citation, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    run_id,
                    kind,
                    source,
                    title,
                    citation,
                    json_dumps(data or {}),
                    now,
                ),
            )
        self.add_audit(run_id, "EVIDENCE_ADDED", "agent-workflow", {"evidence_id": evidence_id, "kind": kind})
        return {
            "evidence_id": evidence_id,
            "run_id": run_id,
            "kind": kind,
            "source": source,
            "title": title,
            "citation": citation,
            "data": data or {},
            "created_at": now,
        }

    def list_evidence(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE run_id = ? ORDER BY created_at, evidence_id",
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["data"] = json_loads(item.pop("data_json", None), {})
            result.append(item)
        return result

    def save_report(
        self,
        run_id: str,
        *,
        phase: str,
        payload: dict[str, Any],
        markdown: str,
        html: str,
    ) -> dict[str, Any]:
        self.initialize()
        now = utc_now()
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM reports WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            version = int(row["version"])
            cursor = connection.execute(
                """
                INSERT INTO reports
                    (run_id, version, phase, payload_json, markdown, html, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, version, phase, json_dumps(payload), markdown, html, now),
            )
            report_id = cursor.lastrowid
        self.add_audit(run_id, "REPORT_SAVED", "agent-workflow", {"version": version, "phase": phase})
        return {
            "report_id": report_id,
            "run_id": run_id,
            "version": version,
            "phase": phase,
            "payload": payload,
            "markdown": markdown,
            "html": html,
            "created_at": now,
        }

    def get_report(self, run_id: str, version: int | None = None) -> dict[str, Any] | None:
        self.initialize()
        query = "SELECT * FROM reports WHERE run_id = ?"
        params: list[Any] = [run_id]
        if version is not None:
            query += " AND version = ?"
            params.append(version)
        query += " ORDER BY version DESC LIMIT 1"
        with closing(self.connect()) as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["report_id"] = item.pop("id")
        item["payload"] = json_loads(item.pop("payload_json", None), {})
        return item

    def list_report_versions(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT id, run_id, version, phase, created_at FROM reports WHERE run_id = ? ORDER BY version",
                (run_id,),
            ).fetchall()
        return [
            {"report_id": row["id"], **{key: row[key] for key in ("run_id", "version", "phase", "created_at")}}
            for row in rows
        ]

    def add_message(
        self,
        run_id: str,
        *,
        role: str,
        created_by: str,
        content: str,
        citations: Iterable[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        message_id = f"msg-{uuid.uuid4().hex}"
        now = utc_now()
        citation_list = list(citations or [])
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO messages
                    (message_id, run_id, role, created_by, content, citations_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    run_id,
                    role,
                    created_by,
                    content,
                    json_dumps(citation_list),
                    json_dumps(metadata or {}),
                    now,
                ),
            )
        self.add_audit(run_id, "MESSAGE_ADDED", created_by, {"message_id": message_id, "role": role})
        return {
            "message_id": message_id,
            "run_id": run_id,
            "role": role,
            "created_by": created_by,
            "content": content,
            "citations": citation_list,
            "metadata": metadata or {},
            "created_at": now,
        }

    def list_messages(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE run_id = ? ORDER BY created_at, message_id",
                (run_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["citations"] = json_loads(item.pop("citations_json", None), [])
            item["metadata"] = json_loads(item.pop("metadata_json", None), {})
            messages.append(item)
        return messages

    def add_review(
        self,
        run_id: str,
        *,
        from_status: str,
        to_status: str,
        action: str,
        reviewer: str,
        reason: str,
        original: list[dict[str, Any]] | None = None,
        modified: list[dict[str, Any]] | None = None,
        platform_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        now = utc_now()
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO reviews
                    (run_id, from_status, to_status, action, reviewer, reason,
                     original_json, modified_json, platform_response_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    from_status,
                    to_status,
                    action,
                    reviewer,
                    reason,
                    json_dumps(original or []),
                    json_dumps(modified or original or []),
                    json_dumps(platform_response or {}),
                    now,
                ),
            )
            review_id = cursor.lastrowid
        self.add_audit(
            run_id,
            f"REVIEW_{action}",
            reviewer,
            {"from": from_status, "to": to_status, "reason": reason},
        )
        return {
            "review_id": review_id,
            "run_id": run_id,
            "from_status": from_status,
            "to_status": to_status,
            "action": action,
            "reviewer": reviewer,
            "reason": reason,
            "original_suggestions": original or [],
            "modified_suggestions": modified or original or [],
            "platform_response": platform_response or {},
            "created_at": now,
            "execution_allowed": False,
        }

    def list_reviews(self, run_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        reviews = []
        for row in rows:
            item = dict(row)
            item["review_id"] = item.pop("id")
            item["original_suggestions"] = json_loads(item.pop("original_json", None), [])
            item["modified_suggestions"] = json_loads(item.pop("modified_json", None), [])
            item["platform_response"] = json_loads(item.pop("platform_response_json", None), {})
            item["execution_allowed"] = False
            reviews.append(item)
        return reviews

    def add_audit(
        self,
        run_id: str | None,
        event_type: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.initialize()
        now = utc_now()
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_logs (run_id, event_type, actor, details_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, event_type, actor, json_dumps(details or {}), now),
            )
            audit_id = int(cursor.lastrowid)
        return {
            "id": audit_id,
            "run_id": run_id,
            "event_type": event_type,
            "actor": actor,
            "details": details or {},
            "created_at": now,
        }

    def list_audit(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        self.initialize()
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM audit_logs WHERE run_id = ? AND id > ? ORDER BY id
                """,
                (run_id, after_id),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json_loads(item.pop("details_json", None), {})
            result.append(item)
        return result
