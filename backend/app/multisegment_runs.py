"""Immutable local curve snapshots, independent review and audited draft exports."""
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
import uuid

from .multisegment_bidding import HEADERS, customer_template_rows


def canonical(value):
    # JSON object keys are strings after persistence. Normalize before sorting
    # so integer period keys have the same hash before and after a DB restart.
    normalized=json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf8')).hexdigest()


def connect(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS multisegment_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, initiator TEXT NOT NULL, input_json TEXT NOT NULL, result_json TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL, review_json TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS multisegment_exports (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, created_at TEXT NOT NULL, format TEXT NOT NULL, content_sha256 TEXT NOT NULL)')
    db.commit()
    return db


def identity(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise ValueError('Nonempty identity up to 128 characters required')
    return value.strip()


def create_run(db_path, initiator, inputs, result):
    initiator = identity(initiator)
    run_id = uuid.uuid4().hex
    stamp = datetime.now(timezone.utc).isoformat()
    checksum = digest(dict(inputs=inputs, result=result))
    with closing(connect(db_path)) as db, db:
        db.execute('INSERT INTO multisegment_runs VALUES (?,?,?,?,?,?,NULL)',
            (run_id, stamp, initiator, canonical(inputs), canonical(result), checksum))
    return get_run(db_path, run_id)


def get_run(db_path, run_id):
    if not re.fullmatch('[a-f0-9]{32}', run_id): raise KeyError('Unknown run')
    with closing(connect(db_path)) as db:
        row = db.execute('SELECT * FROM multisegment_runs WHERE run_id=?', (run_id,)).fetchone()
    if row is None: raise KeyError('Unknown run')
    result, inputs = json.loads(row['result_json']), json.loads(row['input_json'])
    if digest(dict(inputs=inputs,result=result)) != row['snapshot_sha256']:
        raise ValueError('Snapshot integrity mismatch')
    review = json.loads(row['review_json']) if row['review_json'] else None
    return dict(run_id=run_id, created_at=row['created_at'], initiator=row['initiator'],
        snapshot_sha256=row['snapshot_sha256'], review=review,
        review_status=review['decision'] if review else 'PENDING_REVIEW', result=result,
        execution_allowed=False, formal_action='HOLD')


def review_run(db_path, run_id, reviewer, decision, reason):
    run = get_run(db_path, run_id)
    reviewer = identity(reviewer)
    if reviewer.casefold() == run['initiator'].casefold(): raise ValueError('Independent reviewer required')
    if decision not in ('APPROVED', 'REJECTED'): raise ValueError('Invalid review decision')
    if not isinstance(reason,str) or not reason.strip() or len(reason)>2000: raise ValueError('Review reason required, max 2000 characters')
    if run['result']['status'] != 'RESEARCH_ONLY': raise ValueError('Blocked optimization cannot be approved/exported')
    review = dict(reviewer=reviewer, decision=decision, reason=reason.strip(),
        at=datetime.now(timezone.utc).isoformat(), scope='RESEARCH_DRAFT_ONLY',
        snapshot_sha256=run['snapshot_sha256'], execution_allowed=False)
    with closing(connect(db_path)) as db, db:
        updated = db.execute('UPDATE multisegment_runs SET review_json=? WHERE run_id=? AND review_json IS NULL',
            (canonical(review),run_id))
        if updated.rowcount != 1: raise ValueError('Review is immutable; generate a new version to change inputs or decision')
    return get_run(db_path,run_id)


def export_run(db_path, run_id, file_format, artifact_config=None):
    run = get_run(db_path,run_id)
    if run['review_status'] != 'APPROVED': raise ValueError('Independent research review must be approved before export')
    rows = customer_template_rows(run['result'])
    payload = dict(**run, document_title='人工复核申报草稿 · 不可自动提交',
        customer_template=dict(sheet='数据', headers=HEADERS, rows=rows))
    if file_format == 'json':
        content = json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8')
        media = 'application/json'
    elif file_format == 'csv':
        out = io.StringIO(newline=''); writer = csv.writer(out)
        writer.writerow(HEADERS); writer.writerows(rows)
        content = out.getvalue().encode('utf-8-sig'); media = 'text/csv'
    elif file_format == 'xlsx':
        config = artifact_config or {}
        node = os.environ.get('POWER_TRADING_ARTIFACT_NODE') or config.get('node')
        builder = os.environ.get('POWER_TRADING_XLSX_BUILDER') or config.get('builder')
        if not node or not builder or not Path(node).is_file() or not Path(builder).is_file():
            raise RuntimeError('Excel exporter is not configured on this server; CSV/JSON remain available')
        with tempfile.TemporaryDirectory(prefix='sd-curve-export-') as tmp:
            source, target = Path(tmp)/'snapshot.json', Path(tmp)/'draft.xlsx'
            source.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf8')
            arguments=[str(node),str(builder),str(source),str(target)]
            if config.get('template'):
                arguments.extend(['',str(config['template'])])
            try:
                subprocess.run(arguments,check=True,capture_output=True,timeout=90,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except (subprocess.SubprocessError,OSError) as exc:
                raise RuntimeError('Excel export failed; snapshot preserved, CSV/JSON remain available') from exc
            content = target.read_bytes()
        media = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        raise ValueError('Supported formats: csv, json, xlsx')
    with closing(connect(db_path)) as db, db:
        db.execute('INSERT INTO multisegment_exports(run_id,created_at,format,content_sha256) VALUES(?,?,?,?)',
            (run_id,datetime.now(timezone.utc).isoformat(),file_format,hashlib.sha256(content).hexdigest()))
    return content, media, f"{run['result']['business_date']}_人工复核申报草稿_不可自动提交.{file_format}"
