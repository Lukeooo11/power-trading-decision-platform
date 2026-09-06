"""Authorized policy-document ingestion and citation retrieval.

The module deliberately accepts an existing :class:`sqlite3.Connection` so the
agent service can keep policy data in the same database without coupling this
code to the application's persistence layer.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree


MAX_POLICY_BYTES = 20 * 1024 * 1024
ALLOWED_POLICY_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".json", ".pdf", ".docx"})
READY = "READY"
NEEDS_OCR = "NEEDS_OCR"


class PolicyIngestionError(ValueError):
    """A stable, API-friendly validation error raised before persistence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PolicyChunk:
    ordinal: int
    text: str
    page_number: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedPolicy:
    filename: str
    extension: str
    content_type: str
    sha256: str
    size_bytes: int
    status: str
    chunks: tuple[PolicyChunk, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        return data


@dataclass(frozen=True)
class PolicyDocument:
    id: str
    filename: str
    title: str
    market_code: str
    version: str
    effective_date: str
    expires_on: str | None
    source_reference: str
    source_authorized: bool
    sha256: str
    size_bytes: int
    content_type: str
    status: str
    created_at: str
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicySearchHit:
    citation_id: str
    document_id: str
    title: str
    filename: str
    market_code: str
    version: str
    effective_date: str
    expires_on: str | None
    source_reference: str
    page_number: int | None
    section: str | None
    snippet: str
    score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONTENT_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|(?:第[\d一-龥]+章|第[\d一-龥]+节)\s*.*|[\d一-龥]+[.、]　?\S.*)$"
)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


def parse_policy_document(
    filename: str,
    content: bytes,
    *,
    source_authorized: bool,
) -> ParsedPolicy:
    """Validate and extract a policy file without persisting its raw content."""

    if source_authorized is not True:
        raise PolicyIngestionError("UNAUTHORIZED_SOURCE", "政策文档必须确认来源授权后才能导入")
    if not isinstance(content, bytes):
        raise PolicyIngestionError("INVALID_CONTENT", "文档内容必须为字节数据")

    safe_filename = Path(filename).name
    extension = Path(safe_filename).suffix.lower()
    if extension not in ALLOWED_POLICY_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_POLICY_EXTENSIONS))
        raise PolicyIngestionError("UNSUPPORTED_FILE_TYPE", f"不支持的文件类型；可用类型：{supported}")
    if len(content) > MAX_POLICY_BYTES:
        raise PolicyIngestionError("FILE_TOO_LARGE", "政策文档不得超过 20 MiB")
    if not content:
        raise PolicyIngestionError("EMPTY_DOCUMENT", "政策文档内容为空")

    digest = hashlib.sha256(content).hexdigest()
    status = READY
    if extension == ".pdf":
        pages = _extract_pdf_pages(content)
        if not any(_normalize_text(text) for text in pages):
            status = NEEDS_OCR
            chunks: list[PolicyChunk] = []
        else:
            chunks = _chunks_from_pages(pages, fallback_section=Path(safe_filename).stem)
    elif extension == ".docx":
        pages = _extract_docx_pages(content)
        chunks = _chunks_from_pages(pages, fallback_section=Path(safe_filename).stem)
    else:
        text = _decode_text(content)
        if extension == ".json":
            chunks = _chunks_from_json(text)
        else:
            chunks = _chunks_from_pages([text], fallback_section=Path(safe_filename).stem, page_numbers=False)

    if status == READY and not chunks:
        raise PolicyIngestionError("EMPTY_DOCUMENT", "文档中没有可检索文本")

    return ParsedPolicy(
        filename=safe_filename,
        extension=extension,
        content_type=_CONTENT_TYPES[extension],
        sha256=digest,
        size_bytes=len(content),
        status=status,
        chunks=tuple(chunks),
    )


def init_policy_store(conn: sqlite3.Connection) -> None:
    """Create policy metadata, chunks, and a Chinese-capable trigram FTS index."""

    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS policy_documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            market_code TEXT NOT NULL,
            version TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            expires_on TEXT,
            source_reference TEXT NOT NULL,
            source_authorized INTEGER NOT NULL CHECK (source_authorized = 1),
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            content_type TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('READY', 'NEEDS_OCR')),
            created_at TEXT NOT NULL,
            UNIQUE (market_code, version, sha256)
        );

        CREATE TABLE IF NOT EXISTS policy_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL REFERENCES policy_documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            page_number INTEGER,
            section TEXT,
            text TEXT NOT NULL,
            UNIQUE (document_id, ordinal)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS policy_chunks_fts USING fts5(
            text,
            section,
            content='policy_chunks',
            content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS policy_chunks_ai AFTER INSERT ON policy_chunks BEGIN
            INSERT INTO policy_chunks_fts(rowid, text, section)
            VALUES (new.id, new.text, coalesce(new.section, ''));
        END;

        CREATE TRIGGER IF NOT EXISTS policy_chunks_ad AFTER DELETE ON policy_chunks BEGIN
            INSERT INTO policy_chunks_fts(policy_chunks_fts, rowid, text, section)
            VALUES ('delete', old.id, old.text, coalesce(old.section, ''));
        END;

        CREATE TRIGGER IF NOT EXISTS policy_chunks_au AFTER UPDATE ON policy_chunks BEGIN
            INSERT INTO policy_chunks_fts(policy_chunks_fts, rowid, text, section)
            VALUES ('delete', old.id, old.text, coalesce(old.section, ''));
            INSERT INTO policy_chunks_fts(rowid, text, section)
            VALUES (new.id, new.text, coalesce(new.section, ''));
        END;
        """
    )
    conn.commit()


def ingest_policy(
    conn: sqlite3.Connection,
    *,
    filename: str,
    content: bytes,
    market_code: str,
    title: str,
    version: str,
    effective_date: str | date,
    source_authorized: bool,
    source_reference: str,
    expires_on: str | date | None = None,
    document_id: str | None = None,
    created_at: str | None = None,
) -> PolicyDocument:
    """Parse and persist an authorized document; identical imports are idempotent."""

    parsed = parse_policy_document(filename, content, source_authorized=source_authorized)
    market = _required_text(market_code, "market_code").upper()
    normalized_title = _required_text(title, "title")
    normalized_version = _required_text(version, "version")
    normalized_source = _required_text(source_reference, "source_reference")
    effective = _iso_date(effective_date, "effective_date")
    expiry = _iso_date(expires_on, "expires_on") if expires_on is not None else None
    if expiry is not None and expiry < effective:
        raise PolicyIngestionError("INVALID_DATE_RANGE", "expires_on 不得早于 effective_date")

    init_policy_store(conn)
    existing = conn.execute(
        """
        SELECT d.*, count(c.id) AS chunk_count
        FROM policy_documents d
        LEFT JOIN policy_chunks c ON c.document_id = d.id
        WHERE d.market_code = ? AND d.version = ? AND d.sha256 = ?
        GROUP BY d.id
        """,
        (market, normalized_version, parsed.sha256),
    ).fetchone()
    if existing is not None:
        return _document_from_row(existing)

    policy_id = document_id or str(uuid.uuid4())
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO policy_documents (
                id, filename, title, market_code, version, effective_date,
                expires_on, source_reference, source_authorized, sha256,
                size_bytes, content_type, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                policy_id,
                parsed.filename,
                normalized_title,
                market,
                normalized_version,
                effective,
                expiry,
                normalized_source,
                parsed.sha256,
                parsed.size_bytes,
                parsed.content_type,
                parsed.status,
                timestamp,
            ),
        )
        conn.executemany(
            """
            INSERT INTO policy_chunks (document_id, ordinal, page_number, section, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (policy_id, chunk.ordinal, chunk.page_number, chunk.section, chunk.text)
                for chunk in parsed.chunks
            ],
        )

    return PolicyDocument(
        id=policy_id,
        filename=parsed.filename,
        title=normalized_title,
        market_code=market,
        version=normalized_version,
        effective_date=effective,
        expires_on=expiry,
        source_reference=normalized_source,
        source_authorized=True,
        sha256=parsed.sha256,
        size_bytes=parsed.size_bytes,
        content_type=parsed.content_type,
        status=parsed.status,
        created_at=timestamp,
        chunk_count=len(parsed.chunks),
    )


def list_policies(
    conn: sqlite3.Connection,
    *,
    market_code: str | None = None,
    status: str | None = None,
) -> list[PolicyDocument]:
    init_policy_store(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if market_code:
        clauses.append("d.market_code = ?")
        params.append(market_code.upper())
    if status:
        clauses.append("d.status = ?")
        params.append(status.upper())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT d.*, count(c.id) AS chunk_count
        FROM policy_documents d
        LEFT JOIN policy_chunks c ON c.document_id = d.id
        {where}
        GROUP BY d.id
        ORDER BY d.effective_date DESC, d.created_at DESC
        """,  # nosec B608: where contains only constant fragments
        params,
    ).fetchall()
    return [_document_from_row(row) for row in rows]


def search_policy_chunks(
    conn: sqlite3.Connection,
    query: str,
    *,
    market_code: str | None = None,
    as_of_date: str | date | None = None,
    limit: int = 10,
) -> list[PolicySearchHit]:
    """Return stable citation records, excluding OCR, future, and expired rules."""

    init_policy_store(conn)
    normalized_query = _required_text(query, "query")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    as_of = _iso_date(as_of_date, "as_of_date") if as_of_date is not None else None

    clauses = ["d.status = 'READY'", "d.source_authorized = 1"]
    params: list[Any] = []
    if market_code:
        clauses.append("d.market_code = ?")
        params.append(market_code.upper())
    if as_of:
        clauses.extend(["d.effective_date <= ?", "(d.expires_on IS NULL OR d.expires_on >= ?)"])
        params.extend([as_of, as_of])

    # FTS5 trigram does not match one- or two-character queries, so use LIKE for
    # those rare searches while retaining identical policy/date filtering.
    compact_query = re.sub(r"\s+", "", normalized_query)
    if len(compact_query) >= 3:
        match_clause = "policy_chunks_fts MATCH ?"
        search_params: list[Any] = [_fts_phrase(normalized_query)]
        score_expression = "bm25(policy_chunks_fts)"
        from_expression = "policy_chunks_fts JOIN policy_chunks c ON c.id = policy_chunks_fts.rowid"
    else:
        match_clause = "(c.text LIKE ? ESCAPE '\\' OR coalesce(c.section, '') LIKE ? ESCAPE '\\')"
        escaped = _escape_like(normalized_query)
        search_params = [f"%{escaped}%", f"%{escaped}%"]
        score_expression = "NULL"
        from_expression = "policy_chunks c"

    sql = f"""
        SELECT
            c.id AS chunk_id,
            c.page_number,
            c.section,
            c.text,
            d.id AS document_id,
            d.title,
            d.filename,
            d.market_code,
            d.version,
            d.effective_date,
            d.expires_on,
            d.source_reference,
            {score_expression} AS score
        FROM {from_expression}
        JOIN policy_documents d ON d.id = c.document_id
        WHERE {match_clause} AND {' AND '.join(clauses)}
        ORDER BY score ASC, d.effective_date DESC, c.ordinal ASC
        LIMIT ?
    """  # nosec B608: all interpolated SQL fragments are constants from this function
    rows = conn.execute(sql, [*search_params, *params, limit]).fetchall()
    return [_search_hit_from_row(row, normalized_query) for row in rows]


# Public alias kept concise for route/workflow integration.
search_policy = search_policy_chunks


def policy_gate_status(
    conn: sqlite3.Connection,
    market_code: str,
    as_of_date: str | date,
) -> dict[str, Any]:
    """Summarize whether at least one authorized, effective text policy exists."""

    init_policy_store(conn)
    market = _required_text(market_code, "market_code").upper()
    as_of = _iso_date(as_of_date, "as_of_date")
    rows = conn.execute(
        """
        SELECT d.*, count(c.id) AS chunk_count
        FROM policy_documents d
        LEFT JOIN policy_chunks c ON c.document_id = d.id
        WHERE d.market_code = ?
          AND d.effective_date <= ?
          AND (d.expires_on IS NULL OR d.expires_on >= ?)
        GROUP BY d.id
        ORDER BY d.effective_date DESC
        """,
        (market, as_of, as_of),
    ).fetchall()
    documents = [_document_from_row(row) for row in rows]
    ready_documents = [doc for doc in documents if doc.status == READY and doc.chunk_count > 0]
    if ready_documents:
        status = READY
    elif any(doc.status == NEEDS_OCR for doc in documents):
        status = NEEDS_OCR
    else:
        status = "MISSING_POLICY"
    return {
        "ready": bool(ready_documents),
        "status": status,
        "market_code": market,
        "as_of_date": as_of,
        "documents": [document.to_dict() for document in documents],
    }


class PolicyRepository:
    """Small facade suitable for dependency injection into FastAPI routes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        init_policy_store(conn)

    def ingest(self, **kwargs: Any) -> PolicyDocument:
        return ingest_policy(self.conn, **kwargs)

    def list(self, *, market_code: str | None = None, status: str | None = None) -> list[PolicyDocument]:
        return list_policies(self.conn, market_code=market_code, status=status)

    def search(
        self,
        query: str,
        *,
        market_code: str | None = None,
        as_of_date: str | date | None = None,
        limit: int = 10,
    ) -> list[PolicySearchHit]:
        return search_policy_chunks(
            self.conn,
            query,
            market_code=market_code,
            as_of_date=as_of_date,
            limit=limit,
        )

    def gate_status(self, market_code: str, as_of_date: str | date) -> dict[str, Any]:
        return policy_gate_status(self.conn, market_code, as_of_date)


def _extract_pdf_pages(content: bytes) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on deployment packaging
        raise PolicyIngestionError(
            "PARSER_UNAVAILABLE",
            "PDF 解析器未安装；请安装 pypdf 后重试",
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(content))
        return [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise PolicyIngestionError("INVALID_DOCUMENT", "PDF 文件无法解析") from exc


def _extract_docx_pages(content: bytes) -> list[str]:
    """Extract DOCX paragraphs with rendered/manual page-break awareness."""

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise PolicyIngestionError("INVALID_DOCUMENT", "DOCX 文件无法解析") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise PolicyIngestionError("INVALID_DOCUMENT", "DOCX 文件无法解析") from exc

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    pages: list[list[str]] = [[]]
    for paragraph in root.iter(f"{namespace}p"):
        paragraph_style = paragraph.find(f"./{namespace}pPr/{namespace}pStyle")
        style_name = (
            paragraph_style.attrib.get(f"{namespace}val", "")
            if paragraph_style is not None
            else ""
        )
        is_heading = style_name.casefold().startswith(("heading", "title")) or "标题" in style_name
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{namespace}t" and node.text:
                pieces.append(node.text)
            elif node.tag in {f"{namespace}tab"}:
                pieces.append("\t")
            elif node.tag == f"{namespace}br" and node.attrib.get(f"{namespace}type") == "page":
                text = _normalize_text("".join(pieces))
                if text:
                    pages[-1].append(f"# {text}" if is_heading else text)
                pieces = []
                pages.append([])
            elif node.tag == f"{namespace}lastRenderedPageBreak":
                text = _normalize_text("".join(pieces))
                if text:
                    pages[-1].append(f"# {text}" if is_heading else text)
                pieces = []
                pages.append([])
        text = _normalize_text("".join(pieces))
        if text:
            pages[-1].append(f"# {text}" if is_heading else text)
    return ["\n".join(paragraphs) for paragraphs in pages]


def _chunks_from_json(text: str) -> list[PolicyChunk]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyIngestionError("INVALID_DOCUMENT", "JSON 文件语法无效") from exc

    sections: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            sections.append((str(key), json.dumps(child, ensure_ascii=False, indent=2)))
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            sections.append((f"item[{index}]", json.dumps(child, ensure_ascii=False, indent=2)))
    else:
        sections.append(("root", json.dumps(value, ensure_ascii=False)))

    chunks: list[PolicyChunk] = []
    for section, section_text in sections:
        for part in _split_long_text(section_text):
            chunks.append(PolicyChunk(len(chunks), part, section=section))
    return chunks


def _chunks_from_pages(
    pages: Sequence[str],
    *,
    fallback_section: str,
    page_numbers: bool = True,
) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    current_section = fallback_section
    for page_index, raw_page in enumerate(pages, start=1):
        paragraphs: list[str] = []
        for raw_line in raw_page.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = _normalize_text(raw_line)
            if not line:
                continue
            if _HEADING_RE.match(line):
                if paragraphs:
                    for part in _split_long_text("\n".join(paragraphs)):
                        chunks.append(
                            PolicyChunk(
                                len(chunks),
                                part,
                                page_number=page_index if page_numbers else None,
                                section=current_section,
                            )
                        )
                    paragraphs = []
                current_section = line.lstrip("# ").strip()
            else:
                paragraphs.append(line)
        if paragraphs:
            for part in _split_long_text("\n".join(paragraphs)):
                chunks.append(
                    PolicyChunk(
                        len(chunks),
                        part,
                        page_number=page_index if page_numbers else None,
                        section=current_section,
                    )
                )
    return chunks


def _split_long_text(text: str, max_chars: int = 1600) -> Iterable[str]:
    normalized = text.strip()
    while normalized:
        if len(normalized) <= max_chars:
            yield normalized
            return
        cut = normalized.rfind("\n", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = normalized.rfind("。", 0, max_chars + 1)
            if cut >= 0:
                cut += 1
        if cut < max_chars // 2:
            cut = max_chars
        yield normalized[:cut].strip()
        normalized = normalized[cut:].strip()


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise PolicyIngestionError("INVALID_ENCODING", "文本文档必须使用 UTF-8 或 GB18030 编码")


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyIngestionError("INVALID_METADATA", f"{field} 不得为空")
    return value.strip()


def _iso_date(value: str | date, field: str) -> str:
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise PolicyIngestionError("INVALID_METADATA", f"{field} 必须为 YYYY-MM-DD") from exc
    return parsed.isoformat()


def _fts_phrase(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _snippet(text: str, query: str, width: int = 220) -> str:
    compact = text.strip()
    index = compact.casefold().find(query.casefold())
    if index < 0:
        return compact[:width] + ("…" if len(compact) > width else "")
    start = max(0, index - width // 3)
    end = min(len(compact), start + width)
    prefix = "…" if start else ""
    suffix = "…" if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def _row_mapping(row: sqlite3.Row | Sequence[Any], columns: Sequence[str]) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return dict(zip(columns, row, strict=False))


def _document_from_row(row: sqlite3.Row | Sequence[Any]) -> PolicyDocument:
    columns = (
        "id",
        "filename",
        "title",
        "market_code",
        "version",
        "effective_date",
        "expires_on",
        "source_reference",
        "source_authorized",
        "sha256",
        "size_bytes",
        "content_type",
        "status",
        "created_at",
        "chunk_count",
    )
    data = _row_mapping(row, columns)
    return PolicyDocument(
        id=str(data["id"]),
        filename=str(data["filename"]),
        title=str(data["title"]),
        market_code=str(data["market_code"]),
        version=str(data["version"]),
        effective_date=str(data["effective_date"]),
        expires_on=str(data["expires_on"]) if data.get("expires_on") is not None else None,
        source_reference=str(data["source_reference"]),
        source_authorized=bool(data["source_authorized"]),
        sha256=str(data["sha256"]),
        size_bytes=int(data["size_bytes"]),
        content_type=str(data["content_type"]),
        status=str(data["status"]),
        created_at=str(data["created_at"]),
        chunk_count=int(data["chunk_count"]),
    )


def _search_hit_from_row(row: sqlite3.Row | Sequence[Any], query: str) -> PolicySearchHit:
    columns = (
        "chunk_id",
        "page_number",
        "section",
        "text",
        "document_id",
        "title",
        "filename",
        "market_code",
        "version",
        "effective_date",
        "expires_on",
        "source_reference",
        "score",
    )
    data = _row_mapping(row, columns)
    citation_id = f"policy:{data['document_id']}:chunk:{data['chunk_id']}"
    return PolicySearchHit(
        citation_id=citation_id,
        document_id=str(data["document_id"]),
        title=str(data["title"]),
        filename=str(data["filename"]),
        market_code=str(data["market_code"]),
        version=str(data["version"]),
        effective_date=str(data["effective_date"]),
        expires_on=str(data["expires_on"]) if data.get("expires_on") is not None else None,
        source_reference=str(data["source_reference"]),
        page_number=int(data["page_number"]) if data.get("page_number") is not None else None,
        section=str(data["section"]) if data.get("section") else None,
        snippet=_snippet(str(data["text"]), query),
        score=float(data["score"]) if data.get("score") is not None else None,
    )


__all__ = [
    "ALLOWED_POLICY_EXTENSIONS",
    "MAX_POLICY_BYTES",
    "NEEDS_OCR",
    "READY",
    "ParsedPolicy",
    "PolicyChunk",
    "PolicyDocument",
    "PolicyIngestionError",
    "PolicyRepository",
    "PolicySearchHit",
    "ingest_policy",
    "init_policy_store",
    "list_policies",
    "parse_policy_document",
    "policy_gate_status",
    "search_policy_chunks",
    "search_policy",
]
