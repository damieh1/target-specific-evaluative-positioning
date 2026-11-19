"""Small, strict I/O helpers; identifiers are always text."""
import csv
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing/duplicate CSV columns: {path}")
        for lineno, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed CSV row {path}:{lineno}")
            yield row


def write_csv(path, rows, columns=None):
    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """Do not silently disregard damaged checkpoints."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Damaged JSONL {path}:{n}; preserve it and inspect the last write.") from exc
    return rows


def video_id(value):
    s = str(value or "").strip()
    if re.fullmatch(r"\d{18,20}", s):
        return s
    hits = re.findall(r"_video_(\d{18,20})(?=[^\d]|$)", s)
    if hits and len(set(hits)) == 1:
        return hits[0]
    raise ValueError(f"No unambiguous integer video ID in {s[:120]!r}; scientific notation is not accepted")


def row_id(row):
    ids = []
    for key in ("video_id", "id", "source", "stem", "filename"):
        val = row.get(key, "").strip()
        if not val:
            continue
        try:
            ids.append(video_id(val))
        except ValueError:
            if key in {"video_id", "id"}:
                raise
    if not ids or len(set(ids)) != 1:
        raise ValueError("Missing or conflicting identifier columns")
    return ids[0]


def boolean(value):
    s = str(value).strip().lower()
    if s in {"true", "1", "1.0"}:
        return True
    if s in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"Unrecognized boolean {value!r}")


def norm_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def utc_date(value):
    s = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s):
        x = float(s)
        if x >= 1e12:
            x /= 1000
        dt = datetime.fromtimestamp(x, timezone.utc)
    else:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_unique(rows, key):
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {key}")


@contextmanager
def lock(path):
    """Linux/Colab process lock. Released by OS after a crash."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
