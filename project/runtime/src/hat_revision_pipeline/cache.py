from __future__ import annotations


from rineng_content_id import content_identity
import json
import os
import pickle
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _cache_io_path(path):
    path = Path(path)
    text = str(path)
    prefix = chr(92) * 2 + '?' + chr(92)
    if os.name == 'nt' and path.is_absolute() and not text.startswith(prefix):
        return Path(prefix + text)
    return path


class CacheMissError(RuntimeError):
    """A required cache record is absent or invalid in cache-only mode."""

    def __init__(self, kind: str, path: Path, reason: str):
        self.kind = str(kind)
        self.path = Path(path)
        self.reason = str(reason)
        super().__init__(
            f"Cache-only miss for {self.kind!r}: {self.reason}. "
            f"Expected a valid record at {self.path}. "
            "Run the matching quick/full mode first, then retry plot-only."
        )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest_payload(value: Any) -> str:
    return content_identity(canonical_json(value).encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> str:
    digest = content_identity()
    with _cache_io_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = content_identity()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def dependency_snapshot() -> dict[str, str]:
    result: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


class CacheStore:
    def __init__(
        self,
        root: str | Path,
        namespace: str = "standard-gorkov-v1",
        *,
        read_only: bool = False,
    ):
        self.root = Path(root)
        self.namespace = namespace
        self.read_only = bool(read_only)
        self.access_records: list[dict[str, str]] = []
        if not self.read_only:
            _cache_io_path(self.root).mkdir(parents=True, exist_ok=True)

    def path_for(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        create_parent: bool | None = None,
    ) -> Path:
        from .fe_solver_policy import solver_cache_payload
        payload = solver_cache_payload(kind, payload)
        identity = digest_payload({"namespace": self.namespace, "kind": kind, "payload": payload})
        directory = self.root / kind
        should_create = (not self.read_only) if create_parent is None else bool(create_parent)
        if should_create:
            _cache_io_path(directory).mkdir(parents=True, exist_ok=True)
        return directory / f"{identity}.pkl"

    def get_or_compute(
        self,
        kind: str,
        payload: dict[str, Any],
        producer: Callable[[], Any],
        *,
        recompute: bool = False,
    ) -> tuple[Any, str, Path]:
        from .fe_solver_policy import solver_cache_payload
        payload = solver_cache_payload(kind, payload)
        if self.read_only and recompute:
            raise ValueError("recompute=True is incompatible with a read-only cache")
        path = self.path_for(kind, payload, create_parent=not self.read_only)
        io_path = _cache_io_path(path)
        invalid_reason = "record does not exist"
        if io_path.exists() and not recompute:
            try:
                with io_path.open("rb") as stream:
                    record = pickle.load(stream)
                if not isinstance(record, dict):
                    invalid_reason = "record is not a cache mapping"
                elif digest_payload(record.get("payload")) != digest_payload(payload):
                    invalid_reason = "record payload does not match the requested computation"
                elif record.get("namespace") != self.namespace:
                    invalid_reason = "record namespace does not match the current pipeline"
                elif "value" not in record:
                    invalid_reason = "record has no cached value"
                else:
                    self.access_records.append({"kind": kind, "status": "cached", "path": str(path.relative_to(self.root))})
                    return record["value"], "cached", path
            except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError) as error:
                invalid_reason = f"record could not be read ({type(error).__name__})"
        if self.read_only:
            raise CacheMissError(kind, path, invalid_reason)
        value = producer()
        record = {
            "namespace": self.namespace,
            "payload": payload,
            "dependencies": dependency_snapshot(),
            "value": value,
        }
        with tempfile.NamedTemporaryFile("wb", dir=_cache_io_path(path.parent), delete=False) as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary = Path(stream.name)
        os.replace(_cache_io_path(temporary), io_path)
        self.access_records.append({"kind": kind, "status": "new", "path": str(path.relative_to(self.root))})
        return value, "new", path
