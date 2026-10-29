"""Persistent client for the Kylin system vector-engine C++ SDK."""

from __future__ import annotations

import ctypes.util
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Sequence


class KylinVectorEngineError(RuntimeError):
    """Raised when the Kylin vector-engine bridge cannot complete an operation."""


class KylinVectorEngineClient:
    """Reuse one SDK connection through a small JSON-lines C++ bridge."""

    def __init__(
        self,
        *,
        app_id: str = "hust-kylin-os-memory",
        bridge_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._closed = False
        executable = Path(
            bridge_path or os.environ.get("KYLIN_VECTOR_ENGINE_BRIDGE", "") or _build_bridge()
        )
        if not executable.is_file():
            raise KylinVectorEngineError(f"Kylin vector bridge not found: {executable}")
        try:
            self._process = subprocess.Popen(
                [str(executable), app_id],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise KylinVectorEngineError(f"cannot start Kylin vector bridge: {exc}") from exc
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        try:
            ready = self._read_response(timeout=15)
            if not ready.get("ok") or ready.get("stage") != "ready":
                raise KylinVectorEngineError(_format_error(ready))
        except Exception:
            self.close()
            raise

    def ensure_collection(self, collection: str, dimension: int) -> dict[str, Any]:
        return self.request(
            {"op": "ensure_collection", "collection": collection, "dimension": dimension}
        )

    def upsert(self, collection: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"ok": True, "count": 0}
        return self.request({"op": "upsert", "collection": collection, "rows": list(rows)})

    def search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        top_k: int,
        expression: str = "",
        timeout_ms: int = 30000,
        output_fields: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        batches = self.search_many(
            collection,
            [vector],
            top_k=top_k,
            expression=expression,
            timeout_ms=timeout_ms,
            output_fields=output_fields,
        )
        return batches[0]

    def search_many(
        self,
        collection: str,
        vectors: Sequence[Sequence[float]],
        *,
        top_k: int,
        expression: str = "",
        timeout_ms: int = 30000,
        output_fields: Sequence[str] = (),
    ) -> list[list[dict[str, Any]]]:
        """Search multiple vectors with one IPC round trip and one SDK call."""

        if not vectors:
            return []
        response = self.request(
            {
                "op": "search_batch",
                "collection": collection,
                "vectors": [[float(value) for value in vector] for vector in vectors],
                "top_k": top_k,
                "expression": expression,
                "timeout_ms": timeout_ms,
                "output_fields": list(dict.fromkeys(str(field) for field in output_fields)),
            }
        )
        batches = response.get("batches", [])
        if not isinstance(batches, list) or len(batches) != len(vectors):
            raise KylinVectorEngineError("Kylin vector batch search returned invalid batches")
        normalized: list[list[dict[str, Any]]] = []
        for hits in batches:
            if not isinstance(hits, list):
                raise KylinVectorEngineError("Kylin vector search returned invalid hits")
            normalized.append([dict(hit) for hit in hits])
        return normalized

    def delete(self, collection: str, ids: Sequence[int]) -> dict[str, Any]:
        return self.request(
            {"op": "delete", "collection": collection, "ids": [int(value) for value in ids]}
        )

    def drop_collection(self, collection: str) -> dict[str, Any]:
        return self.request({"op": "drop_collection", "collection": collection})

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise KylinVectorEngineError("Kylin vector bridge is not running")
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise KylinVectorEngineError(f"Kylin vector bridge pipe failed: {exc}") from exc
            response = self._read_response(timeout=60)
            if not response.get("ok"):
                raise KylinVectorEngineError(_format_error(response))
            return response

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Some SDK builds log connection diagnostics to stdout.
                continue
            if isinstance(payload, dict):
                self._responses.put(payload)
        self._responses.put(None)

    def _read_response(self, *, timeout: float) -> dict[str, Any]:
        try:
            payload = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self._process.terminate()
            raise KylinVectorEngineError(
                f"Kylin vector bridge timed out after {timeout:.0f}s"
            ) from exc
        if payload is None:
            raise KylinVectorEngineError(
                f"Kylin vector bridge closed unexpectedly (exit={self._process.poll()})"
            )
        return payload

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.poll() is None and self._process.stdin is not None:
                try:
                    self._process.stdin.write('{"op":"close"}\n')
                    self._process.stdin.flush()
                    self._process.wait(timeout=5)
                except Exception:
                    self._process.terminate()
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait(timeout=5)
            self._reader.join(timeout=5)
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()

    def __enter__(self) -> KylinVectorEngineClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def collection_name(kind: str, identity: str, dimension: int) -> str:
    """Return a deterministic SDK-safe collection name."""

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    safe_kind = "".join(character if character.isalnum() else "_" for character in kind)
    return f"hkom_{safe_kind}_{dimension}_{digest}".lower()


def default_vector_store_backend() -> str:
    """Use the system service on deployment hosts; retain Windows development."""
    return os.environ.get("VECTOR_STORE_BACKEND", "sqlite" if os.name == "nt" else "kylin")


def system_vector_engine_available() -> bool:
    return (
        os.name != "nt"
        and shutil.which("kylin-ai-vector-engine") is not None
        and ctypes.util.find_library("kysdk-vector-engine-client") is not None
    )


def scalar_expression(field: str, value: str | bool | int) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, int):
        rendered = str(value)
    else:
        rendered = json.dumps(str(value), ensure_ascii=False)
    return f"{field} == {rendered}"


def _build_bridge() -> Path:
    if os.name == "nt":
        raise KylinVectorEngineError(
            "kylin-ai-vector-engine is only available on the Kylin target system"
        )
    compiler = shutil.which("g++")
    pkg_config = shutil.which("pkg-config")
    if not compiler or not pkg_config:
        raise KylinVectorEngineError("g++ and pkg-config are required for the Kylin bridge")
    root = Path(__file__).resolve().parents[3]
    source = root / "tools" / "kylin_vector_engine_bridge.cpp"
    if not source.is_file():
        raise KylinVectorEngineError(f"Kylin vector bridge source missing: {source}")
    try:
        flags = subprocess.run(
            [pkg_config, "--cflags", "--libs", "kysdk-vector-engine-client"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise KylinVectorEngineError(f"cannot resolve Kylin vector SDK flags: {exc}") from exc
    identity = hashlib.sha256(source.read_bytes() + "\0".join(flags).encode()).hexdigest()[:16]
    cache_root = Path(
        os.environ.get(
            "KYLIN_VECTOR_ENGINE_BRIDGE_CACHE",
            Path.home() / ".cache" / "hust-kylin-os-memory" / "vector-engine",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    executable = cache_root / f"kylin-vector-engine-bridge-{identity}"
    if executable.is_file():
        return executable
    temporary = executable.with_suffix(".tmp")
    try:
        subprocess.run(
            [compiler, "-std=c++17", "-O2", str(source), "-o", str(temporary), *flags],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        temporary.unlink(missing_ok=True)
        raise KylinVectorEngineError(f"cannot compile Kylin vector bridge: {exc}") from exc
    temporary.replace(executable)
    executable.chmod(0o755)
    return executable


def _format_error(response: dict[str, Any]) -> str:
    return (
        "Kylin vector-engine operation failed: "
        f"stage={response.get('stage')} code={response.get('status_code')} "
        f"message={response.get('message')}"
    )
