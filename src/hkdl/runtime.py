"""Variant environment and subprocess protocol."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .authoring import VariantRecord
from .config import ContractError
from .environments import EnvironmentFailure, EnvironmentHandle, EnvironmentStore
from .runs import validate_tracker
from .storage import RepositoryPaths, validate_repository_root


class RuntimeFailure(RuntimeError):
    """The Variant environment or worker failed."""


class RuntimeInterrupted(KeyboardInterrupt):
    """The Variant worker reported an interruption."""


class RuntimeOwnershipConflict(RuntimeError):
    """The Variant worker found an external ownership conflict."""


class VariantRuntime:
    def __init__(self, repository: RepositoryPaths | None = None):
        self._environment_store = (
            EnvironmentStore(repository) if repository is not None else None
        )
        self._lease_lock = threading.Lock()
        self._lease_descriptors: dict[Path, list[int]] = {}

    def acquire_environment(self, variant: VariantRecord) -> EnvironmentHandle:
        try:
            environment = self._store(variant).acquire(variant)
        except EnvironmentFailure as error:
            raise RuntimeFailure(str(error)) from error
        with self._lease_lock:
            self._lease_descriptors.setdefault(environment.python, []).append(
                environment.descriptor
            )

        def unregister(descriptor: int) -> None:
            with self._lease_lock:
                descriptors = self._lease_descriptors.get(environment.python, [])
                if descriptor in descriptors:
                    descriptors.remove(descriptor)
                if not descriptors:
                    self._lease_descriptors.pop(environment.python, None)

        environment.on_close = unregister
        return environment

    def prepare_environment(self, variant: VariantRecord) -> Path:
        with self.acquire_environment(variant) as environment:
            return environment.python

    def _store(self, variant: VariantRecord) -> EnvironmentStore:
        if self._environment_store is None:
            repository = validate_repository_root(variant.path.parents[2])
            self._environment_store = EnvironmentStore(repository)
        return self._environment_store

    def preflight(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        action: str = "train",
        cfg: dict[str, Any],
        selected: dict[str, str],
        seed: int,
        device: str,
        identity_fallback: dict[str, Any] | None = None,
        runtime_target: dict[str, Any] | None = None,
        environment_descriptor: int | None = None,
    ) -> dict[str, Any]:
        runtime_cfg = dict(cfg)
        runtime_cfg["runtime"] = {
            "action": action,
            "target": runtime_target or {},
        }
        result = self._invoke(
            python,
            variant,
            {
                "operation": "validate",
                "action": action,
                "source": str(variant.path / "src"),
                "cfg": runtime_cfg,
                "selected": selected,
                "exec": {"seed": seed, "device": device},
                "identity_fallback": identity_fallback or {},
                "repository_root": str(variant.path.parents[2]),
                "tracker_backends": list(
                    validate_tracker(runtime_cfg["variant"]["tracker"])
                ),
            },
            environment_descriptor=environment_descriptor,
        )
        if result.get("status") == "contract_error":
            raise ContractError(
                f"Variant {action} readiness failed: "
                f"{result.get('error_type', 'error')}"
            )
        return _successful(result)

    def train(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        cfg: dict[str, Any],
        selected: dict[str, str],
        exec_info: dict[str, Any],
        run_dir: Path,
        resume_from: Path | None = None,
        tracker_run_id: str | None = None,
        attempt_path: Path | None = None,
        lock_descriptor: int | None = None,
        runtime_target: dict[str, Any] | None = None,
        environment_descriptor: int | None = None,
    ) -> dict[str, Any]:
        runtime_cfg = dict(cfg)
        runtime_cfg["runtime"] = {
            "action": "train",
            "target": runtime_target or {},
        }
        result = self._invoke(
            python,
            variant,
            {
                "operation": "train",
                "action": "train",
                "source": str(variant.path / "src"),
                "cfg": runtime_cfg,
                "selected": selected,
                "exec": exec_info,
                "run_dir": str(run_dir),
                "resume_from": str(resume_from) if resume_from is not None else None,
                "tracker_run_id": tracker_run_id,
                "attempt_path": (
                    str(attempt_path) if attempt_path is not None else None
                ),
                "repository_root": str(variant.path.parents[2]),
                "tracker_backends": list(
                    validate_tracker(runtime_cfg["variant"]["tracker"])
                ),
            },
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
            log_path=run_dir / "worker.log",
        )
        return _successful(result)

    def evaluate(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        cfg: dict[str, Any],
        selected: dict[str, str],
        exec_info: dict[str, Any],
        run_dir: Path,
        checkpoint: Path,
        results_dir: Path | None = None,
        tracker_run_id: str | None = None,
        attempt_path: Path | None = None,
        lock_descriptor: int | None = None,
        runtime_target: dict[str, Any] | None = None,
        environment_descriptor: int | None = None,
    ) -> dict[str, Any]:
        runtime_cfg = dict(cfg)
        runtime_cfg["runtime"] = {
            "action": "eval",
            "target": runtime_target or {},
        }
        result = self._invoke(
            python,
            variant,
            {
                "operation": "evaluate",
                "action": "eval",
                "source": str(variant.path / "src"),
                "cfg": runtime_cfg,
                "selected": selected,
                "exec": exec_info,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "results_dir": str(
                    results_dir
                    if results_dir is not None
                    else run_dir / "artifacts/results"
                ),
                "tracker_run_id": tracker_run_id,
                "attempt_path": (
                    str(attempt_path) if attempt_path is not None else None
                ),
                "tracker_backends": list(
                    validate_tracker(runtime_cfg["variant"]["tracker"])
                ),
            },
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
            log_path=run_dir / "worker.log",
        )
        return _successful(result)

    def export(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        cfg: dict[str, Any],
        selected: dict[str, str],
        exec_info: dict[str, Any],
        run_dir: Path,
        export_dir: Path,
        checkpoint: Path,
        tracker_run_id: str | None = None,
        attempt_path: Path | None = None,
        lock_descriptor: int | None = None,
        runtime_target: dict[str, Any] | None = None,
        environment_descriptor: int | None = None,
    ) -> dict[str, Any]:
        runtime_cfg = dict(cfg)
        runtime_cfg["runtime"] = {
            "action": "export",
            "target": runtime_target or {},
        }
        result = self._invoke(
            python,
            variant,
            {
                "operation": "export",
                "action": "export",
                "source": str(variant.path / "src"),
                "cfg": runtime_cfg,
                "selected": selected,
                "exec": exec_info,
                "run_dir": str(run_dir),
                "export_dir": str(export_dir),
                "checkpoint": str(checkpoint),
                "tracker_run_id": tracker_run_id,
                "attempt_path": (
                    str(attempt_path) if attempt_path is not None else None
                ),
                "tracker_backends": list(
                    validate_tracker(runtime_cfg["variant"]["tracker"])
                ),
            },
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
            log_path=run_dir / "worker.log",
        )
        return _successful(result)

    def ensure_tracker(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        cfg: dict[str, Any],
        run_dir: Path,
        current_tracker_run_id: str | None,
        lock_descriptor: int,
        metadata: dict[str, Any],
        environment_descriptor: int | None = None,
    ) -> str | None:
        tracker_backends = validate_tracker(cfg["variant"]["tracker"])
        if "mlflow" not in tracker_backends:
            return None
        result = self._invoke(
            python,
            variant,
            {
                "operation": "tracker",
                "source": str(variant.path / "src"),
                "cfg": cfg,
                "run_dir": str(run_dir),
                "repository_root": str(variant.path.parents[2]),
                "current_tracker_run_id": current_tracker_run_id,
                "metadata": metadata,
                "tracker_backends": list(tracker_backends),
            },
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
        )
        successful = _successful(result)
        tracker_run_id = successful.get("tracker_run_id")
        if not isinstance(tracker_run_id, str):
            raise RuntimeFailure("Variant worker returned an invalid tracker identity")
        return tracker_run_id

    def log_tracker_metrics(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        tracker_run_id: str | None,
        values: dict[str, Any],
        lock_descriptor: int,
        environment_descriptor: int | None = None,
    ) -> None:
        if tracker_run_id is None:
            return
        self._successful_operation(
            python,
            variant,
            {
                "operation": "tracker_metrics",
                "tracker_run_id": tracker_run_id,
                "values": values,
                "repository_root": str(variant.path.parents[2]),
            },
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
        )

    def finish_tracker(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        tracker_run_id: str | None,
        status: str,
        lock_descriptor: int,
        environment_descriptor: int | None = None,
    ) -> None:
        if tracker_run_id is None:
            return
        self._successful_operation(
            python,
            variant,
            {
                "operation": "tracker_finish",
                "tracker_run_id": tracker_run_id,
                "run_status": status,
                "repository_root": str(variant.path.parents[2]),
            },
            lock_descriptor=lock_descriptor,
            environment_descriptor=environment_descriptor,
        )

    def _successful_operation(
        self,
        python: Path,
        variant: VariantRecord,
        request: dict[str, Any],
        *,
        lock_descriptor: int,
        environment_descriptor: int | None = None,
    ) -> dict[str, Any]:
        return _successful(
            self._invoke(
                python,
                variant,
                request,
                lock_descriptor=lock_descriptor,
                environment_descriptor=environment_descriptor,
            )
        )

    def _invoke(
        self,
        python: Path,
        variant: VariantRecord,
        request: dict[str, Any],
        *,
        lock_descriptor: int | None = None,
        environment_descriptor: int | None = None,
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        worker = Path(__file__).with_name("_runtime_worker.py")
        with tempfile.TemporaryDirectory(
            prefix=".hkdl-runtime-",
            dir=variant.path,
        ) as temporary_name:
            temporary = Path(temporary_name)
            request_path = temporary / "request.json"
            result_path = temporary / "result.json"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            command = [str(python), str(worker), str(request_path), str(result_path)]
            if environment_descriptor is None:
                with self._lease_lock:
                    descriptors = self._lease_descriptors.get(python, [])
                    environment_descriptor = descriptors[-1] if descriptors else None
            pass_fds = tuple(
                dict.fromkeys(
                    descriptor
                    for descriptor in (lock_descriptor, environment_descriptor)
                    if descriptor is not None
                )
            )
            if log_path is None:
                returncode = subprocess.run(
                    command,
                    cwd=variant.path / "src",
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    check=False,
                    pass_fds=pass_fds,
                ).returncode
            else:
                returncode = _run_logged_worker(
                    command,
                    cwd=variant.path / "src",
                    pass_fds=pass_fds,
                    log_path=log_path,
                )
            if returncode != 0 or not result_path.is_file():
                raise RuntimeFailure("Variant worker failed without a result")
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeFailure("Variant worker returned invalid JSON") from error
            if not isinstance(result, dict):
                raise RuntimeFailure("Variant worker result must be a mapping")
            return result


def _run_logged_worker(
    command: list[str],
    *,
    cwd: Path,
    pass_fds: tuple[int, ...],
    log_path: Path,
) -> int:
    with log_path.open("xb", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=pass_fds,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                ready = selector.select(timeout=0.1)
                if ready:
                    chunk = os.read(process.stdout.fileno(), 64 * 1024)
                    if not chunk:
                        break
                    log.write(chunk)
                    _write_stderr(chunk)
                elif process.poll() is not None:
                    break
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            selector.close()
            process.stdout.close()


def _write_stderr(chunk: bytes) -> None:
    buffer = getattr(sys.stderr, "buffer", None)
    if buffer is not None:
        buffer.write(chunk)
        buffer.flush()
        return
    sys.stderr.write(
        chunk.decode(getattr(sys.stderr, "encoding", None) or "utf-8", errors="replace")
    )
    sys.stderr.flush()


def _successful(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status")
    if status == "interrupted":
        raise RuntimeInterrupted
    if status == "ownership_conflict":
        raise RuntimeOwnershipConflict(
            f"Variant worker ownership conflict: {result.get('error_type', 'error')}"
        )
    if status != "ok":
        raise RuntimeFailure(
            f"Variant worker failed: {result.get('error_type', 'error')}"
        )
    return result
