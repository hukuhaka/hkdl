"""Variant environment and subprocess protocol."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .authoring import VariantRecord
from .config import ContractError
from .storage import directory_lock


class RuntimeFailure(RuntimeError):
    """The Variant environment or worker failed."""


class RuntimeInterrupted(KeyboardInterrupt):
    """The Variant worker reported an interruption."""


class RuntimeOwnershipConflict(RuntimeError):
    """The Variant worker found an external ownership conflict."""


class VariantRuntime:
    def prepare_environment(self, variant: VariantRecord) -> Path:
        environment = variant.path / ".venv"
        if os.path.lexists(environment):
            _require_real_directory(environment)
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeFailure("uv is unavailable")
        variables = os.environ.copy()
        variables["UV_PROJECT_ENVIRONMENT"] = str(environment)
        command = [
            uv,
            "sync",
            "--project",
            str(variant.path / "src"),
            "--locked",
            "--no-dev",
            "--no-install-project",
        ]
        tracker = variant.document.get("tracker")
        if tracker == {"backend": "mlflow"}:
            command.extend(["--extra", "mlflow"])
        with directory_lock(variant.path) as lock_descriptor:
            result = subprocess.run(
                command,
                env=variables,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                pass_fds=(lock_descriptor,),
            )
        if result.returncode != 0:
            raise RuntimeFailure("Variant environment synchronization failed")
        _require_real_directory(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not python.exists():
            raise RuntimeFailure("Variant Python is unavailable")
        return python

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
            },
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
            },
            lock_descriptor=lock_descriptor,
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
            },
            lock_descriptor=lock_descriptor,
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
            },
            lock_descriptor=lock_descriptor,
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
    ) -> str | None:
        if cfg["variant"]["tracker"] == {"backend": "none"}:
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
            },
            lock_descriptor=lock_descriptor,
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
        )

    def finish_tracker(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        tracker_run_id: str | None,
        status: str,
        lock_descriptor: int,
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
        )

    def _successful_operation(
        self,
        python: Path,
        variant: VariantRecord,
        request: dict[str, Any],
        *,
        lock_descriptor: int,
    ) -> dict[str, Any]:
        return _successful(
            self._invoke(
                python,
                variant,
                request,
                lock_descriptor=lock_descriptor,
            )
        )

    @staticmethod
    def _invoke(
        python: Path,
        variant: VariantRecord,
        request: dict[str, Any],
        *,
        lock_descriptor: int | None = None,
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
            process = subprocess.run(
                [str(python), str(worker), str(request_path), str(result_path)],
                cwd=variant.path / "src",
                stdout=sys.stderr,
                stderr=sys.stderr,
                check=False,
                pass_fds=((lock_descriptor,) if lock_descriptor is not None else ()),
            )
            if process.returncode != 0 or not result_path.is_file():
                raise RuntimeFailure("Variant worker failed without a result")
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeFailure("Variant worker returned invalid JSON") from error
            if not isinstance(result, dict):
                raise RuntimeFailure("Variant worker result must be a mapping")
            return result


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


def _require_real_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RuntimeFailure(f"Variant environment is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise RuntimeFailure(f"Variant environment is invalid: {path}")
