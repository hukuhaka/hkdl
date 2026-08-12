"""Content-addressed Variant environments and explicit cache pruning."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable

from .authoring import VariantRecord
from .config import ContractError
from .runs import validate_tracker
from .storage import RepositoryPaths, publish_directory

ENVIRONMENT_SCHEMA_VERSION = 1
KEY_LENGTH = 64


class EnvironmentFailure(RuntimeError):
    """Environment discovery, build, validation, or deletion failed."""


@dataclass(frozen=True)
class EnvironmentIdentity:
    key: str
    document: dict[str, Any]
    uv: Path
    python: Path
    extras: tuple[str, ...]


@dataclass
class EnvironmentHandle:
    key: str
    python: Path
    descriptor: int
    on_close: Callable[[int], None] | None = None

    def close(self) -> None:
        if self.descriptor < 0:
            return
        descriptor = self.descriptor
        self.descriptor = -1
        if self.on_close is not None:
            self.on_close(descriptor)
        _unlock(descriptor)

    def __enter__(self) -> EnvironmentHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class PruneEntry:
    kind: str
    path: Path
    key: str | None
    bytes: int


@dataclass(frozen=True)
class PrunePlan:
    entries: tuple[PruneEntry, ...]
    retained: int
    busy: int

    @property
    def bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)


@dataclass(frozen=True)
class PruneResult:
    removed: int
    bytes: int
    retained: int
    busy: int


class EnvironmentStore:
    def __init__(self, repository: RepositoryPaths):
        self.repository = repository
        self.root = repository.root / ".hkdl/environments"
        self.store = self.root / f"v{ENVIRONMENT_SCHEMA_VERSION}"
        self.locks = self.root / "locks"
        self._uv_runtime: tuple[Path, str] | None = None
        self._interpreters: dict[Path, dict[str, str]] = {}

    def identity(self, variant: VariantRecord) -> EnvironmentIdentity:
        source = variant.path / "src"
        project = _read_regular(source / "pyproject.toml")
        lock = _read_regular(source / "uv.lock")
        extras = (
            ("mlflow",)
            if "mlflow" in validate_tracker(variant.document.get("tracker"))
            else ()
        )
        uv, uv_version, python, interpreter = self._runtime_identity(source)
        document: dict[str, Any] = {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "project_digest": _digest(project),
            "lock_digest": _digest(lock),
            "extras": list(extras),
            "interpreter": interpreter,
            "uv_version": uv_version,
        }
        encoded = json.dumps(
            document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        return EnvironmentIdentity(key, document, uv, python, extras)

    def acquire(self, variant: VariantRecord) -> EnvironmentHandle:
        identity = self.identity(variant)
        self._ensure_layout()
        descriptor = _lock_file(self.locks / f"{identity.key}.lock", shared=False)
        candidate: Path | None = None
        try:
            target = self.store / identity.key
            if not os.path.lexists(target):
                candidate = Path(
                    tempfile.mkdtemp(
                        prefix=f".{identity.key}.candidate-",
                        dir=self.store,
                    )
                )
                self._synchronize(candidate, variant, identity)
                _write_manifest(candidate / "environment.json", identity)
                self._validate(candidate, identity)
                try:
                    publish_directory(candidate, target)
                    candidate = None
                except Exception:
                    if os.path.lexists(target):
                        self._validate(target, identity)
                    else:
                        raise
            self._validate(target, identity)
            self._check(target, variant, identity)
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            return EnvironmentHandle(
                identity.key,
                _environment_python(target),
                descriptor,
            )
        except BaseException:
            _unlock(descriptor)
            raise
        finally:
            if candidate is not None:
                shutil.rmtree(candidate, ignore_errors=True)

    def prepare(self, variant: VariantRecord) -> Path:
        with self.acquire(variant) as environment:
            return environment.python

    def plan_prune(
        self,
        variants: Iterable[VariantRecord],
        *,
        active_variants: set[tuple[str, str]],
        remove_all: bool,
    ) -> PrunePlan:
        records = tuple(variants)
        store_entries = self._store_entries()
        referenced = (
            {self.identity(variant).key for variant in records}
            if not remove_all
            and any(kind == "environment" for _, _, kind in store_entries)
            else set()
        )
        entries: list[PruneEntry] = []
        retained = 0
        busy = 0

        for variant in records:
            legacy = variant.path / ".venv"
            if not os.path.lexists(legacy):
                continue
            _require_real_directory(legacy, "legacy Variant environment")
            if (variant.experiment, str(variant.document["name"])) in active_variants:
                busy += 1
                continue
            entries.append(PruneEntry("legacy", legacy, None, _logical_bytes(legacy)))

        for path, key, kind in store_entries:
            if kind == "environment" and not remove_all and key in referenced:
                retained += 1
                continue
            if self._is_busy(key):
                busy += 1
                continue
            entries.append(PruneEntry(kind, path, key, _logical_bytes(path)))

        entries.sort(key=lambda entry: os.fsencode(str(entry.path)))
        return PrunePlan(tuple(entries), retained, busy)

    def prune(
        self,
        variants: Iterable[VariantRecord],
        *,
        active_variants: set[tuple[str, str]],
        remove_all: bool,
    ) -> PruneResult:
        plan = self.plan_prune(
            variants,
            active_variants=active_variants,
            remove_all=remove_all,
        )
        removed = 0
        removed_bytes = 0
        busy = plan.busy
        for entry in plan.entries:
            if entry.kind == "legacy":
                try:
                    descriptor = _lock_directory(entry.path.parent, blocking=False)
                except BlockingIOError:
                    busy += 1
                    continue
            else:
                assert entry.key is not None
                self._ensure_layout()
                try:
                    descriptor = _lock_file(
                        self.locks / f"{entry.key}.lock",
                        shared=False,
                        blocking=False,
                    )
                except BlockingIOError:
                    busy += 1
                    continue
            try:
                if not os.path.lexists(entry.path):
                    continue
                _require_real_directory(entry.path, "environment prune target")
                self._ensure_layout()
                trash_key = (
                    entry.key
                    or hashlib.sha256(
                        entry.path.relative_to(self.repository.root)
                        .as_posix()
                        .encode("utf-8")
                    ).hexdigest()
                )
                if entry.kind == "trash":
                    detached = entry.path
                else:
                    detached = (
                        self.store / f".{trash_key}.trash-{os.getpid()}-{removed}"
                    )
                    if os.path.lexists(detached):
                        raise EnvironmentFailure(
                            f"prune trash already exists: {detached}"
                        )
                    os.rename(entry.path, detached)
                try:
                    shutil.rmtree(detached)
                except OSError as error:
                    raise EnvironmentFailure(
                        f"could not remove detached environment {detached}: {error}"
                    ) from error
                removed += 1
                removed_bytes += entry.bytes
            except EnvironmentFailure:
                raise
            except OSError as error:
                raise EnvironmentFailure(
                    f"could not prune environment {entry.path}: {error}"
                ) from error
            finally:
                _unlock(descriptor)
        return PruneResult(removed, removed_bytes, plan.retained, busy)

    def _runtime_identity(
        self,
        source: Path,
    ) -> tuple[Path, str, Path, dict[str, str]]:
        if self._uv_runtime is None:
            uv_name = shutil.which("uv")
            if uv_name is None:
                raise EnvironmentFailure("uv is unavailable")
            uv = Path(uv_name).absolute()
            uv_version = _run_text([str(uv), "--version"], "uv version discovery")
            self._uv_runtime = (uv, uv_version)
        uv, uv_version = self._uv_runtime
        python = Path(
            _run_text(
                [str(uv), "python", "find", "--project", str(source)],
                "Variant Python discovery",
            )
        ).absolute()
        if python in self._interpreters:
            return uv, uv_version, python, self._interpreters[python]
        probe = (
            "import json,platform,sys,sysconfig;"
            "print(json.dumps({"
            "'implementation':platform.python_implementation(),"
            "'version':platform.python_version(),"
            "'cache_tag':sys.implementation.cache_tag,"
            "'abi':sysconfig.get_config_var('SOABI') or '',"
            "'system':platform.system(),"
            "'machine':platform.machine()"
            "},sort_keys=True,separators=(',',':')))"
        )
        raw = _run_text([str(python), "-c", probe], "Variant Python inspection")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EnvironmentFailure(
                "Variant Python returned invalid identity"
            ) from error
        if not isinstance(document, dict) or any(
            not isinstance(document.get(name), str)
            for name in (
                "implementation",
                "version",
                "cache_tag",
                "abi",
                "system",
                "machine",
            )
        ):
            raise EnvironmentFailure("Variant Python returned invalid identity")
        interpreter = {name: str(document[name]) for name in sorted(document)}
        self._interpreters[python] = interpreter
        return uv, uv_version, python, interpreter

    def _synchronize(
        self,
        environment: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> None:
        command = self._sync_command(environment, variant, identity)
        result = subprocess.run(
            command,
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise EnvironmentFailure("Variant environment synchronization failed")

    def _check(
        self,
        environment: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> None:
        command = [*self._sync_command(environment, variant, identity), "--check"]
        result = subprocess.run(
            command,
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise EnvironmentFailure("shared Variant environment is inconsistent")

    @staticmethod
    def _sync_command(
        environment: Path,
        variant: VariantRecord,
        identity: EnvironmentIdentity,
    ) -> list[str]:
        command = [
            str(identity.uv),
            "sync",
            "--project",
            str(variant.path / "src"),
            "--python",
            str(identity.python),
            "--locked",
            "--no-dev",
            "--no-install-project",
        ]
        for extra in identity.extras:
            command.extend(["--extra", extra])
        return command

    def _ensure_layout(self) -> None:
        current = self.repository.root
        for part in (".hkdl", "environments", f"v{ENVIRONMENT_SCHEMA_VERSION}"):
            current /= part
            _ensure_real_directory(current)
        _ensure_real_directory(self.locks)

    def _validate(self, path: Path, identity: EnvironmentIdentity) -> None:
        _require_real_directory(path, "shared Variant environment")
        manifest = path / "environment.json"
        try:
            mode = manifest.lstat().st_mode
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EnvironmentFailure(
                f"invalid environment manifest: {manifest}"
            ) from error
        if manifest.is_symlink() or not stat.S_ISREG(mode):
            raise EnvironmentFailure(f"invalid environment manifest: {manifest}")
        expected = {"key": identity.key, "identity": identity.document}
        if document != expected:
            raise EnvironmentFailure(f"environment identity mismatch: {path}")
        python = _environment_python(path)
        if (
            not python.exists()
            or not python.is_file()
            or not os.access(python, os.X_OK)
        ):
            raise EnvironmentFailure(f"shared Variant Python is unavailable: {python}")

    def _store_entries(self) -> list[tuple[Path, str, str]]:
        if not os.path.lexists(self.store):
            return []
        _require_real_directory(self.store, "shared environment store")
        values: list[tuple[Path, str, str]] = []
        with os.scandir(self.store) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise ContractError(f"invalid shared environment entry: {path}")
                if len(entry.name) == KEY_LENGTH and all(
                    character in "0123456789abcdef" for character in entry.name
                ):
                    values.append((path, entry.name, "environment"))
                    continue
                if entry.name.startswith(".") and ".candidate-" in entry.name:
                    key = entry.name[1 : KEY_LENGTH + 1]
                    if len(key) == KEY_LENGTH and all(
                        character in "0123456789abcdef" for character in key
                    ):
                        values.append((path, key, "candidate"))
                        continue
                if entry.name.startswith(".") and ".trash-" in entry.name:
                    key = entry.name[1 : KEY_LENGTH + 1]
                    if len(key) == KEY_LENGTH and all(
                        character in "0123456789abcdef" for character in key
                    ):
                        values.append((path, key, "trash"))
                        continue
                raise ContractError(f"invalid shared environment entry: {path}")
        return sorted(values, key=lambda item: os.fsencode(item[0].name))

    def _is_busy(self, key: str) -> bool:
        lock = self.locks / f"{key}.lock"
        if not os.path.lexists(lock):
            return False
        try:
            descriptor = _lock_file(lock, shared=False, blocking=False, create=False)
        except BlockingIOError:
            return True
        _unlock(descriptor)
        return False


def _write_manifest(path: Path, identity: EnvironmentIdentity) -> None:
    payload = (
        json.dumps(
            {"key": identity.key, "identity": identity.document},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as error:
        raise EnvironmentFailure(
            f"could not write environment manifest: {path}"
        ) from error


def _read_regular(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
        payload = path.read_bytes()
    except OSError as error:
        raise EnvironmentFailure(f"environment input is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise EnvironmentFailure(f"environment input is invalid: {path}")
    return payload


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _run_text(command: list[str], purpose: str) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise EnvironmentFailure(f"{purpose} failed") from error
    if result.returncode != 0 or not result.stdout.strip():
        raise EnvironmentFailure(f"{purpose} failed")
    return result.stdout.strip()


def _environment_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _ensure_real_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        pass
    except OSError as error:
        raise EnvironmentFailure(
            f"could not create environment directory: {path}"
        ) from error
    _require_real_directory(path, "environment directory")


def _require_real_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise EnvironmentFailure(f"{label} is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise EnvironmentFailure(f"{label} is invalid: {path}")


def _lock_file(
    path: Path,
    *,
    shared: bool,
    blocking: bool = True,
    create: bool = True,
) -> int:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise EnvironmentFailure(f"environment lock is unavailable: {path}") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise EnvironmentFailure(f"environment lock is invalid: {path}")
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, operation)
    except BlockingIOError:
        os.close(descriptor)
        raise
    return descriptor


def _lock_directory(path: Path, *, blocking: bool) -> int:
    _require_real_directory(path, "Variant directory")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(descriptor, operation)
    except BlockingIOError:
        os.close(descriptor)
        raise
    return descriptor


def _unlock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _logical_bytes(root: Path) -> int:
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise ContractError(
            f"cannot inspect environment tree {root}: {error}"
        ) from error
    if not stat.S_ISDIR(root_stat.st_mode):
        return root_stat.st_size
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ContractError(
                f"cannot scan environment tree {directory}: {error}"
            ) from error
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ContractError(
                    f"cannot inspect environment entry {entry.path}: {error}"
                ) from error
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(Path(entry.path))
            else:
                total += metadata.st_size
    return total


__all__ = [
    "EnvironmentFailure",
    "EnvironmentHandle",
    "EnvironmentIdentity",
    "EnvironmentStore",
    "PruneEntry",
    "PrunePlan",
    "PruneResult",
]
