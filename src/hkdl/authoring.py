"""Template, Experiment, and Variant authoring operations."""

from __future__ import annotations

import copy
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import (
    ContractError,
    NAME_PATTERN,
    dump_yaml,
    load_yaml_file,
    validate_experiment,
    validate_variant,
)
from .storage import (
    AlreadyExistsError,
    NotFoundError,
    OwnershipError,
    RepositoryPaths,
    ResolvedTemplate,
    TemplateResolver,
    atomic_write_new,
    compute_bundle_digest,
    compute_source_digest,
    publish_directory,
    validate_locked_source_tree,
)

RESERVED_VARIANT_NAMES = frozenset({"notes", "src", "variants"})


@dataclass(frozen=True)
class ExperimentRecord:
    path: Path
    document: dict[str, object]


@dataclass(frozen=True)
class VariantRecord:
    path: Path
    experiment: str
    document: dict[str, object]


class Authoring:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
    ):
        self.repository = repository
        self.templates = TemplateResolver(repository)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def list_templates(self) -> list[ResolvedTemplate]:
        return self.templates.list()

    def show_template(self, reference: str) -> ResolvedTemplate:
        return self.templates.resolve(reference)

    def create_experiment(self, name: str, template_name: str) -> ExperimentRecord:
        _name(name, "Experiment")
        _name(template_name, "Template")
        experiments = self._ensure_experiment_catalog()
        target = experiments / name
        if os.path.lexists(target):
            raise AlreadyExistsError(f"Experiment already exists: {name}")
        template = self.templates.latest(template_name)

        candidate = Path(
            tempfile.mkdtemp(prefix=f".{name}.candidate-", dir=experiments)
        )
        try:
            (candidate / "notes").mkdir()

            document = {
                "schema_version": template.experiment_seed["schema_version"],
                "name": name,
                "type": template.experiment_seed["type"],
                "question": template.experiment_seed["question"],
                "template": {"name": template.manifest["name"]},
                "created_at": _timestamp(self._now()),
            }
            atomic_write_new(candidate / "experiment.yaml", dump_yaml(document))

            record = _load_experiment(candidate, expected_name=name)
            if record.document["type"] != template.manifest["type"]:
                raise ContractError("copied Experiment type changed")
            if record.document["template"] != {"name": template.manifest["name"]}:
                raise ContractError("copied Experiment provenance changed")

            publish_directory(candidate, target)
            return ExperimentRecord(target, record.document)
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    def list_experiments(self) -> list[ExperimentRecord]:
        if not os.path.lexists(self.repository.experiments):
            return []
        records = [
            self.load_experiment(entry.name)
            for entry in _catalog_entries(self.repository.experiments)
        ]
        return sorted(
            records,
            key=lambda record: record.document["name"].encode("utf-8"),
        )

    def load_experiment(self, name: str) -> ExperimentRecord:
        _name(name, "Experiment")
        if os.path.lexists(self.repository.experiments):
            _require_directory(self.repository.experiments)
        path = self.repository.experiments / name
        if not os.path.lexists(path):
            raise NotFoundError(f"Experiment not found: {name}")
        return _load_experiment(path, expected_name=name)

    def create_variant(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        template_version: str | None = None,
    ) -> VariantRecord:
        _variant_name(variant_name)
        experiment = self.load_experiment(experiment_name)
        target = experiment.path / variant_name
        if os.path.lexists(target):
            raise AlreadyExistsError(
                f"Variant already exists: {experiment_name}/{variant_name}"
            )

        template_name = str(experiment.document["template"]["name"])
        if template_version is None:
            template = self.templates.latest(template_name)
        else:
            template = self.templates.resolve(f"{template_name}@{template_version}")
        if template.manifest["type"] != experiment.document["type"]:
            raise ContractError("Template type does not match Experiment type")

        candidate = Path(
            tempfile.mkdtemp(prefix=f".{variant_name}.candidate-", dir=experiment.path)
        )
        try:
            shutil.copytree(template.path / "src", candidate / "src")
            seed = copy.deepcopy(template.variant_seed)
            document = {
                "schema_version": seed.pop("schema_version"),
                "name": variant_name,
                "template": {
                    "name": template.manifest["name"],
                    "version": template.manifest["version"],
                    "digest": template.bundle_digest,
                },
                **seed,
            }
            atomic_write_new(candidate / "variant.yaml", dump_yaml(document))
            record = _load_variant_directory(
                candidate,
                experiment_name=experiment_name,
                expected_name=variant_name,
            )
            if compute_source_digest(candidate / "src") != compute_source_digest(
                template.path / "src"
            ):
                raise ContractError("copied Variant source digest changed")
            if compute_bundle_digest(template.path) != template.bundle_digest:
                raise ContractError("Template Bundle changed during Variant creation")
            publish_directory(candidate, target)
            return VariantRecord(target, experiment_name, record.document)
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    def clone_variant(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        source_variant: str,
        source_experiment: str | None = None,
    ) -> VariantRecord:
        _variant_name(variant_name)
        _variant_name(source_variant)
        target_experiment = self.load_experiment(experiment_name)
        source_experiment = source_experiment or experiment_name
        source = self.load_variant(source_experiment, source_variant)
        source_owner = self.load_experiment(source_experiment)
        if source_owner.document["type"] != target_experiment.document["type"]:
            raise ContractError("source and target Experiment types do not match")
        if (
            source_owner.document["template"]["name"]
            != target_experiment.document["template"]["name"]
        ):
            raise ContractError(
                "source and target Experiment Template families do not match"
            )

        target = target_experiment.path / variant_name
        if os.path.lexists(target):
            raise AlreadyExistsError(
                f"Variant already exists: {experiment_name}/{variant_name}"
            )
        candidate = Path(
            tempfile.mkdtemp(
                prefix=f".{variant_name}.candidate-", dir=target_experiment.path
            )
        )
        try:
            source_digest = compute_source_digest(source.path / "src")
            shutil.copytree(source.path / "src", candidate / "src")
            document = copy.deepcopy(source.document)
            document["name"] = variant_name
            validate_variant(document, expected_name=variant_name)
            atomic_write_new(candidate / "variant.yaml", dump_yaml(document))
            record = _load_variant_directory(
                candidate,
                experiment_name=experiment_name,
                expected_name=variant_name,
            )
            if (
                compute_source_digest(candidate / "src") != source_digest
                or compute_source_digest(source.path / "src") != source_digest
            ):
                raise ContractError("source Variant changed during clone")
            publish_directory(candidate, target)
            return VariantRecord(target, experiment_name, record.document)
        finally:
            if candidate.exists():
                shutil.rmtree(candidate)

    def list_variants(self, experiment_name: str) -> list[VariantRecord]:
        experiment = self.load_experiment(experiment_name)
        records: list[VariantRecord] = []
        for entry in _variant_catalog_entries(experiment.path):
            _variant_name(entry.name)
            records.append(self.load_variant(experiment_name, entry.name))
        return sorted(
            records,
            key=lambda record: record.document["name"].encode("utf-8"),
        )

    def check_variant(self, experiment_name: str, variant_name: str) -> VariantRecord:
        return self.load_variant(experiment_name, variant_name)

    def load_variant(self, experiment_name: str, variant_name: str) -> VariantRecord:
        _name(experiment_name, "Experiment")
        _variant_name(variant_name)
        experiment = self.load_experiment(experiment_name)
        path = experiment.path / variant_name
        if not os.path.lexists(path):
            raise NotFoundError(f"Variant not found: {experiment_name}/{variant_name}")
        record = _load_variant_directory(
            path,
            experiment_name=experiment_name,
            expected_name=variant_name,
        )
        if (
            record.document["template"]["name"]
            != experiment.document["template"]["name"]
        ):
            raise OwnershipError(
                f"Variant Template family does not match its Experiment: "
                f"{experiment_name}/{variant_name}"
            )
        return record

    def _ensure_experiment_catalog(self) -> Path:
        path = self.repository.experiments
        try:
            path.mkdir(mode=0o755)
        except FileExistsError:
            pass
        _require_directory(path)
        return path


def _load_experiment(path: Path, *, expected_name: str) -> ExperimentRecord:
    _require_directory(path)
    if any(os.path.lexists(path / name) for name in ("src", "variants", ".venv")):
        raise ContractError(
            "legacy Experiment layout is unsupported; migration is deferred"
        )
    _require_directory(path / "notes")
    document = load_yaml_file(path / "experiment.yaml")
    validate_experiment(document)
    if document["name"] != expected_name:
        raise OwnershipError(
            f"Experiment name does not match its directory: {expected_name}"
        )
    return ExperimentRecord(path, document)


def _load_variant_directory(
    path: Path,
    *,
    experiment_name: str,
    expected_name: str,
) -> VariantRecord:
    _require_directory(path)
    validate_locked_source_tree(path / "src")
    document = load_yaml_file(path / "variant.yaml")
    validate_variant(document)
    if document["name"] != expected_name:
        raise OwnershipError(
            f"Variant name does not match its path: {experiment_name}/{expected_name}"
        )
    return VariantRecord(path, experiment_name, document)


def _catalog_entries(path: Path) -> list[Path]:
    _require_directory(path)
    entries: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise ContractError(f"symlinked catalog entry: {entry.path}")
        entries.append(Path(entry.path))
    return sorted(entries, key=lambda entry: entry.name.encode("utf-8"))


def _variant_catalog_entries(path: Path) -> list[Path]:
    entries: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith(".") or entry.name in {"experiment.yaml", "notes"}:
            continue
        if entry.name in RESERVED_VARIANT_NAMES:
            raise ContractError(
                "legacy Experiment layout is unsupported; migration is deferred"
            )
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ContractError(f"invalid Variant catalog entry: {entry.name}")
        entries.append(Path(entry.path))
    return sorted(entries, key=lambda entry: entry.name.encode("utf-8"))


def _require_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required directory is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ContractError(f"required directory is invalid: {path}")


def _name(value: object, kind: str) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ContractError(f"{kind} has an invalid name")


def _variant_name(value: object) -> None:
    _name(value, "Variant")
    if value in RESERVED_VARIANT_NAMES:
        raise ContractError("Variant uses a reserved name")


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContractError("creation clock must include a timezone")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
