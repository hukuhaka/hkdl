"""Variant-managed action Runs and immutable Model records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .attempts import TRACKER_ID_PATTERN
from .authoring import ExperimentRecord, VariantRecord
from .config import (
    DIGEST_PATTERN,
    NAME_PATTERN,
    ContractError,
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
    atomic_replace,
    atomic_write_new,
    directory_lock,
)

RUN_ID_PATTERN = re.compile(r"run-[0-9]+")
MODEL_ID_PATTERN = re.compile(r"model-[0-9a-f]{32}")
TRAIN_COMPONENTS = frozenset({"model", "loss", "optimizer", "dataloader", "trainer"})
EVAL_COMPONENTS = frozenset({"model", "dataloader", "evaluator"})
EXPORT_COMPONENTS = frozenset({"exporter"})
ACTIONS = frozenset({"train", "eval", "export"})
RUN_STATUSES = frozenset(
    {"allocated", "running", "done", "failed", "interrupted", "abandoned"}
)
TERMINAL_STATUSES = frozenset({"done", "failed", "interrupted", "abandoned"})
MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class RunRecord:
    path: Path
    address: str
    snapshot: dict[str, Any]
    request: dict[str, Any]
    state: dict[str, Any]


@dataclass(frozen=True)
class ModelRecord:
    path: Path
    address: str
    document: dict[str, Any]


def validate_tracker(value: Any) -> tuple[str, ...]:
    tracker = _mapping(value, "variant.tracker")
    if set(tracker) != {"backend"}:
        raise ContractError("tracker must contain only backend")
    backend = tracker["backend"]
    if isinstance(backend, str):
        if backend == "none":
            return ()
        if backend in {"local", "mlflow"}:
            return (backend,)
    elif isinstance(backend, (list, tuple)):
        if (
            backend
            and all(
                isinstance(item, str) and item in {"local", "mlflow"}
                for item in backend
            )
            and len(backend) == len(set(backend))
        ):
            return tuple(item for item in ("local", "mlflow") if item in backend)
    raise ContractError(
        "tracker.backend must be none, local, mlflow, or a non-empty unique list "
        "of local and mlflow"
    )


def validate_training_readiness(
    experiment: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, str]:
    del experiment
    validate_tracker(variant["tracker"])
    return _required_components(variant, TRAIN_COMPONENTS, "training")


def validate_evaluation_readiness(
    experiment: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    case: str = "default",
) -> dict[str, str]:
    del experiment
    validate_tracker(variant["tracker"])
    evaluation_case(variant, case)
    metric_spec(variant, case)
    return _required_components(variant, EVAL_COMPONENTS, "evaluation")


def validate_export_readiness(
    experiment: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, str]:
    del experiment
    validate_tracker(variant["tracker"])
    return _required_components(variant, EXPORT_COMPONENTS, "export")


def evaluation_case(variant: Mapping[str, Any], case: str) -> dict[str, Any]:
    _identity(case, "Evaluation Case")
    evaluation = _mapping(variant["eval"], "variant.eval")
    cases = evaluation.get("cases")
    if cases is None:
        if case != "default":
            raise NotFoundError(f"Evaluation Case not found: {case}")
        return deepcopy(dict(evaluation))
    cases = _mapping(cases, "variant.eval.cases")
    if case not in cases:
        raise NotFoundError(f"Evaluation Case not found: {case}")
    return deepcopy(dict(_mapping(cases[case], f"variant.eval.cases.{case}")))


def metric_spec(variant: Mapping[str, Any], case: str) -> dict[str, Any]:
    selected_case = evaluation_case(variant, case)
    value = selected_case.get("metrics", variant["metrics"])
    metrics = _mapping(value, f"evaluation case {case} metrics")
    primary = metrics.get("primary")
    report = metrics.get("report")
    if not isinstance(primary, str) or not NAME_PATTERN.fullmatch(primary):
        raise ContractError("metrics.primary is invalid")
    if (
        not isinstance(report, (list, tuple))
        or not report
        or any(
            not isinstance(name, str) or not NAME_PATTERN.fullmatch(name)
            for name in report
        )
        or len(report) != len(set(report))
        or list(report).count(primary) != 1
    ):
        raise ContractError(
            "metrics.report must be unique and contain metrics.primary once"
        )
    return {"primary": primary, "report": list(report)}


def validate_snapshot(document: dict[str, Any]) -> None:
    _exact_fields(document, {"schema_version", "experiment", "variant", "provenance"})
    _integer_one(document["schema_version"], "snapshot.schema_version")
    experiment = _mapping(document["experiment"], "snapshot.experiment")
    variant = _mapping(document["variant"], "snapshot.variant")
    validate_experiment(experiment)
    validate_variant(variant)
    provenance = _mapping(document["provenance"], "snapshot.provenance")
    if set(provenance) != {
        "variant_file",
        "source_digest",
        "vcs_revision",
        "frozen_at",
    }:
        raise ContractError("snapshot.provenance fields are invalid")
    _owned_path(provenance["variant_file"], "snapshot.provenance.variant_file")
    _digest(provenance["source_digest"], "snapshot.provenance.source_digest")
    if provenance["vcs_revision"] is not None:
        raise ContractError("snapshot.provenance.vcs_revision must be null")
    _timestamp(provenance["frozen_at"], "snapshot.provenance.frozen_at")


def validate_request(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_id",
        "experiment",
        "variant",
        "action",
        "target",
        "retry_of",
        "source_digest",
        "identity_fingerprint",
        "exec",
        "created_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "request.schema_version")
    _run_id(document["run_id"])
    _identity(document["experiment"], "request Experiment")
    _identity(document["variant"], "request Variant")
    action = document["action"]
    if action not in ACTIONS:
        raise ContractError("request.action is invalid")
    target = _mapping(document["target"], "request.target")
    if action == "train":
        _exact_fields(target, {"training_group", "seed"})
        _identity(target["training_group"], "Training Group")
        _seed(target["seed"])
    elif action == "eval":
        _exact_fields(
            target,
            {"training_group", "seed", "model_id", "evaluation_case"},
        )
        _identity(target["training_group"], "Training Group")
        _seed(target["seed"])
        _model_id(target["model_id"])
        _identity(target["evaluation_case"], "Evaluation Case")
    else:
        _exact_fields(target, {"model_id"})
        _model_id(target["model_id"])
    retry_of = document["retry_of"]
    if retry_of is not None:
        _run_id(retry_of)
        if retry_of == document["run_id"]:
            raise ContractError("request.retry_of cannot reference itself")
    _digest(document["source_digest"], "request.source_digest")
    _digest(document["identity_fingerprint"], "request.identity_fingerprint")
    exec_info = _mapping(document["exec"], "request.exec")
    _exact_fields(exec_info, {"seed", "device"})
    _seed(exec_info["seed"])
    if (
        not isinstance(exec_info["device"], str)
        or not exec_info["device"]
        or exec_info["device"] == "auto"
    ):
        raise ContractError("request.exec.device must be concrete")
    _timestamp(document["created_at"], "request.created_at")
    _json_compatible(document)


def validate_state(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_id",
        "action",
        "status",
        "reason",
        "tracker_run_id",
        "result",
        "best_checkpoint",
        "last_checkpoint",
        "created_at",
        "updated_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "state.schema_version")
    _run_id(document["run_id"])
    if document["action"] not in ACTIONS:
        raise ContractError("state.action is invalid")
    status = document["status"]
    if status not in RUN_STATUSES:
        raise ContractError("state.status is invalid")
    reason = document["reason"]
    if status in {"failed", "interrupted", "abandoned"}:
        if not isinstance(reason, str) or not reason:
            raise ContractError("stopped state requires a reason")
    elif reason is not None:
        raise ContractError("non-stopped state reason must be null")
    tracker_run_id = document["tracker_run_id"]
    if tracker_run_id is not None and (
        not isinstance(tracker_run_id, str)
        or not TRACKER_ID_PATTERN.fullmatch(tracker_run_id)
    ):
        raise ContractError("state.tracker_run_id is invalid")
    result = document["result"]
    if status == "done":
        result = _mapping(result, "state.result")
        if document["action"] == "train":
            _exact_fields(result, {"model_id"})
            _model_id(result["model_id"])
        elif document["action"] == "eval":
            _exact_fields(result, {"metrics"})
            _owned_path(result["metrics"], "state.result.metrics")
        else:
            _exact_fields(result, {"export"})
            _owned_path(result["export"], "state.result.export")
    elif result is not None:
        raise ContractError("non-completed state.result must be null")
    for field in ("best_checkpoint", "last_checkpoint"):
        value = document[field]
        if value is not None:
            _owned_path(value, f"state.{field}")
    if document["action"] != "train" and (
        document["best_checkpoint"] is not None
        or document["last_checkpoint"] is not None
    ):
        raise ContractError("only training Runs may record checkpoints")
    for field in ("created_at", "updated_at"):
        _timestamp(document[field], f"state.{field}")
    _json_compatible(document)


def validate_model(document: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "model_id",
        "experiment",
        "variant",
        "training_group",
        "seed",
        "device",
        "training_fingerprint",
        "producer_run",
        "checkpoint",
        "created_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "model.schema_version")
    _model_id(document["model_id"])
    _identity(document["experiment"], "model Experiment")
    _identity(document["variant"], "model Variant")
    _identity(document["training_group"], "Training Group")
    _seed(document["seed"])
    if not isinstance(document["device"], str) or not document["device"]:
        raise ContractError("model.device is invalid")
    _digest(document["training_fingerprint"], "model.training_fingerprint")
    _run_id(document["producer_run"])
    checkpoint = _mapping(document["checkpoint"], "model.checkpoint")
    _exact_fields(checkpoint, {"path", "digest"})
    _variant_output_path(checkpoint["path"], "model.checkpoint.path")
    _digest(checkpoint["digest"], "model.checkpoint.digest")
    _timestamp(document["created_at"], "model.created_at")
    _json_compatible(document)


def validate_evaluation(
    document: dict[str, Any],
    snapshot: Mapping[str, Any],
    request: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "schema_version",
        "model_id",
        "evaluation_case",
        "case_fingerprint",
        "primary",
        "values",
        "artifacts",
        "evaluated_at",
    }
    _exact_fields(document, expected)
    _integer_one(document["schema_version"], "evaluation.schema_version")
    _model_id(document["model_id"])
    _identity(document["evaluation_case"], "evaluation case")
    _digest(document["case_fingerprint"], "evaluation.case_fingerprint")
    if request is not None:
        target = request["target"]
        if (
            document["model_id"] != target["model_id"]
            or document["evaluation_case"] != target["evaluation_case"]
            or document["case_fingerprint"] != request["identity_fingerprint"]
        ):
            raise ContractError("evaluation ownership mismatch")
    metrics = metric_spec(snapshot["variant"], document["evaluation_case"])
    values = _mapping(document["values"], "evaluation.values")
    if set(values) != set(metrics["report"]):
        raise ContractError("evaluation values do not match metrics.report")
    for value in values.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ContractError("evaluation metric must be a finite JSON number")
    primary = _mapping(document["primary"], "evaluation.primary")
    _exact_fields(primary, {"name", "value"})
    if (
        primary["name"] != metrics["primary"]
        or primary["value"] != values[metrics["primary"]]
    ):
        raise ContractError("evaluation primary metric is inconsistent")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(set(artifacts)):
        raise ContractError("evaluation artifacts must be a unique list")
    for artifact in artifacts:
        _owned_path(artifact, "evaluation artifact")
    _timestamp(document["evaluated_at"], "evaluation.evaluated_at")


def fingerprint_document(document: Mapping[str, Any]) -> str:
    _json_compatible(document)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class RunStore:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        now: Callable[[], datetime] | None = None,
        nonce: Callable[[int], bytes] | None = None,
    ):
        self.repository = repository
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._nonce = nonce or secrets.token_bytes

    def freeze(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        source_digest: str,
    ) -> dict[str, Any]:
        snapshot = {
            "schema_version": 1,
            "experiment": deepcopy(experiment.document),
            "variant": deepcopy(variant.document),
            "provenance": {
                "variant_file": f"{variant.document['name']}/variant.yaml",
                "source_digest": source_digest,
                "vcs_revision": None,
                "frozen_at": _utc_timestamp(self._now()),
            },
        }
        validate_snapshot(snapshot)
        return snapshot

    def allocate(
        self,
        experiment: ExperimentRecord,
        variant: VariantRecord,
        *,
        action: str,
        target: Mapping[str, Any],
        exec_info: Mapping[str, Any],
        source_digest: str,
        identity_fingerprint: str,
        retry_of: str | None = None,
        snapshot: dict[str, Any] | None = None,
        catalog_validator: (
            Callable[[list[RunRecord], list[ModelRecord]], None] | None
        ) = None,
    ) -> RunRecord:
        root = self.variant_root(
            experiment.document["name"],
            variant.document["name"],
            create=True,
        )
        runs = root / "runs"
        _ensure_directory(runs)
        with directory_lock(root):
            if catalog_validator is not None:
                catalog_validator(
                    self.scan(
                        experiment=experiment.document["name"],
                        variant=variant.document["name"],
                    ),
                    self.scan_models(
                        experiment=experiment.document["name"],
                        variant=variant.document["name"],
                    ),
                )
            run_id = f"run-{self._next_run_number(runs):03d}"
            timestamp = _utc_timestamp(self._now())
            if snapshot is None:
                snapshot = self.freeze(experiment, variant, source_digest)
            request = {
                "schema_version": 1,
                "run_id": run_id,
                "experiment": experiment.document["name"],
                "variant": variant.document["name"],
                "action": action,
                "target": deepcopy(dict(target)),
                "retry_of": retry_of,
                "source_digest": source_digest,
                "identity_fingerprint": identity_fingerprint,
                "exec": deepcopy(dict(exec_info)),
                "created_at": timestamp,
            }
            state = {
                "schema_version": 1,
                "run_id": run_id,
                "action": action,
                "status": "allocated",
                "reason": None,
                "tracker_run_id": None,
                "result": None,
                "best_checkpoint": None,
                "last_checkpoint": None,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            validate_snapshot(snapshot)
            validate_request(request)
            validate_state(state)
            candidate = Path(tempfile.mkdtemp(prefix=f".{run_id}.candidate-", dir=runs))
            try:
                (candidate / "metrics").mkdir()
                (candidate / "artifacts/checkpoints").mkdir(parents=True)
                atomic_write_new(candidate / "snapshot.yaml", dump_yaml(snapshot))
                atomic_write_new(candidate / "request.json", _json_text(request))
                atomic_write_new(candidate / "state.json", _json_text(state))
                target_path = runs / run_id
                if os.path.lexists(target_path):
                    raise AlreadyExistsError(f"Run already exists: {run_id}")
                os.rename(candidate, target_path)
                _fsync_directory(runs)
                return RunRecord(
                    target_path,
                    f"{request['experiment']}/{request['variant']}/{run_id}",
                    snapshot,
                    request,
                    state,
                )
            finally:
                if candidate.exists():
                    shutil.rmtree(candidate)

    def update_state(self, record: RunRecord, **changes: Any) -> RunRecord:
        if record.state["status"] in TERMINAL_STATUSES:
            raise ContractError(f"terminal Run is sealed: {record.address}")
        state = dict(record.state)
        state.update(changes)
        state["updated_at"] = _utc_timestamp(self._now())
        validate_state(state)
        atomic_replace(record.path / "state.json", _json_text(state))
        return RunRecord(
            record.path,
            record.address,
            record.snapshot,
            record.request,
            state,
        )

    def load(self, experiment: str, variant: str, run_id: str) -> RunRecord:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        _run_id(run_id)
        root = self.variant_root(experiment, variant, create=False)
        path = root / "runs" / run_id
        if not os.path.lexists(path):
            raise NotFoundError(f"Run not found: {experiment}/{variant}/{run_id}")
        _existing_directory(path)
        snapshot = load_yaml_file(path / "snapshot.yaml")
        request = _load_json(path / "request.json")
        state = _load_json(path / "state.json")
        validate_snapshot(snapshot)
        validate_request(request)
        validate_state(state)
        address = f"{experiment}/{variant}/{run_id}"
        if (
            request["run_id"] != run_id
            or request["experiment"] != experiment
            or request["variant"] != variant
            or state["run_id"] != run_id
            or state["action"] != request["action"]
            or snapshot["experiment"]["name"] != experiment
            or snapshot["variant"]["name"] != variant
            or snapshot["provenance"]["source_digest"] != request["source_digest"]
        ):
            raise OwnershipError(f"Run ownership mismatch: {address}")
        return RunRecord(path, address, snapshot, request, state)

    def scan(
        self,
        *,
        experiment: str | None = None,
        variant: str | None = None,
    ) -> list[RunRecord]:
        if experiment is not None:
            _identity(experiment, "Experiment")
        if variant is not None:
            _identity(variant, "Variant")
        outputs = self.repository.outputs
        if not os.path.lexists(outputs):
            return []
        _existing_directory(outputs)
        records: list[RunRecord] = []
        for experiment_entry in _catalog_directories(outputs, ignore={"index.db"}):
            _identity(experiment_entry.name, "Run Experiment")
            if experiment is not None and experiment_entry.name != experiment:
                continue
            for variant_entry in _catalog_directories(experiment_entry):
                _identity(variant_entry.name, "Run Variant")
                if variant is not None and variant_entry.name != variant:
                    continue
                self._reject_legacy_layout(variant_entry)
                runs = variant_entry / "runs"
                if not os.path.lexists(runs):
                    continue
                _existing_directory(runs)
                for run_entry in _catalog_directories(runs):
                    _run_id(run_entry.name)
                    records.append(
                        self.load(
                            experiment_entry.name,
                            variant_entry.name,
                            run_entry.name,
                        )
                    )
        return sorted(records, key=_run_sort_key)

    def allocate_model(
        self,
        record: RunRecord,
        *,
        checkpoint: str,
        checkpoint_digest: str,
    ) -> ModelRecord:
        if record.request["action"] != "train":
            raise ContractError("only a Train Run can produce a Model")
        target = record.request["target"]
        root = self.variant_root(
            record.request["experiment"],
            record.request["variant"],
            create=True,
        )
        models = root / "models"
        _ensure_directory(models)
        with directory_lock(root):
            for existing in self.scan_models(
                experiment=record.request["experiment"],
                variant=record.request["variant"],
            ):
                if (
                    existing.document["training_group"] == target["training_group"]
                    and existing.document["seed"] == target["seed"]
                ):
                    raise AlreadyExistsError(
                        "Model already exists for Training Group and seed"
                    )
            timestamp = _utc_timestamp(self._now())
            model_id = self._new_model_id(
                record.request["experiment"],
                record.request["variant"],
                models,
            )
            document = {
                "schema_version": 1,
                "model_id": model_id,
                "experiment": record.request["experiment"],
                "variant": record.request["variant"],
                "training_group": target["training_group"],
                "seed": target["seed"],
                "device": record.request["exec"]["device"],
                "training_fingerprint": record.request["identity_fingerprint"],
                "producer_run": record.request["run_id"],
                "checkpoint": {
                    "path": f"runs/{record.request['run_id']}/{checkpoint}",
                    "digest": checkpoint_digest,
                },
                "created_at": timestamp,
            }
            validate_model(document)
            candidate = Path(
                tempfile.mkdtemp(prefix=f".{model_id}.candidate-", dir=models)
            )
            try:
                atomic_write_new(candidate / "model.json", _json_text(document))
                target_path = models / model_id
                if os.path.lexists(target_path):
                    raise AlreadyExistsError(f"Model already exists: {model_id}")
                os.rename(candidate, target_path)
                _fsync_directory(models)
                return ModelRecord(
                    target_path,
                    f"{document['experiment']}/{document['variant']}/{model_id}",
                    document,
                )
            finally:
                if candidate.exists():
                    shutil.rmtree(candidate)

    def load_model(self, experiment: str, variant: str, model_id: str) -> ModelRecord:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        _model_id(model_id)
        root = self.variant_root(experiment, variant, create=False)
        path = root / "models" / model_id
        if not os.path.lexists(path):
            raise NotFoundError(f"Model not found: {experiment}/{variant}/{model_id}")
        _existing_directory(path)
        document = _load_json(path / "model.json")
        validate_model(document)
        if (
            document["model_id"] != model_id
            or document["experiment"] != experiment
            or document["variant"] != variant
        ):
            raise OwnershipError(
                f"Model ownership mismatch: {experiment}/{variant}/{model_id}"
            )
        checkpoint = self.resolve_model_checkpoint(
            ModelRecord(path, f"{experiment}/{variant}/{model_id}", document)
        )
        if _file_digest(checkpoint) != document["checkpoint"]["digest"]:
            raise ContractError("Model checkpoint digest changed")
        return ModelRecord(path, f"{experiment}/{variant}/{model_id}", document)

    def scan_models(
        self,
        *,
        experiment: str,
        variant: str,
    ) -> list[ModelRecord]:
        try:
            root = self.variant_root(experiment, variant, create=False)
        except NotFoundError:
            return []
        models = root / "models"
        if not os.path.lexists(models):
            return []
        _existing_directory(models)
        result: list[ModelRecord] = []
        for entry in _catalog_directories(models):
            _model_id(entry.name)
            result.append(self.load_model(experiment, variant, entry.name))
        return sorted(
            result,
            key=lambda item: (
                item.document["created_at"],
                item.document["model_id"].encode("utf-8"),
            ),
        )

    def resolve_model_checkpoint(self, model: ModelRecord) -> Path:
        root = self.variant_root(
            model.document["experiment"],
            model.document["variant"],
            create=False,
        )
        relative = PurePosixPath(model.document["checkpoint"]["path"])
        path = root.joinpath(*relative.parts)
        _contained_regular_file(root, path, "Model checkpoint")
        return path

    def evaluation_document(
        self,
        record: RunRecord,
        *,
        values: Mapping[str, Any],
        artifacts: Sequence[str],
    ) -> dict[str, Any]:
        case = record.request["target"]["evaluation_case"]
        metrics = metric_spec(record.snapshot["variant"], case)
        document = {
            "schema_version": 1,
            "model_id": record.request["target"]["model_id"],
            "evaluation_case": case,
            "case_fingerprint": record.request["identity_fingerprint"],
            "primary": {
                "name": metrics["primary"],
                "value": values.get(metrics["primary"]),
            },
            "values": dict(values),
            "artifacts": list(artifacts),
            "evaluated_at": _utc_timestamp(self._now()),
        }
        validate_evaluation(document, record.snapshot, record.request)
        return document

    def load_evaluation(self, record: RunRecord) -> dict[str, Any]:
        if record.request["action"] != "eval":
            raise ContractError("Run is not an evaluation")
        path = record.path / "metrics/eval.json"
        if not os.path.lexists(path):
            raise ContractError(f"completed Eval Run has no metrics: {record.address}")
        document = _load_json(path)
        validate_evaluation(document, record.snapshot, record.request)
        return document

    def load_training_metric_summary(
        self,
        record: RunRecord,
        *,
        required: bool | None = None,
    ) -> dict[str, Any]:
        if record.request["action"] != "train":
            return {}
        if "local" not in validate_tracker(record.snapshot["variant"]["tracker"]):
            return {}
        if required is None:
            required = record.state["status"] == "done"
        path = record.path / "metrics/train-summary.json"
        if not os.path.lexists(path):
            if required:
                raise ContractError(
                    f"completed Train Run has no metric summary: {record.address}"
                )
            return {}
        _contained_regular_file(record.path, path, "Train metric summary")
        summary = _load_json(path)
        _validate_training_metric_summary(summary)
        return deepcopy(summary["metrics"])

    def load_training_metrics(self, record: RunRecord) -> dict[str, Any]:
        chunk = self.load_training_metric_chunk(record, offset=0)
        events = chunk["events"]
        partial = chunk["partial"]
        summary_document = _training_metric_summary(events)
        summary_path = record.path / "metrics/train-summary.json"
        if os.path.lexists(summary_path):
            _contained_regular_file(record.path, summary_path, "Train metric summary")
            persisted = _load_json(summary_path)
            _validate_training_metric_summary(persisted)
            if persisted != summary_document:
                raise ContractError("Train metric summary disagrees with history")
        elif record.state["status"] == "done":
            raise ContractError(
                f"completed Train Run has no metric summary: {record.address}"
            )
        if record.state["status"] == "done" and partial:
            raise ContractError("completed Train metric history is partial")
        return {
            "events": events,
            "partial": partial,
            "summary": summary_document["metrics"],
        }

    def load_training_metric_chunk(
        self,
        record: RunRecord,
        *,
        offset: int,
    ) -> dict[str, Any]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ContractError("Train metric history offset is invalid")
        if record.request["action"] != "train":
            raise ContractError("Run is not a Train Run")
        if "local" not in validate_tracker(record.snapshot["variant"]["tracker"]):
            raise ContractError("Train Run does not use local tracking")
        history_path = record.path / "metrics/train.jsonl"
        if not os.path.lexists(history_path):
            if offset:
                raise ContractError("Train metric history disappeared while following")
            if record.state["status"] == "done":
                raise ContractError(
                    f"completed Train Run has no metric history: {record.address}"
                )
            return {"events": [], "offset": 0, "partial": False}
        _contained_regular_file(record.path, history_path, "Train metric history")
        events: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        partial = False
        next_offset = offset
        try:
            with history_path.open("rb") as handle:
                if os.fstat(handle.fileno()).st_size < offset:
                    raise ContractError("Train metric history was truncated")
                handle.seek(offset)
                for line in handle:
                    if not line.endswith(b"\n"):
                        partial = True
                        break
                    row = _load_training_metric_event(line)
                    key = (row["name"], row["step"])
                    if key in seen:
                        raise ContractError(
                            "Train metric history contains a duplicate step"
                        )
                    seen.add(key)
                    events.append(row)
                    next_offset = handle.tell()
        except OSError as error:
            raise ContractError("Train metric history is unavailable") from error
        return {
            "events": events,
            "offset": next_offset,
            "partial": partial,
        }

    def validate_completed_training_metrics(
        self,
        record: RunRecord,
        result: Any,
    ) -> dict[str, str]:
        uses_local = "local" in validate_tracker(record.snapshot["variant"]["tracker"])
        if not uses_local:
            if result is not None:
                raise ContractError("Train worker returned unexpected metric files")
            return {}
        if not isinstance(result, dict) or result != {
            "history": "metrics/train.jsonl",
            "summary": "metrics/train-summary.json",
        }:
            raise ContractError("Train worker metric files are invalid")
        metrics = self.load_training_metrics(record)
        if metrics["partial"]:
            raise ContractError("completed Train metric history is partial")
        return {
            relative: _file_digest(record.path / relative)
            for relative in result.values()
        }

    def direct_retry(self, record: RunRecord) -> RunRecord | None:
        matches = [
            candidate
            for candidate in self.scan(
                experiment=record.request["experiment"],
                variant=record.request["variant"],
            )
            if candidate.request["retry_of"] == record.request["run_id"]
        ]
        if len(matches) > 1:
            raise ContractError("Run has multiple direct retries")
        return matches[0] if matches else None

    def variant_root(
        self,
        experiment: str,
        variant: str,
        *,
        create: bool,
    ) -> Path:
        _identity(experiment, "Experiment")
        _identity(variant, "Variant")
        outputs = self.repository.outputs
        if create:
            _ensure_directory(outputs)
            experiment_root = outputs / experiment
            _ensure_directory(experiment_root)
            root = experiment_root / variant
            _ensure_directory(root)
        else:
            root = outputs / experiment / variant
            if not os.path.lexists(root):
                raise NotFoundError(f"Variant output not found: {experiment}/{variant}")
            _existing_directory(root)
        self._reject_legacy_layout(root)
        return root

    @staticmethod
    def json_text(document: Mapping[str, Any]) -> str:
        return _json_text(document)

    @staticmethod
    def _next_run_number(runs: Path) -> int:
        maximum = 0
        for entry in os.scandir(runs):
            if entry.name.startswith("."):
                continue
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise ContractError(f"invalid Run catalog entry: {entry.name}")
            _run_id(entry.name)
            maximum = max(maximum, int(entry.name.removeprefix("run-")))
        return maximum + 1

    def _new_model_id(self, experiment: str, variant: str, models: Path) -> str:
        for _ in range(16):
            timestamp = str(
                self._now().timestamp_ns()
                if hasattr(self._now(), "timestamp_ns")
                else int(self._now().timestamp() * 1_000_000_000)
            )
            payload = b"\0".join(
                (
                    experiment.encode("utf-8"),
                    variant.encode("utf-8"),
                    timestamp.encode("ascii"),
                    self._nonce(16),
                )
            )
            model_id = f"model-{hashlib.sha256(payload).hexdigest()[:32]}"
            if not os.path.lexists(models / model_id):
                return model_id
        raise AlreadyExistsError("unable to allocate a unique Model ID")

    @staticmethod
    def _reject_legacy_layout(root: Path) -> None:
        if not os.path.lexists(root):
            return
        for entry in os.scandir(root):
            if RUN_ID_PATTERN.fullmatch(entry.name):
                raise ContractError(
                    f"legacy pipeline Run layout is unsupported: {entry.path}"
                )


def _required_components(
    variant: Mapping[str, Any],
    required: frozenset[str],
    action: str,
) -> dict[str, str]:
    components = _mapping(variant["components"], "variant.components")
    missing = required - components.keys()
    if missing:
        raise ContractError(
            f"{action} components are missing: {', '.join(sorted(missing))}"
        )
    return {kind: components[kind] for kind in sorted(required)}


def _run_sort_key(record: RunRecord) -> tuple[Any, ...]:
    return (
        record.request["experiment"].encode("utf-8"),
        record.request["variant"].encode("utf-8"),
        int(record.request["run_id"].removeprefix("run-")),
        record.request["run_id"].encode("utf-8"),
    )


def _json_text(document: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    _existing_regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON document must be a mapping: {path}")
    return value


def _load_training_metric_event(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("Train metric history contains invalid JSON") from error
    _validate_training_metric_event(value)
    return value


def _validate_training_metric_event(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "name",
        "step",
        "value",
    }:
        raise ContractError("Train metric event fields are invalid")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != 1
    ):
        raise ContractError("Train metric event schema version is invalid")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ContractError("Train metric name is invalid")
    if (
        isinstance(value["step"], bool)
        or not isinstance(value["step"], int)
        or value["step"] < 0
    ):
        raise ContractError("Train metric step is invalid")
    if (
        isinstance(value["value"], bool)
        or not isinstance(value["value"], (int, float))
        or not math.isfinite(float(value["value"]))
    ):
        raise ContractError("Train metric value is invalid")


def _training_metric_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    last: dict[str, tuple[int, Any]] = {}
    for event in events:
        name = event["name"]
        counts[name] = counts.get(name, 0) + 1
        last[name] = (event["step"], event["value"])
    return {
        "schema_version": 1,
        "events": len(events),
        "metrics": {
            name: {
                "count": counts[name],
                "last_step": last[name][0],
                "last_value": last[name][1],
            }
            for name in sorted(counts, key=lambda item: item.encode("utf-8"))
        },
    }


def _validate_training_metric_summary(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "events",
        "metrics",
    }:
        raise ContractError("Train metric summary fields are invalid")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != 1
    ):
        raise ContractError("Train metric summary schema version is invalid")
    if (
        isinstance(value["events"], bool)
        or not isinstance(value["events"], int)
        or value["events"] < 0
        or not isinstance(value["metrics"], dict)
    ):
        raise ContractError("Train metric summary is invalid")
    total = 0
    for name, metric in value["metrics"].items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(metric, dict)
            or set(metric) != {"count", "last_step", "last_value"}
        ):
            raise ContractError("Train metric summary entry is invalid")
        if (
            isinstance(metric["count"], bool)
            or not isinstance(metric["count"], int)
            or metric["count"] < 1
            or isinstance(metric["last_step"], bool)
            or not isinstance(metric["last_step"], int)
            or metric["last_step"] < 0
            or isinstance(metric["last_value"], bool)
            or not isinstance(metric["last_value"], (int, float))
            or not math.isfinite(float(metric["last_value"]))
        ):
            raise ContractError("Train metric summary entry is invalid")
        total += metric["count"]
    if total != value["events"]:
        raise ContractError("Train metric summary event count is invalid")


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o755)
    except FileExistsError:
        pass
    _existing_directory(path)


def _existing_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required directory is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise ContractError(f"required directory is invalid: {path}")


def _existing_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ContractError(f"required file is unavailable: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ContractError(f"required file is invalid: {path}")


def _contained_regular_file(root: Path, path: Path, location: str) -> None:
    try:
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ContractError(f"{location} is outside its owner") from error
    _existing_regular_file(path)


def _catalog_directories(path: Path, *, ignore: set[str] | None = None) -> list[Path]:
    ignored = ignore or set()
    result: list[Path] = []
    for entry in os.scandir(path):
        if entry.name.startswith(".") or entry.name in ignored:
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ContractError(f"invalid catalog entry: {entry.path}")
        result.append(Path(entry.path))
    return sorted(result, key=lambda item: item.name.encode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_digest(path: Path) -> str:
    _existing_regular_file(path)
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _exact_fields(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ContractError("document fields are invalid")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{location} must be a mapping")
    return value


def _identity(value: Any, location: str) -> None:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ContractError(f"{location} name is invalid")


def _run_id(value: Any) -> None:
    if not isinstance(value, str) or not RUN_ID_PATTERN.fullmatch(value):
        raise ContractError("invalid Run ID")


def _model_id(value: Any) -> None:
    if not isinstance(value, str) or not MODEL_ID_PATTERN.fullmatch(value):
        raise ContractError("invalid Model ID")


def _seed(value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_SEED
    ):
        raise ContractError(f"seed must be an integer from 0 to {MAX_SEED}")


def _digest(value: Any, location: str) -> None:
    if not isinstance(value, str) or not DIGEST_PATTERN.fullmatch(value):
        raise ContractError(f"{location} digest is invalid")


def _timestamp(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} timestamp is invalid")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{location} timestamp is invalid") from error


def _owned_path(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} path is invalid")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise ContractError(f"{location} path is invalid")


def _variant_output_path(value: Any, location: str) -> None:
    _owned_path(value, location)
    parsed = PurePosixPath(value)
    if not parsed.parts or parsed.parts[0] != "runs":
        raise ContractError(f"{location} must reference a Run artifact")


def _integer_one(value: Any, location: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ContractError(f"{location} must be integer 1")


def _json_compatible(value: Any) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("document contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _json_compatible(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("document mapping keys must be strings")
            _json_compatible(item)
        return
    raise ContractError("document contains a non-JSON value")


__all__ = [
    "ACTIONS",
    "EVAL_COMPONENTS",
    "EXPORT_COMPONENTS",
    "MAX_SEED",
    "MODEL_ID_PATTERN",
    "RUN_ID_PATTERN",
    "RUN_STATUSES",
    "TERMINAL_STATUSES",
    "TRAIN_COMPONENTS",
    "ModelRecord",
    "RunRecord",
    "RunStore",
    "evaluation_case",
    "fingerprint_document",
    "metric_spec",
    "validate_evaluation",
    "validate_evaluation_readiness",
    "validate_export_readiness",
    "validate_model",
    "validate_request",
    "validate_snapshot",
    "validate_state",
    "validate_training_readiness",
]
