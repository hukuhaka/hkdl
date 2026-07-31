"""Variant-managed action execution, Model publication, and retry."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .attempts import load_attempt, new_attempt, remove_attempt, write_attempt
from .authoring import Authoring, ExperimentRecord, VariantRecord
from .config import ContractError
from .runs import (
    ModelRecord,
    RunRecord,
    RunStore,
    TERMINAL_STATUSES,
    evaluation_case,
    fingerprint_document,
    metric_spec,
    validate_tracker,
    validate_evaluation_readiness,
    validate_export_readiness,
    validate_training_readiness,
)
from .runtime import (
    RuntimeFailure,
    RuntimeInterrupted,
    RuntimeOwnershipConflict,
    VariantRuntime,
)
from .storage import (
    LockUnavailableError,
    RepositoryPaths,
    atomic_write_new,
    compute_source_digest,
    publish_directory,
    publish_file,
    try_directory_lock,
)


class ExecutionFailure(RuntimeError):
    def __init__(self, action: str, message: str, *, address: str | None = None):
        super().__init__(message)
        self.action = action
        self.address = address


class ExecutionInterrupted(KeyboardInterrupt):
    def __init__(self, action: str, address: str):
        super().__init__(address)
        self.action = action
        self.address = address


class LifecycleConflict(RuntimeError):
    """The requested Variant-managed mutation conflicts with durable state."""


class RunExecution:
    def __init__(
        self,
        repository: RepositoryPaths,
        *,
        runtime: VariantRuntime | None = None,
        store: RunStore | None = None,
    ):
        self.repository = repository
        self.authoring = Authoring(repository)
        self.runtime = runtime or VariantRuntime()
        self.store = store or RunStore(repository)

    def train(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        *,
        seed: int = 0,
        device: str = "auto",
    ) -> RunRecord:
        experiment, variant, python, source_digest, snapshot = self._prepare(
            experiment_name,
            variant_name,
            action="train",
        )
        selected = validate_training_readiness(
            experiment.document,
            variant.document,
        )
        target = {"training_group": training_group, "seed": seed}
        fallback = {
            "dataset": variant.document["dataset"],
            "train": variant.document["train"],
            "components": selected,
        }
        preflight = self._preflight(
            python,
            variant,
            action="train",
            snapshot=snapshot,
            selected=selected,
            seed=seed,
            device=device,
            fallback=fallback,
            target=target,
        )
        fingerprint = self._action_fingerprint(
            action="train",
            source_digest=source_digest,
            selected=selected,
            identity=preflight["identity"],
            device=preflight["exec"]["device"],
        )

        def validate_slot(
            runs: list[RunRecord],
            models: list[ModelRecord],
        ) -> None:
            del models
            same_group = [
                item
                for item in runs
                if item.request["action"] == "train"
                and item.request["target"]["training_group"] == training_group
            ]
            if any(
                item.request["identity_fingerprint"] != fingerprint
                for item in same_group
            ):
                raise LifecycleConflict(
                    f"Training Group fingerprint changed: {training_group}"
                )
            if any(item.request["target"]["seed"] == seed for item in same_group):
                raise LifecycleConflict(
                    f"Training seed already has an execution: {training_group}/{seed}"
                )

        record = self.store.allocate(
            experiment,
            variant,
            action="train",
            target=target,
            exec_info=preflight["exec"],
            source_digest=source_digest,
            identity_fingerprint=fingerprint,
            snapshot=snapshot,
            catalog_validator=validate_slot,
        )
        return self._execute_train(
            record,
            variant,
            python=python,
            selected=selected,
            expected_identity=preflight["identity"],
        )

    def evaluate(
        self,
        experiment_name: str,
        variant_name: str,
        training_group: str,
        evaluation_case_name: str,
        *,
        seed: int | str | None = None,
        device: str = "auto",
    ) -> list[RunRecord]:
        experiment, variant, python, source_digest, snapshot = self._prepare(
            experiment_name,
            variant_name,
            action="eval",
        )
        models = [
            model
            for model in self.store.scan_models(
                experiment=experiment_name,
                variant=variant_name,
            )
            if model.document["training_group"] == training_group
        ]
        if not models:
            raise LifecycleConflict(f"Training Group has no Models: {training_group}")
        if seed is None:
            if len(models) != 1:
                raise LifecycleConflict(
                    f"Training Group has multiple Models; specify --seed: "
                    f"{training_group}"
                )
            selected_models = models
        elif seed == "all":
            selected_models = models
        else:
            selected_models = [
                model for model in models if model.document["seed"] == seed
            ]
            if not selected_models:
                raise LifecycleConflict(
                    f"Training Group seed has no Model: {training_group}/{seed}"
                )
        selected_models.sort(key=lambda item: item.document["seed"])

        selected = validate_evaluation_readiness(
            experiment.document,
            variant.document,
            case=evaluation_case_name,
        )
        case_document = evaluation_case(variant.document, evaluation_case_name)
        metrics = metric_spec(variant.document, evaluation_case_name)
        results: list[RunRecord] = []
        for model in selected_models:
            if self._evaluation_exists(
                experiment_name,
                variant_name,
                model.document["model_id"],
                evaluation_case_name,
            ):
                if seed == "all":
                    continue
                raise LifecycleConflict(
                    "Model and Evaluation Case already have an execution"
                )
            target = {
                "training_group": training_group,
                "seed": model.document["seed"],
                "model_id": model.document["model_id"],
                "evaluation_case": evaluation_case_name,
            }
            fallback = {
                "case": case_document,
                "metrics": metrics,
                "components": selected,
            }
            preflight = self._preflight(
                python,
                variant,
                action="eval",
                snapshot=snapshot,
                selected=selected,
                seed=model.document["seed"],
                device=device,
                fallback=fallback,
                target=target,
            )
            fingerprint = self._action_fingerprint(
                action="eval",
                source_digest=source_digest,
                selected=selected,
                identity=preflight["identity"],
            )

            def validate_slot(
                runs: list[RunRecord],
                models: list[ModelRecord],
            ) -> None:
                del models
                same_case = [
                    item
                    for item in runs
                    if item.request["action"] == "eval"
                    and item.request["target"]["evaluation_case"]
                    == evaluation_case_name
                ]
                if any(
                    item.request["identity_fingerprint"] != fingerprint
                    for item in same_case
                ):
                    raise LifecycleConflict(
                        f"Evaluation Case fingerprint changed: {evaluation_case_name}"
                    )
                if any(
                    item.request["target"]["model_id"] == model.document["model_id"]
                    for item in same_case
                ):
                    raise LifecycleConflict(
                        "Model and Evaluation Case already have an execution"
                    )

            record = self.store.allocate(
                experiment,
                variant,
                action="eval",
                target=target,
                exec_info=preflight["exec"],
                source_digest=source_digest,
                identity_fingerprint=fingerprint,
                snapshot=snapshot,
                catalog_validator=validate_slot,
            )
            results.append(
                self._execute_eval(
                    record,
                    variant,
                    model,
                    python=python,
                    selected=selected,
                    expected_identity=preflight["identity"],
                )
            )
        return results

    def export(
        self,
        experiment_name: str,
        variant_name: str,
        model_id: str,
        *,
        device: str = "auto",
    ) -> RunRecord:
        model = self.store.load_model(experiment_name, variant_name, model_id)
        experiment, variant, python, source_digest, snapshot = self._prepare(
            experiment_name,
            variant_name,
            action="export",
        )
        selected = validate_export_readiness(
            experiment.document,
            variant.document,
        )
        target = {"model_id": model_id}
        fallback = {
            "infer": variant.document["infer"],
            "components": selected,
        }
        preflight = self._preflight(
            python,
            variant,
            action="export",
            snapshot=snapshot,
            selected=selected,
            seed=model.document["seed"],
            device=device,
            fallback=fallback,
            target=target,
        )
        fingerprint = self._action_fingerprint(
            action="export",
            source_digest=source_digest,
            selected=selected,
            identity=preflight["identity"],
        )

        def validate_slot(
            runs: list[RunRecord],
            models: list[ModelRecord],
        ) -> None:
            del models
            if any(
                item.request["action"] == "export"
                and item.request["target"]["model_id"] == model_id
                for item in runs
            ):
                raise LifecycleConflict(f"Model already has an export: {model_id}")

        record = self.store.allocate(
            experiment,
            variant,
            action="export",
            target=target,
            exec_info=preflight["exec"],
            source_digest=source_digest,
            identity_fingerprint=fingerprint,
            snapshot=snapshot,
            catalog_validator=validate_slot,
        )
        return self._execute_export(
            record,
            variant,
            model,
            python=python,
            selected=selected,
            expected_identity=preflight["identity"],
        )

    def retry(
        self,
        experiment_name: str,
        variant_name: str,
        run_id: str,
    ) -> RunRecord:
        original = self.store.load(experiment_name, variant_name, run_id)
        abandoned_tracker: str | None = None
        try:
            with try_directory_lock(original.path):
                original = self.store.load(experiment_name, variant_name, run_id)
                if self.store.direct_retry(original) is not None:
                    raise LifecycleConflict(
                        f"Run already has a retry: {original.address}"
                    )
                if original.state["status"] == "done":
                    raise LifecycleConflict(
                        f"completed Run cannot be retried: {original.address}"
                    )
                journal = load_attempt(original.path / ".attempt.json")
                if original.state["status"] in {"allocated", "running"}:
                    if journal is not None and journal["phase"] == "ready":
                        return self._commit_receipt(original, journal)
                    checkpoint_changes: dict[str, Any] = {}
                    if journal is not None and journal["action"] == "train":
                        checkpoint = journal["checkpoint"]
                        if checkpoint["best"] is not None:
                            checkpoint_changes["best_checkpoint"] = checkpoint["best"]
                        if checkpoint["last"] is not None:
                            checkpoint_changes["last_checkpoint"] = checkpoint["last"]
                    original = self.store.update_state(
                        original,
                        status="abandoned",
                        reason="AbandonedExecution",
                        **checkpoint_changes,
                    )
                    abandoned_tracker = original.state["tracker_run_id"]
        except LockUnavailableError as error:
            raise LifecycleConflict(f"Run is busy: {original.address}") from error

        current_variant = self.authoring.check_variant(
            experiment_name,
            variant_name,
        )
        current_digest = compute_source_digest(current_variant.path / "src")
        if current_digest != original.request["source_digest"]:
            raise ContractError("Variant source changed since the original Run")
        try:
            python = self.runtime.prepare_environment(current_variant)
            if abandoned_tracker is not None:
                with try_directory_lock(original.path) as descriptor:
                    self.runtime.finish_tracker(
                        python,
                        current_variant,
                        tracker_run_id=abandoned_tracker,
                        status="KILLED",
                        lock_descriptor=descriptor,
                    )
        except RuntimeFailure as error:
            raise ExecutionFailure(
                original.request["action"],
                type(error).__name__,
                address=original.address,
            ) from error
        runtime_variant = VariantRecord(
            current_variant.path,
            current_variant.experiment,
            original.snapshot["variant"],
        )
        selected = self._selected_for_request(original)
        fallback = self._fallback_for_request(original, selected)
        preflight = self._preflight(
            python,
            runtime_variant,
            action=original.request["action"],
            snapshot=original.snapshot,
            selected=selected,
            seed=original.request["exec"]["seed"],
            device=original.request["exec"]["device"],
            fallback=fallback,
            target=original.request["target"],
        )
        fingerprint = self._action_fingerprint(
            action=original.request["action"],
            source_digest=current_digest,
            selected=selected,
            identity=preflight["identity"],
            device=(
                preflight["exec"]["device"]
                if original.request["action"] == "train"
                else None
            ),
        )
        if (
            fingerprint != original.request["identity_fingerprint"]
            or preflight["exec"] != original.request["exec"]
        ):
            raise ContractError("retry preflight differs from the original Run")
        experiment = ExperimentRecord(
            self.repository.experiments / experiment_name,
            original.snapshot["experiment"],
        )

        def validate_retry(
            runs: list[RunRecord],
            models: list[ModelRecord],
        ) -> None:
            del models
            if any(
                item.request["retry_of"] == original.request["run_id"] for item in runs
            ):
                raise LifecycleConflict(f"Run already has a retry: {original.address}")

        retry = self.store.allocate(
            experiment,
            runtime_variant,
            action=original.request["action"],
            target=original.request["target"],
            exec_info=original.request["exec"],
            source_digest=current_digest,
            identity_fingerprint=fingerprint,
            retry_of=original.request["run_id"],
            snapshot=original.snapshot,
            catalog_validator=validate_retry,
        )
        if retry.request["action"] == "train":
            resume_from = self._resume_checkpoint(original)
            return self._execute_train(
                retry,
                runtime_variant,
                python=python,
                selected=selected,
                expected_identity=preflight["identity"],
                resume_from=resume_from,
            )
        model = self.store.load_model(
            experiment_name,
            variant_name,
            retry.request["target"]["model_id"],
        )
        if retry.request["action"] == "eval":
            return self._execute_eval(
                retry,
                runtime_variant,
                model,
                python=python,
                selected=selected,
                expected_identity=preflight["identity"],
            )
        return self._execute_export(
            retry,
            runtime_variant,
            model,
            python=python,
            selected=selected,
            expected_identity=preflight["identity"],
        )

    def _prepare(
        self,
        experiment_name: str,
        variant_name: str,
        *,
        action: str,
    ) -> tuple[ExperimentRecord, VariantRecord, Path, str, dict[str, Any]]:
        variant = self.authoring.check_variant(experiment_name, variant_name)
        experiment = self.authoring.load_experiment(experiment_name)
        try:
            python = self.runtime.prepare_environment(variant)
        except RuntimeFailure as error:
            raise ExecutionFailure(action, str(error)) from error
        source_digest = compute_source_digest(variant.path / "src")
        snapshot = self.store.freeze(experiment, variant, source_digest)
        return experiment, variant, python, source_digest, snapshot

    def _preflight(
        self,
        python: Path,
        variant: VariantRecord,
        *,
        action: str,
        snapshot: dict[str, Any],
        selected: dict[str, str],
        seed: int,
        device: str,
        fallback: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.runtime.preflight(
            python,
            variant,
            action=action,
            cfg=snapshot,
            selected=selected,
            seed=seed,
            device=device,
            identity_fallback=deepcopy(fallback),
            runtime_target=deepcopy(target),
        )
        identity = result.get("identity")
        if not isinstance(identity, dict):
            raise ContractError("Variant preflight identity is invalid")
        return {"exec": result["exec"], "identity": identity}

    @staticmethod
    def _action_fingerprint(
        *,
        action: str,
        source_digest: str,
        selected: Mapping[str, str],
        identity: Mapping[str, Any],
        device: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "action": action,
            "source_digest": source_digest,
            "components": dict(selected),
            "identity": dict(identity),
        }
        if action == "train":
            payload["device"] = device
        return fingerprint_document(payload)

    def _execute_train(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
        resume_from: Path | None = None,
    ) -> RunRecord:
        return self._execute(
            record,
            variant,
            python=python,
            selected=selected,
            expected_identity=expected_identity,
            resume_from=resume_from,
        )

    def _execute_eval(
        self,
        record: RunRecord,
        variant: VariantRecord,
        model: ModelRecord,
        *,
        python: Path,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
    ) -> RunRecord:
        return self._execute(
            record,
            variant,
            python=python,
            selected=selected,
            expected_identity=expected_identity,
            model=model,
        )

    def _execute_export(
        self,
        record: RunRecord,
        variant: VariantRecord,
        model: ModelRecord,
        *,
        python: Path,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
    ) -> RunRecord:
        return self._execute(
            record,
            variant,
            python=python,
            selected=selected,
            expected_identity=expected_identity,
            model=model,
        )

    def _execute(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
        model: ModelRecord | None = None,
        resume_from: Path | None = None,
    ) -> RunRecord:
        action = record.request["action"]
        try:
            with try_directory_lock(record.path) as lock_descriptor:
                record, attempt_path, candidate = self._start_attempt(
                    record,
                    lock_descriptor=lock_descriptor,
                )
                tracker_run_id = self._ensure_tracker(
                    record,
                    variant,
                    python,
                    attempt_path,
                    lock_descriptor,
                )
                record = self.store.load(
                    record.request["experiment"],
                    record.request["variant"],
                    record.request["run_id"],
                )
                if action == "train":
                    result = self.runtime.train(
                        python,
                        variant,
                        cfg=record.snapshot,
                        selected=selected,
                        exec_info=record.request["exec"],
                        run_dir=record.path,
                        resume_from=resume_from,
                        tracker_run_id=tracker_run_id,
                        attempt_path=attempt_path,
                        lock_descriptor=lock_descriptor,
                        runtime_target=record.request["target"],
                    )
                elif action == "eval":
                    if model is None:
                        raise ContractError("Eval Run has no Model")
                    result = self.runtime.evaluate(
                        python,
                        variant,
                        cfg=record.snapshot,
                        selected=selected,
                        exec_info=record.request["exec"],
                        run_dir=record.path,
                        checkpoint=self.store.resolve_model_checkpoint(model),
                        results_dir=candidate,
                        tracker_run_id=tracker_run_id,
                        attempt_path=attempt_path,
                        lock_descriptor=lock_descriptor,
                        runtime_target=record.request["target"],
                    )
                else:
                    if model is None:
                        raise ContractError("Export Run has no Model")
                    result = self.runtime.export(
                        python,
                        variant,
                        cfg=record.snapshot,
                        selected=selected,
                        exec_info=record.request["exec"],
                        run_dir=record.path,
                        export_dir=candidate,
                        checkpoint=self.store.resolve_model_checkpoint(model),
                        tracker_run_id=tracker_run_id,
                        attempt_path=attempt_path,
                        lock_descriptor=lock_descriptor,
                        runtime_target=record.request["target"],
                    )
                self._record_worker_result(attempt_path, result)
                self._verify_identity_after_execution(
                    record,
                    variant,
                    python=python,
                    selected=selected,
                    expected_identity=expected_identity,
                )
                if action == "eval":
                    values = result.get("values")
                    if not isinstance(values, dict):
                        raise RuntimeFailure("Evaluator values must be a mapping")
                    self.runtime.log_tracker_metrics(
                        python,
                        variant,
                        tracker_run_id=tracker_run_id,
                        values=values,
                        lock_descriptor=lock_descriptor,
                    )
                completed = self._complete_from_journal(
                    record,
                    attempt_path,
                    variant=variant,
                )
                self.runtime.finish_tracker(
                    python,
                    variant,
                    tracker_run_id=tracker_run_id,
                    status="FINISHED",
                    lock_descriptor=lock_descriptor,
                )
                return completed
        except LockUnavailableError as error:
            raise LifecycleConflict(f"Run is busy: {record.address}") from error
        except (RuntimeInterrupted, KeyboardInterrupt) as error:
            self._stop_attempt(
                record,
                variant,
                python,
                status="interrupted",
                reason="interrupted",
            )
            raise ExecutionInterrupted(action, record.address) from error
        except RuntimeOwnershipConflict as error:
            self._stop_attempt(
                record,
                variant,
                python,
                status="failed",
                reason="TrackerOwnershipConflict",
            )
            raise LifecycleConflict(
                f"Tracker ownership conflict: {record.address}"
            ) from error
        except Exception as error:
            self._stop_attempt(
                record,
                variant,
                python,
                status="failed",
                reason=type(error).__name__,
            )
            raise ExecutionFailure(
                action,
                type(error).__name__,
                address=record.address,
            ) from error

    def _start_attempt(
        self,
        record: RunRecord,
        *,
        lock_descriptor: int,
    ) -> tuple[RunRecord, Path, Path]:
        attempt_path = record.path / ".attempt.json"
        if os.path.lexists(attempt_path):
            raise ContractError("Run already has an attempt journal")
        if record.request["action"] == "eval":
            candidate_relative = "artifacts/.results.candidate"
        elif record.request["action"] == "export":
            candidate_relative = "artifacts/.export.candidate"
        else:
            candidate_relative = None
        candidate = (
            record.path / candidate_relative
            if candidate_relative is not None
            else record.path
        )
        if candidate_relative is not None:
            if os.path.lexists(candidate):
                raise ContractError("action candidate already exists")
            candidate.mkdir()
        journal = new_attempt(
            action=record.request["action"],
            tracker_run_id=record.state["tracker_run_id"],
            candidate=candidate_relative,
        )
        write_attempt(
            attempt_path,
            journal,
            directory_descriptor=lock_descriptor,
        )
        try:
            record = self.store.update_state(
                record,
                status="running",
                reason=None,
            )
        except BaseException:
            remove_attempt(attempt_path)
            if candidate_relative is not None:
                shutil.rmtree(candidate)
            raise
        return record, attempt_path, candidate

    def _ensure_tracker(
        self,
        record: RunRecord,
        variant: VariantRecord,
        python: Path,
        attempt_path: Path,
        lock_descriptor: int,
    ) -> str | None:
        tracker_run_id = record.state["tracker_run_id"]
        if "mlflow" in validate_tracker(record.snapshot["variant"]["tracker"]):
            target = record.request["target"]
            metadata = {
                "action": record.request["action"],
                "training_group": target.get("training_group"),
                "seed": target.get("seed"),
                "model_id": target.get("model_id"),
                "evaluation_case": target.get("evaluation_case"),
                "retry_of": record.request["retry_of"],
            }
            tracker_run_id = self.runtime.ensure_tracker(
                python,
                variant,
                cfg=record.snapshot,
                run_dir=record.path,
                current_tracker_run_id=tracker_run_id,
                lock_descriptor=lock_descriptor,
                metadata=metadata,
            )
            record = self.store.update_state(
                record,
                tracker_run_id=tracker_run_id,
            )
            journal = load_attempt(attempt_path)
            if journal is None:
                raise ContractError("attempt journal disappeared")
            journal["tracker_run_id"] = tracker_run_id
            write_attempt(attempt_path, journal)
        return tracker_run_id

    def _verify_identity_after_execution(
        self,
        record: RunRecord,
        variant: VariantRecord,
        *,
        python: Path,
        selected: dict[str, str],
        expected_identity: dict[str, Any],
    ) -> None:
        if (
            compute_source_digest(variant.path / "src")
            != record.request["source_digest"]
        ):
            raise RuntimeFailure("Variant source changed during execution")
        result = self._preflight(
            python,
            variant,
            action=record.request["action"],
            snapshot=record.snapshot,
            selected=selected,
            seed=record.request["exec"]["seed"],
            device=record.request["exec"]["device"],
            fallback=self._fallback_for_request(record, selected),
            target=record.request["target"],
        )
        if (
            result["exec"] != record.request["exec"]
            or result["identity"] != expected_identity
        ):
            raise RuntimeFailure("Variant action identity changed during execution")

    def _complete_from_journal(
        self,
        record: RunRecord,
        attempt_path: Path,
        *,
        variant: VariantRecord,
    ) -> RunRecord:
        journal = load_attempt(attempt_path)
        if journal is None or journal["phase"] not in {"worker_done", "ready"}:
            raise ContractError("attempt journal has no durable worker result")
        if journal["action"] == "train":
            return self._complete_train(record, journal, attempt_path)
        if journal["action"] == "eval":
            return self._complete_eval(record, journal, attempt_path)
        return self._complete_export(record, journal, attempt_path)

    def _complete_train(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        attempt_path: Path,
    ) -> RunRecord:
        result = journal["result"]
        if journal["phase"] == "worker_done":
            best = self._checkpoint_result(record, result.get("best_checkpoint"))
            last = self._checkpoint_result(record, result.get("last_checkpoint"))
            digests = {
                best: _file_digest(record.path / best),
                last: _file_digest(record.path / last),
            }
            metric_files = result.get("metrics")
            metric_digests = self.store.validate_completed_training_metrics(
                record,
                metric_files,
            )
        else:
            best = result.get("best_checkpoint")
            last = result.get("last_checkpoint")
            digests = result.get("digests")
            metric_files = result.get("metrics")
            metric_digests = result.get("metric_digests")
            if (
                not isinstance(best, str)
                or not isinstance(last, str)
                or journal["checkpoint"] != {"best": best, "last": last}
                or not isinstance(digests, dict)
                or set(digests) != {best, last}
                or not isinstance(metric_digests, dict)
            ):
                raise ContractError("Train receipt is invalid")
            for relative in {best, last}:
                try:
                    validated = self._checkpoint_result(
                        record,
                        str(record.path / relative),
                    )
                except RuntimeFailure as error:
                    raise ContractError(
                        "Train receipt checkpoint is invalid"
                    ) from error
                if validated != relative or digests[relative] != _file_digest(
                    record.path / relative
                ):
                    raise ContractError("Train receipt checkpoint changed")
            current_metric_digests = self.store.validate_completed_training_metrics(
                record,
                metric_files,
            )
            if metric_digests != current_metric_digests:
                raise ContractError("Train receipt metric files changed")
        best_digest = digests[best]
        models = [
            model
            for model in self.store.scan_models(
                experiment=record.request["experiment"],
                variant=record.request["variant"],
            )
            if model.document["producer_run"] == record.request["run_id"]
        ]
        if len(models) > 1:
            raise ContractError("Train Run produced multiple Models")
        if models:
            model = models[0]
            if (
                model.document["checkpoint"]["path"]
                != f"runs/{record.request['run_id']}/{best}"
                or model.document["checkpoint"]["digest"] != best_digest
            ):
                raise ContractError("published Model disagrees with Train receipt")
        else:
            if journal["phase"] == "ready":
                raise ContractError("ready Train receipt has no Model")
            model = self.store.allocate_model(
                record,
                checkpoint=best,
                checkpoint_digest=best_digest,
            )
        if journal["phase"] == "ready":
            if result.get("model_id") != model.document["model_id"]:
                raise ContractError("Train receipt Model disagrees with publication")
        else:
            journal["phase"] = "ready"
            journal["checkpoint"] = {"best": best, "last": last}
            journal["result"] = {
                "model_id": model.document["model_id"],
                "best_checkpoint": best,
                "last_checkpoint": last,
                "digests": digests,
                "metrics": metric_files,
                "metric_digests": metric_digests,
            }
            write_attempt(attempt_path, journal)
        record = self.store.update_state(
            record,
            status="done",
            result={"model_id": model.document["model_id"]},
            best_checkpoint=best,
            last_checkpoint=last,
        )
        remove_attempt(attempt_path)
        return record

    def _complete_eval(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        attempt_path: Path,
    ) -> RunRecord:
        candidate_value = journal["candidate"]
        if not isinstance(candidate_value, str):
            raise ContractError("Eval receipt candidate is invalid")
        candidate = record.path / candidate_value
        final_results = record.path / "artifacts/results"
        final_metrics = record.path / "metrics/eval.json"
        if journal["phase"] == "worker_done":
            files = self._validate_candidate_files(
                candidate,
                journal["result"].get("files"),
            )
            artifacts = [f"artifacts/results/{relative}" for relative in files]
            values = journal["result"].get("values")
            if not isinstance(values, dict):
                raise RuntimeFailure("Evaluator values must be a mapping")
            document = self.store.evaluation_document(
                record,
                values=values,
                artifacts=artifacts,
            )
            metrics_candidate = record.path / "metrics/.eval.candidate.json"
            atomic_write_new(metrics_candidate, self.store.json_text(document))
            journal["phase"] = "ready"
            journal["result"] = {
                "document": document,
                "metrics_digest": _file_digest(metrics_candidate),
                "files": [
                    {"path": relative, "digest": _file_digest(candidate / relative)}
                    for relative in files
                ],
            }
            write_attempt(attempt_path, journal)
        document = journal["result"].get("document")
        files = journal["result"].get("files")
        if not isinstance(document, dict) or not isinstance(files, list):
            raise ContractError("Eval receipt is invalid")
        metrics_candidate = record.path / "metrics/.eval.candidate.json"
        if files:
            self._publish_or_verify_directory(candidate, final_results, files)
        elif candidate.exists():
            candidate.rmdir()
        self._publish_or_verify_file(
            metrics_candidate,
            final_metrics,
            self.store.json_text(document).encode("utf-8"),
        )
        record = self.store.update_state(
            record,
            status="done",
            result={"metrics": "metrics/eval.json"},
        )
        remove_attempt(attempt_path)
        return record

    def _complete_export(
        self,
        record: RunRecord,
        journal: dict[str, Any],
        attempt_path: Path,
    ) -> RunRecord:
        candidate_value = journal["candidate"]
        if not isinstance(candidate_value, str):
            raise ContractError("Export receipt candidate is invalid")
        candidate = record.path / candidate_value
        final = record.path / "artifacts/export"
        if journal["phase"] == "worker_done":
            paths = self._validate_candidate_files(
                candidate,
                journal["result"].get("files"),
                require_nonempty=True,
            )
            journal["phase"] = "ready"
            journal["result"] = {
                "files": [
                    {"path": relative, "digest": _file_digest(candidate / relative)}
                    for relative in paths
                ]
            }
            write_attempt(attempt_path, journal)
        files = journal["result"].get("files")
        if not isinstance(files, list) or not files:
            raise ContractError("Export receipt is invalid")
        self._publish_or_verify_directory(candidate, final, files)
        record = self.store.update_state(
            record,
            status="done",
            result={"export": "artifacts/export"},
        )
        remove_attempt(attempt_path)
        return record

    def _commit_receipt(
        self,
        record: RunRecord,
        journal: dict[str, Any],
    ) -> RunRecord:
        current_variant = self.authoring.check_variant(
            record.request["experiment"],
            record.request["variant"],
        )
        if (
            compute_source_digest(current_variant.path / "src")
            != record.request["source_digest"]
        ):
            raise ContractError("Variant source changed since the original Run")
        return self._complete_from_journal(
            record,
            record.path / ".attempt.json",
            variant=current_variant,
        )

    def _stop_attempt(
        self,
        record: RunRecord,
        variant: VariantRecord,
        python: Path,
        *,
        status: str,
        reason: str,
    ) -> None:
        try:
            current = self.store.load(
                record.request["experiment"],
                record.request["variant"],
                record.request["run_id"],
            )
        except Exception:
            return
        if current.state["status"] in TERMINAL_STATUSES:
            return
        attempt_path = current.path / ".attempt.json"
        journal = load_attempt(attempt_path)
        if journal is not None and journal["phase"] == "ready":
            return
        changes: dict[str, Any] = {
            "status": status,
            "reason": reason,
        }
        if journal is not None and journal["action"] == "train":
            checkpoint = journal["checkpoint"]
            if checkpoint["best"] is not None:
                changes["best_checkpoint"] = checkpoint["best"]
            if checkpoint["last"] is not None:
                changes["last_checkpoint"] = checkpoint["last"]
        current = self.store.update_state(current, **changes)
        if journal is not None and journal["candidate"] is not None:
            candidate = current.path / journal["candidate"]
            if candidate.exists() and not candidate.is_symlink():
                shutil.rmtree(candidate)
        remove_attempt(attempt_path)
        tracker = current.state["tracker_run_id"]
        if tracker is not None:
            try:
                with try_directory_lock(current.path) as descriptor:
                    self.runtime.finish_tracker(
                        python,
                        variant,
                        tracker_run_id=tracker,
                        status="KILLED" if status == "interrupted" else "FAILED",
                        lock_descriptor=descriptor,
                    )
            except Exception:
                pass

    def _record_worker_result(
        self,
        attempt_path: Path,
        result: dict[str, Any],
    ) -> None:
        journal = load_attempt(attempt_path)
        if journal is None:
            raise ContractError("attempt journal disappeared")
        if journal["phase"] == "running":
            journal["phase"] = "worker_done"
            journal["result"] = result
            write_attempt(attempt_path, journal)
        elif journal["phase"] != "worker_done":
            raise ContractError("attempt journal phase is invalid")

    def _checkpoint_result(self, record: RunRecord, value: Any) -> str:
        if not isinstance(value, str):
            raise RuntimeFailure("Trainer checkpoint result must be a path")
        path = Path(value)
        if not path.is_absolute():
            raise RuntimeFailure("Trainer checkpoint result must be absolute")
        try:
            relative = path.relative_to(record.path)
            resolved = path.resolve(strict=True)
            resolved.relative_to((record.path / "artifacts/checkpoints").resolve())
        except (OSError, ValueError) as error:
            raise RuntimeFailure("Trainer checkpoint is outside its root") from error
        if path.is_symlink() or not path.is_file():
            raise RuntimeFailure("Trainer checkpoint must be a regular file")
        return relative.as_posix()

    def _resume_checkpoint(self, record: RunRecord) -> Path | None:
        relative = record.state["last_checkpoint"]
        if relative is None:
            return None
        path = record.path / relative
        if path.is_symlink() or not path.is_file():
            raise ContractError("retry checkpoint is invalid")
        try:
            path.resolve(strict=True).relative_to(
                (record.path / "artifacts/checkpoints").resolve(strict=True)
            )
        except (OSError, ValueError) as error:
            raise ContractError("retry checkpoint is outside its root") from error
        return path.resolve()

    def _selected_for_request(self, record: RunRecord) -> dict[str, str]:
        experiment = record.snapshot["experiment"]
        variant = record.snapshot["variant"]
        if record.request["action"] == "train":
            return validate_training_readiness(experiment, variant)
        if record.request["action"] == "eval":
            return validate_evaluation_readiness(
                experiment,
                variant,
                case=record.request["target"]["evaluation_case"],
            )
        return validate_export_readiness(experiment, variant)

    def _fallback_for_request(
        self,
        record: RunRecord,
        selected: Mapping[str, str],
    ) -> dict[str, Any]:
        variant = record.snapshot["variant"]
        if record.request["action"] == "train":
            return {
                "dataset": variant["dataset"],
                "train": variant["train"],
                "components": dict(selected),
            }
        if record.request["action"] == "eval":
            case = record.request["target"]["evaluation_case"]
            return {
                "case": evaluation_case(variant, case),
                "metrics": metric_spec(variant, case),
                "components": dict(selected),
            }
        return {"infer": variant["infer"], "components": dict(selected)}

    def _evaluation_exists(
        self,
        experiment: str,
        variant: str,
        model_id: str,
        case: str,
    ) -> bool:
        return any(
            record.request["action"] == "eval"
            and record.request["target"]["model_id"] == model_id
            and record.request["target"]["evaluation_case"] == case
            for record in self.store.scan(experiment=experiment, variant=variant)
        )

    @staticmethod
    def _validate_candidate_files(
        candidate: Path,
        returned: Any,
        *,
        require_nonempty: bool = False,
    ) -> list[str]:
        if not isinstance(returned, list) or any(
            not isinstance(value, str) for value in returned
        ):
            raise RuntimeFailure("worker file result must be a list of paths")
        if require_nonempty and not returned:
            raise RuntimeFailure("worker file result must not be empty")
        if len(returned) != len(set(returned)):
            raise RuntimeFailure("worker file result contains duplicates")
        expected: set[str] = set()
        for raw in returned:
            path = Path(raw)
            if not path.is_absolute():
                raise RuntimeFailure("worker file result must be absolute")
            try:
                relative = path.relative_to(candidate)
                resolved = path.resolve(strict=True)
                resolved.relative_to(candidate.resolve(strict=True))
            except (OSError, ValueError) as error:
                raise RuntimeFailure(
                    "worker file result is outside candidate"
                ) from error
            if path.is_symlink() or not path.is_file():
                raise RuntimeFailure("worker file result must be a regular file")
            expected.add(relative.as_posix())
        actual = set(_candidate_files(candidate))
        if actual != expected:
            raise RuntimeFailure("worker file result does not match candidate files")
        return sorted(actual, key=lambda value: value.encode("utf-8"))

    @staticmethod
    def _publish_or_verify_file(
        candidate: Path,
        final: Path,
        expected: bytes,
    ) -> None:
        if os.path.lexists(final):
            if (
                final.is_symlink()
                or not final.is_file()
                or final.read_bytes() != expected
            ):
                raise ContractError("published file disagrees with receipt")
            candidate.unlink(missing_ok=True)
            return
        if not candidate.is_file() or candidate.is_symlink():
            raise ContractError("receipt candidate file is unavailable")
        if candidate.read_bytes() != expected:
            raise ContractError("receipt candidate file changed")
        publish_file(candidate, final)

    @staticmethod
    def _publish_or_verify_directory(
        candidate: Path,
        final: Path,
        files: Sequence[Mapping[str, Any]],
    ) -> None:
        expected = {
            item["path"]: item["digest"]
            for item in files
            if isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("digest"), str)
        }
        if len(expected) != len(files):
            raise ContractError("directory receipt is invalid")
        if os.path.lexists(final):
            if final.is_symlink() or not final.is_dir():
                raise ContractError("published directory is invalid")
            actual = {
                relative: _file_digest(final / relative)
                for relative in _candidate_files(final)
            }
            if actual != expected:
                raise ContractError("published directory disagrees with receipt")
            if candidate.exists():
                shutil.rmtree(candidate)
            return
        actual = {
            relative: _file_digest(candidate / relative)
            for relative in _candidate_files(candidate)
        }
        if actual != expected:
            raise ContractError("candidate directory disagrees with receipt")
        publish_directory(candidate, final)


def _candidate_files(root: Path) -> list[str]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError("candidate directory is invalid")
    files: list[str] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            path = directory_path / name
            if path.is_symlink():
                raise ContractError("candidate contains a symlink")
        for name in filenames:
            path = directory_path / name
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise ContractError("candidate file is unavailable") from error
            if path.is_symlink() or not stat.S_ISREG(mode):
                raise ContractError("candidate contains a non-regular file")
            files.append(path.relative_to(root).as_posix())
    return sorted(files, key=lambda value: value.encode("utf-8"))


def _file_digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ContractError("artifact must be a regular file")
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


__all__ = [
    "ExecutionFailure",
    "ExecutionInterrupted",
    "LifecycleConflict",
    "RunExecution",
]
