"""Read-only Variant-managed status projections."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .config import ContractError
from .runs import ModelRecord, RunRecord, RunStore
from .storage import RepositoryPaths


class Status:
    def __init__(self, repository: RepositoryPaths):
        self.store = RunStore(repository)

    def query(
        self,
        *,
        experiment: str | None = None,
        variant: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if variant is not None and experiment is None:
            raise ContractError("Variant status filter requires Experiment")
        if run_id is not None:
            if experiment is None or variant is None:
                raise ContractError("Run status filter requires Experiment and Variant")
            records = [self.store.load(experiment, variant, run_id)]
        else:
            records = self.store.scan(experiment=experiment, variant=variant)
        return self._tree(records, include_all_models=run_id is None)

    def _tree(
        self,
        records: list[RunRecord],
        *,
        include_all_models: bool,
    ) -> dict[str, Any]:
        by_variant: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
        for record in records:
            by_variant[
                (
                    record.request["experiment"],
                    record.request["variant"],
                )
            ].append(record)
        experiments: dict[str, dict[str, Any]] = {}
        for (experiment, variant), variant_runs in sorted(
            by_variant.items(),
            key=lambda item: (
                item[0][0].encode("utf-8"),
                item[0][1].encode("utf-8"),
            ),
        ):
            experiment_node = experiments.setdefault(
                experiment,
                {"name": experiment, "variants": []},
            )
            all_models = self.store.scan_models(
                experiment=experiment,
                variant=variant,
            )
            if not include_all_models:
                referenced = {
                    record.request["target"].get("model_id") for record in variant_runs
                }
                producers = {
                    record.request["run_id"]
                    for record in variant_runs
                    if record.request["action"] == "train"
                }
                all_models = [
                    model
                    for model in all_models
                    if model.document["model_id"] in referenced
                    or model.document["producer_run"] in producers
                ]
            model_by_id = {model.document["model_id"]: model for model in all_models}
            groups: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
            for model in all_models:
                self._seed_node(groups, model.document["training_group"], model)[
                    "model"
                ] = _model_summary(model)
            for record in variant_runs:
                group, seed = _run_group_seed(record, model_by_id)
                seed_node = self._seed_node(
                    groups,
                    group,
                    model_by_id.get(record.request["target"].get("model_id")),
                    seed=seed,
                )
                seed_node["runs"].append(self._run_summary(record))
            group_documents: list[dict[str, Any]] = []
            for group_name in sorted(groups, key=lambda value: value.encode("utf-8")):
                seeds = [
                    groups[group_name][seed] for seed in sorted(groups[group_name])
                ]
                for seed_node in seeds:
                    seed_node["runs"].sort(key=_run_number)
                group_documents.append(
                    {
                        "name": group_name,
                        "seeds": seeds,
                        "aggregates": _aggregates(seeds),
                    }
                )
            experiment_node["variants"].append(
                {
                    "name": variant,
                    "training_groups": group_documents,
                }
            )
        return {"experiments": list(experiments.values())}

    @staticmethod
    def _seed_node(
        groups: dict[str, dict[int, dict[str, Any]]],
        group: str,
        model: ModelRecord | None,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if model is not None:
            seed = model.document["seed"]
        if seed is None:
            raise ContractError("Run has no Training Group seed")
        return groups[group].setdefault(
            seed,
            {
                "seed": seed,
                "model": None,
                "runs": [],
            },
        )

    def _run_summary(self, record: RunRecord) -> dict[str, Any]:
        primary = None
        values: dict[str, float] = {}
        artifacts: list[str] = []
        if record.request["action"] == "eval" and record.state["status"] == "done":
            evaluation = self.store.load_evaluation(record)
            primary = dict(evaluation["primary"])
            values = dict(evaluation["values"])
            artifacts = list(evaluation["artifacts"])
        return {
            "run_id": record.request["run_id"],
            "action": record.request["action"],
            "status": record.state["status"],
            "retry_of": record.request["retry_of"],
            "model_id": record.request["target"].get("model_id"),
            "evaluation_case": record.request["target"].get("evaluation_case"),
            "primary": primary,
            "values": values,
            "artifacts": artifacts,
            "reason": record.state["reason"],
            "tracker_run_id": record.state["tracker_run_id"],
            "metric_summary": self.store.load_training_metric_summary(record),
            "updated_at": record.state["updated_at"],
        }


def render_status_tree(document: dict[str, Any]) -> str:
    experiments = document["experiments"]
    if not experiments:
        return "no runs\n"
    lines: list[str] = []
    for experiment in experiments:
        lines.append(experiment["name"])
        variants = experiment["variants"]
        for variant_index, variant in enumerate(variants):
            variant_last = variant_index == len(variants) - 1
            lines.append(f"{'`--' if variant_last else '|--'} {variant['name']}")
            variant_prefix = "    " if variant_last else "|   "
            groups = variant["training_groups"]
            for group_index, group in enumerate(groups):
                group_last = group_index == len(groups) - 1
                group_branch = "`--" if group_last else "|--"
                lines.append(f"{variant_prefix}{group_branch} {group['name']}")
                group_prefix = variant_prefix + ("    " if group_last else "|   ")
                seeds = group["seeds"]
                for seed_index, seed in enumerate(seeds):
                    seed_last = seed_index == len(seeds) - 1
                    model = (
                        f"  model={seed['model']['model_id']}"
                        if seed["model"] is not None
                        else ""
                    )
                    seed_branch = "`--" if seed_last else "|--"
                    lines.append(
                        f"{group_prefix}{seed_branch} seed={seed['seed']}{model}"
                    )
                    seed_prefix = group_prefix + ("    " if seed_last else "|   ")
                    runs = seed["runs"]
                    for run_index, run in enumerate(runs):
                        run_last = run_index == len(runs) - 1
                        branch = "`--" if run_last else "|--"
                        fields = [
                            run["action"],
                            run["run_id"],
                            run["status"],
                        ]
                        if run["retry_of"] is not None:
                            fields.append(f"retry_of={run['retry_of']}")
                        if run["evaluation_case"] is not None:
                            fields.append(f"case={run['evaluation_case']}")
                        if run["primary"] is not None:
                            fields.append(
                                f"primary={run['primary']['name']}:"
                                f"{run['primary']['value']}"
                            )
                        if run["reason"] is not None:
                            fields.append(f"reason={run['reason']}")
                        if run["tracker_run_id"] is not None:
                            fields.append(f"tracker={run['tracker_run_id']}")
                        if run["metric_summary"]:
                            summary = ",".join(
                                f"{name}:{metric['last_value']}@"
                                f"{metric['last_step']}({metric['count']})"
                                for name, metric in sorted(
                                    run["metric_summary"].items(),
                                    key=lambda item: item[0].encode("utf-8"),
                                )
                            )
                            fields.append(f"metrics={summary}")
                        fields.append(f"updated={run['updated_at']}")
                        lines.append(f"{seed_prefix}{branch} {'  '.join(fields)}")
    return "\n".join(lines) + "\n"


def _model_summary(model: ModelRecord) -> dict[str, Any]:
    return {
        "model_id": model.document["model_id"],
        "producer_run": model.document["producer_run"],
        "device": model.document["device"],
        "training_fingerprint": model.document["training_fingerprint"],
        "created_at": model.document["created_at"],
    }


def _run_group_seed(
    record: RunRecord,
    models: dict[str, ModelRecord],
) -> tuple[str, int]:
    target = record.request["target"]
    if "training_group" in target and "seed" in target:
        return target["training_group"], target["seed"]
    model_id = target["model_id"]
    model = models.get(model_id)
    if model is None:
        raise ContractError(f"Run references a missing Model: {model_id}")
    return model.document["training_group"], model.document["seed"]


def _run_number(run: dict[str, Any]) -> int:
    return int(run["run_id"].removeprefix("run-"))


def _aggregates(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    eligible = sum(seed["model"] is not None for seed in seeds)
    for seed in seeds:
        for run in seed["runs"]:
            if (
                run["action"] == "eval"
                and run["status"] == "done"
                and run["evaluation_case"] is not None
            ):
                for metric, value in run["values"].items():
                    values[(run["evaluation_case"], metric)].append(float(value))
    result: list[dict[str, Any]] = []
    for (case, metric), samples in sorted(
        values.items(),
        key=lambda item: (
            item[0][0].encode("utf-8"),
            item[0][1].encode("utf-8"),
        ),
    ):
        result.append(
            {
                "evaluation_case": case,
                "metric": metric,
                "eligible": eligible,
                "count": len(samples),
                "mean": statistics.fmean(samples),
                "sample_std": statistics.stdev(samples) if len(samples) >= 2 else None,
            }
        )
    return result


__all__ = ["Status", "render_status_tree"]
