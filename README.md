# HKDL

HKDL authors self-contained ML Variants and records immutable execution
history. Version 1.0.0 supports authoring, multi-seed training, immutable
Models, named evaluation cases with optional result artifacts, Variant-owned
export, new-Run retry, status projections, and opt-in MLflow tracking.

## Setup

From the cloned repository root:

```text
./setup.sh
source .venv/bin/activate
hkdl --help
```

`setup.sh` is the maintained environment entrypoint. It performs a locked
root sync and verifies the installed CLI.

## End-to-end example

Create an Experiment and copy the latest Template into a Variant:

```text
hkdl experiment create demo --template resnet18
hkdl variant create demo baseline
```

Train two seeds in one Training Group:

```text
hkdl run train demo baseline stability --seed 0,1
hkdl model list demo baseline
```

Evaluate all Models with a named Evaluation Case:

```text
hkdl run eval demo baseline stability default --seed all
hkdl run eval demo baseline stability daisy-only --seed 0
```

Export one exact Model:

```text
hkdl run export demo baseline model-0123456789abcdef0123456789abcdef
```

Inspect the reconstructed hierarchy:

```text
hkdl status demo baseline
hkdl status demo baseline --output json
```

Each command above creates one immutable action Run. A successful Train Run
also creates an immutable Model. Evaluation and export target Models; they do
not advance a shared pipeline Run.

## Retry

A failed, interrupted, or abandoned action is retried as a new Run:

```text
hkdl run retry demo baseline run-003
```

The parent stays sealed and the child records `retry_of`. A valid last
checkpoint may be used to continue training.

## Schema compatibility

Authored Experiment and Variant files currently use schema version 1. Inspect
one authored file through the stable migration boundary:

```text
hkdl migrate experiments/demo/experiment.yaml
hkdl migrate experiments/demo/baseline/variant.yaml
```

Version 1 is already current, so these commands validate the file and do not
rewrite it. No migration path is registered yet. Unsupported schema versions
fail without changing the target. Generated Run and Model records are immutable
and cannot be migrated in place.

## ResNet18 fixtures

`resnet18@1.0.0` includes the attributed small TF-Flowers JPEG fixture, two
evaluation cases (`default` and `daisy-only`), prediction result output, ONNX
export support, checkpoint continuation, and optional MLflow dependencies.
It performs no dataset or pretrained-weight download at runtime.

## MLflow

MLflow is opt-in through `tracker.backend: mlflow` and requires an external
`MLFLOW_TRACKING_URI`. Every local execution Run gets a distinct external Run.
Retry uses a new external identity with parent relation tags. HKDL does not
start or manage an MLflow server.

## License

HKDL is released under the MIT License. The bundled TF-Flowers fixture retains
its own attribution and CC BY 4.0 terms in its `ATTRIBUTION.md`.
