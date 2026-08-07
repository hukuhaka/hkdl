# HKDL

HKDL authors self-contained ML Variants and records immutable execution
history. Version 1.1.0 supports authoring, multi-seed training, immutable
Models, named evaluation cases with optional result artifacts, Variant-owned
export, new-Run retry, brief and full status projections, live local metric
following, shell completion, and opt-in MLflow tracking.

## Setup

From the cloned repository root:

```text
./setup.sh
source ./activate.sh
hkdl --version
hkdl --help
```

`setup.sh` is the maintained environment entrypoint. It performs a locked
root sync and verifies the installed CLI. `activate.sh` activates that root
environment and registers completion in the current Bash or Zsh session.

## Shell completion

The recommended `source ./activate.sh` flow registers completion automatically.
To register it manually after activating the environment, run only the line for
your shell:

```text
eval "$(hkdl completion zsh)"
eval "$(hkdl completion bash)"
```

HKDL completes commands, options, Templates, Experiments, Variants, Runs,
Models, Training Groups, Evaluation Cases, seeds, and devices from the current
repository. Static command and option completion also works outside a
repository. Neither setup nor activation modifies shell startup files or
installs completion globally; source `activate.sh` once in each new shell.

## Update

From a public source checkout root:

```text
hkdl update
```

The command shows the installed and available versions, lists what will change
and what will be preserved, and asks before updating. It fast-forwards the
public source and reinstalls HKDL. Experiments, outputs, existing Variants, and
authored schemas are not changed. Use `hkdl update --yes` only when confirmation
has already been provided.

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
hkdl status demo baseline --full
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

`resnet18@1.0.1` includes the attributed small TF-Flowers JPEG fixture, two
evaluation cases (`default` and `daisy-only`), prediction result output, ONNX
export support, checkpoint continuation, and optional MLflow dependencies.
It records loss and per-batch wall time and performs no dataset or
pretrained-weight download at runtime. The immutable `1.0.0` remains available.

## YOLO26n object detection

`yolo26n@1.0.1` includes a deterministic synthetic shapes dataset with eight
training images, four validation images, and two classes (`circle` and
`rectangle`). It trains the architecture from scratch for ten epochs and
records loss and per-epoch wall time. It performs no dataset or
pretrained-weight download at runtime. The immutable `1.0.0` remains available.

```text
hkdl experiment create detection --template yolo26n
hkdl variant create detection baseline
hkdl run train detection baseline smoke --seed 0 -d cpu
hkdl run eval detection baseline smoke default --seed 0
hkdl model list detection baseline
```

The Template reports finite detection metrics, writes deterministic prediction
JSON, and exports a fixed-shape ONNX model together with its AGPL license.
Ultralytics network checks, automatic dependency installation, and third-party
tracking integrations are disabled. HKDL remains the only tracking owner.

## Tracking

New bundled `1.0.1` Variants default to `tracker.backend: local`. Training
scalars, including `train.batch_seconds` for ResNet18 and
`train.epoch_seconds` for YOLO26n, are stored with the Run and can be inspected
with:

```text
.venv/bin/hkdl run metrics <experiment> <variant> <run-id>
```

Follow an active local-tracked Train Run from another terminal:

```text
.venv/bin/hkdl run metrics <experiment> <variant> <run-id> --follow
```

Follow prints existing metric rows, then newly completed rows until the Run
becomes terminal. It remains a read-only local view: JSON streaming, progress
percentages, ETA, and MLflow history polling are not provided.

Use `tracker.backend: none` to disable tracking, `mlflow` for MLflow only, or
`[local, mlflow]` for both. MLflow requires an external `MLFLOW_TRACKING_URI`.
Every local execution Run gets a distinct external Run. Retry uses a new
external identity with parent relation tags. HKDL does not start or manage an
MLflow server.

## License

HKDL Core and files without a more specific notice are released under the MIT
License. The bundled TF-Flowers fixture retains its own attribution and CC BY
4.0 terms in its `ATTRIBUTION.md`.

The `yolo26n` Template, Variants derived from it, and model artifacts produced
with Ultralytics are licensed under GNU AGPL version 3 or later. Its source
bundle includes the complete license and scope notice. Commercial users who do
not wish to comply with the AGPL requirements must obtain an Ultralytics
Enterprise License.
