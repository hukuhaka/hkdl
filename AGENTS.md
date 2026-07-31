# HKDL User Guide for Coding Agents

## Default role

This is a runnable HKDL source checkout.

Unless the user explicitly asks to develop HKDL itself, help them operate
experiments through the HKDL CLI. Do not modify HKDL Core or the bundled
Template catalog as part of normal experiment work.

## Start

- Work from the repository root.
- If `.venv/bin/hkdl` is unavailable, run `./setup.sh`.
- Invoke the CLI as `.venv/bin/hkdl`; shell activation is not required.
- Inspect existing state before creating anything:

  ```text
  .venv/bin/hkdl template list
  .venv/bin/hkdl experiment list
  .venv/bin/hkdl status
  ```

- Use `<command> --help` when the requested operation is not covered below.
  Do not invent flags, addresses, or filesystem layouts.

## Update HKDL

Run `.venv/bin/hkdl update` only when the user explicitly asks to update this
public source checkout. Report the current and available versions and let the
command request confirmation. Use `--yes` only after the user has already
approved the shown update.

Do not reset, stash, discard, or merge local changes to make an update pass.
If the command reports a dirty checkout, branch mismatch, divergence, or major
version transition, stop and report it. If source update succeeds but setup
fails, preserve the updated checkout and follow the reported `./setup.sh`
recovery instruction.

## Normal workflow

Create an Experiment and Variant:

```text
.venv/bin/hkdl experiment create <experiment> -t <template>
.venv/bin/hkdl variant create <experiment> <variant>
```

The latest numeric Template version is used unless the user requests an exact
version.

After changing authored Variant configuration, validate it:

```text
.venv/bin/hkdl variant check <experiment> <variant>
```

Train one or more seeds:

```text
.venv/bin/hkdl run train <experiment> <variant> <group> -s <seeds>
```

If the user does not specify execution options, retain the CLI defaults:
seed `0` and device `auto`.

Inspect the resulting state and Models:

```text
.venv/bin/hkdl status <experiment> <variant>
.venv/bin/hkdl run metrics <experiment> <variant> <run-id>
.venv/bin/hkdl model list <experiment> <variant>
```

New ResNet18 Variants record training metrics locally by default. Do not enable
MLflow unless the user requests it and provides an external tracking service.

Evaluate an existing Evaluation Case:

```text
.venv/bin/hkdl run eval <experiment> <variant> <group> <case> -s <seed|all>
```

Export one exact Model ID:

```text
.venv/bin/hkdl run export <experiment> <variant> <model-id>
```

Retry a failed, interrupted, or abandoned Run as a new Run:

```text
.venv/bin/hkdl run retry <experiment> <variant> <run-id>
```

## Ownership and safety

- `experiments/` contains authored Experiment and Variant inputs.
- Modify authored YAML or Variant source only when required by the user's
  experiment request.
- Preserve `schema_version`, path identity, and Template provenance fields.
- Run `variant check` after editing a Variant.
- `outputs/` contains generated immutable Runs and Models. Never edit, move,
  normalize, or delete generated records.
- Never work around a contract error by modifying generated files.
- Use `run retry` for a stopped Run; do not reuse or overwrite its Run ID.
- Use `migrate` only with an authored `experiment.yaml` or `variant.yaml`.
- Keep MLflow credentials and tracking URIs outside authored and generated
  files.
- Ask before deleting authored work, starting an unexpectedly expensive run,
  or enabling an external tracking service.
- After an operation, report the created Run or Model IDs and the resulting
  status.
