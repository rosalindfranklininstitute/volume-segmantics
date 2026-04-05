# Test Harness

All harness-driven tests live in `tests/scripts/`. The framework is config-driven:
a YAML file defines a sequence of steps (train, predict, slicer, pytest)
and the harness executes them in order, passing artifacts between steps.

## Quick start

```bash
# List available named configs
python tests/scripts/run_smoke.py --list

# Run a named config, e.g.
python tests/scripts/run_smoke.py basic
python tests/scripts/run_smoke.py losses
python tests/scripts/run_smoke.py multitask
python tests/scripts/run_smoke.py 2.5d
python tests/scripts/run_smoke.py semisup

# Dry-run (print resolved commands, skip execution)
python tests/scripts/run_smoke.py basic --dry-run

# Run any config by path
python tests/scripts/run_smoke.py --config tests/scripts/configs/my_custom.yaml
```

You can also invoke the harness directly:

```bash
python tests/scripts/real_data_harness.py --config tests/scripts/configs/basic_smoke_tests.yaml
```

## Framework

- `tests/scripts/real_data_harness.py` — core harness engine
- `tests/scripts/run_smoke.py` — unified CLI with named shortcuts

### Supported step types
 
 `train` - Run `train_2d_model` 
 `predict` - Run `predict_2d_model` with a registered model 
 `unlabeled_slicer` - Slice unlabeled volumes into 2D images 
 `pytest` - Run targeted pytest checks 

### Token substitution

Tokens in step `args` are resolved automatically:

- `${path:image}`, `${path:label}`, `${path:unlabeled}`, `${path:run_dir}`
- `${path:unlabeled_all}` — expands to all configured unlabeled volumes
- `${path:task2}`, `${path:task3}` — multitask label volumes
- `${model:<model_key>}` — path to a model registered by a previous train step
- `${artifact:<artifact_key>}` — path to an artifact from a previous step


## Customizing data paths

All configs default to `training_data/vessels_256cube_*.h5`. To use your own data, edit the `paths:` section:

```yaml
paths:
  image: path/to/your/image.h5       # or .tiff, .mrc
  label: path/to/your/labels.h5
  task2: path/to/boundary_labels.h5  # for multitask tests
  unlabeled:
    - path/to/unlabeled_vol1.h5
```

## Dry-run mode

In dry-run mode, commands and token resolution are printed, artifacts/models are
registered logically, and execution/assertions are skipped:

```bash
python tests/scripts/run_smoke.py losses --dry-run
```
