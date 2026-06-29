# Optuna Hyperparameter Search 

Volume Segmantics supports automated hyperparameter optimization using [Optuna](https://optuna.org/), an optional add-on that searches for better training settings than manual trial-and-error typically finds. It uses Bayesian optimization via a TPE sampler, rather than searching blindly or trying every combination on a grid.

## Configuration and command line use

After installing Optuna, a new command becomes available alongside `model-train-2d` and `model-predict-2d`:

```
model-train-2d-optuna --data path/to/image/data.h5 --labels path/to/corresponding/segmentation/labels.h5 --optimization optuna_config.yaml
```
- `optuna_config.yaml`: a new YAML file defining which parameters to search over and how many trials to run.

A minimal `optuna_config.yaml` looks like this:

```yaml
study_name: "volseg_optimization"
n_trials: 30
seed: 42

search_space:
  starting_lr:
    type: "loguniform"
    low: 1e-5
    high: 1e-3

  model_type:
    type: "categorical"
    choices: ["U_Net", "U_Net_Plus_plus", "FPN"]

  encoder_name:
    type: "categorical"
    choices: ["resnet34", "resnet50", "efficientnet-b3"]
```

Each entry in `search_space` corresponds to a parameter Optuna will vary from trial to trial. Four parameter types are supported:

| Type | Use for | Example |
|---|---|---|
| `loguniform` | learning rates, anything spanning orders of magnitude | `starting_lr` |
| `uniform` | weights, multipliers | `encoder_lr_multiplier` |
| `int` | epoch counts, patience | `num_cyc_frozen` |
| `categorical` | discrete choices | `model_type`, `encoder_name`, `loss_criterion` |

Once optimization finishes, the best configuration found is written to `best_config_<study_name>.yaml` in the working directory. You can then use it for a final training run in the normal way:

```
cp best_config_volseg_optimization.yaml volseg-settings/2d_model_train_settings.yaml
model-train-2d --data path/to/image/data.h5 --labels path/to/corresponding/segmentation/labels.h5
```

### Additional command line options

| Flag | Description |
|---|---|
| `--n-trials N` | Number of trials to run, overriding `n_trials` in the Optuna config |
| `--timeout SECONDS` | Maximum total optimization time |
| `--study-name NAME` | Name for the study (also controls the output filename) |
| `--storage URL` | Database URL (e.g. `sqlite:///optuna.db`), enabling distributed optimization across multiple processes |
| `--direction {maximize,minimize}` | Optimization direction (default: `maximize`) |
| `--visualize` | Save HTML plots of the optimization to `optuna_plots/` |
| `--verbose` / `-v` | Enable debug-level logging |

### Pruning

Volume Segmantics' trainer reports its validation metric to Optuna after every epoch. Trials that are clearly underperforming compared to others at the same point are stopped early (pruned), so time is not wasted fully training configurations that were never going to be competitive. Pruning behaviour can be tuned via `n_startup_trials` and `n_warmup_steps` in the Optuna config, which control how many trials and epochs are given a guaranteed full run before pruning kicks in.

## Example configuration

A fuller example search space, covering more of the available parameters, is provided in [`optuna_config_example.yaml`](optuna_config_example.yaml).