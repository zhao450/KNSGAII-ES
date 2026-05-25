## Run One Instance from the Command Line

Example:

```bash
python route_c_delayed_periodic_auto_trials.py \
  --instance dataset/comparison_experiment/small/comparison_experiment_small_001.txt \
  --results-root results \
  --population 100 \
  --seed 42 \
  --workers 1
```

The result folder will be created under:

```text
results/comparison_experiment/small/comparison_experiment_small_001/
```

Typical output files are:

```text
pareto_front.csv
pareto_objectives.csv
history.csv
summary.json
```
