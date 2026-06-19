# Experiment 1 Runner

This folder-level script runs GapPTR for Experiment 1 from the saved simulation folders.

## Expected project layout

Place the shared GapPTR utility under `Utility/`, and place this runner under the Experiment 1 folder:

```text
ProjectRoot/
  Utility/
    gapptr_utility.py
  Experiment1/
    run_exp1_all_scenarios.py
    E1S15000/
    E1S110000/
    E1S115000/
    ...
    E1S25000/
    E1S210000/
    E1S215000/
    ...
```

The folder name convention is:

```text
E1S1{n}   # Experiment 1, Scenario 1, network size n
E1S2{n}   # Experiment 1, Scenario 2, network size n
```

For example:

```text
E1S15000    = Experiment 1, Scenario 1, n = 5000
E1S110000   = Experiment 1, Scenario 1, n = 10000
E1S25000    = Experiment 1, Scenario 2, n = 5000
E1S210000   = Experiment 1, Scenario 2, n = 10000
```

Each folder should contain replication folders such as:

```text
rep000/
rep001/
rep002/
rep003/
rep004/
```

The current default is `--n-rep 5`, so the script uses replications `0,1,2,3,4`.

## What the script runs

For each scenario and each network size, the script runs:

1. `NonPrivate`: ordinary spectral clustering from the saved eigenvectors.
2. `oracle`: GapPTR using oracle `theta0`, i.e. `theta0_mode="oracle"`.
3. `tilde`: GapPTR using noisy/estimated `theta0`, i.e. `theta0_mode="use_tilde"` and `eps1=0.2`.

The default release privacy budgets are:

```python
eps_list = [0.5, 0.8, 1.0, 2.0]
```

## Basic command

From the `Experiment1/` folder, run:

```bash
python run_exp1_all_scenarios.py
```

This searches for both Scenario 1 and Scenario 2 folders under the current directory and saves results to:

```text
results_exp1/
```

## Run only Scenario 1

```bash
python run_exp1_all_scenarios.py \
  --scenarios 1 \
  --n-rep 5
```

## Run both scenarios but only the noisy-theta0 case

```bash
python run_exp1_all_scenarios.py \
  --scenarios 1 2 \
  --cases tilde \
  --n-rep 5
```

## Main outputs

The script saves the following files.

### Per-scenario and per-case outputs

```text
summary_exp1_s1_oracle.csv
summary_exp1_s1_tilde.csv
summary_exp1_s2_oracle.csv
summary_exp1_s2_tilde.csv

long_exp1_s1_oracle.csv
long_exp1_s1_tilde.csv
long_exp1_s2_oracle.csv
long_exp1_s2_tilde.csv
```

### Combined outputs

```text
summary_exp1_all_scenarios_cases.csv
long_exp1_all_scenarios_cases.csv
summary_exp1_all_nonprivate.csv
```

### Plots

```text
plot_exp1_s1_oracle.png
plot_exp1_s1_tilde.png
plot_exp1_s1_all_cases.png

plot_exp1_s2_oracle.png
plot_exp1_s2_tilde.png
plot_exp1_s2_all_cases.png
```
