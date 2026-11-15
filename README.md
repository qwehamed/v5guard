# V5Guard

Evaluation harness for **V5Guard**, a defense for an MQTT-style broker. The scripts run synthetic traffic against several attacker models and defense configurations, then report availability, latency, resource use, and detection metrics.

> **Note:** All results come from a simulation. Randomizing the cost parameters shows robustness to assumed costs, but it does not replace measuring them on a real broker.

## Repository contents

| File | Description |
|---|---|
| `5.v5guard_eval_v3.py` | Evaluation harness, version 3 |
| `6.v5guard_eval_v4.py` | Evaluation harness, version 4 (current). Colab-ready, no GPU, no network |
| `results_colab_4.zip` | Results from a Google Colab run |
| `results_colab_5.zip`, `results_colab_5.txt` | Results from a later Colab run |
| `results_mqttset1.zip` | Results from a run based on MQTTset |

## What v4 does

v4 builds on v3 and targets five review concerns: parameters that were only simulated, missing strong baselines, tuning on test data, weak statistics, and a weak learned detector.

- **Traffic:** generated synthetically by `run_sim` (benign clients plus attacker populations). No external dataset is needed.
- **Attack families:** `A1`, `A2`, `A3`, `A4_B1`, `A4_enf`, plus the extra variants `A2fit`, `A4_bud`, `A4_B5`.
- **Features:** `rate`, `props`, `bytes`, `queue`, `reconn`.
- **Detectors and baselines:**
  - IsolationForest (`if`, and `c10` with a 90th-percentile threshold)
  - RandomForest (`rf`) and HistGradientBoosting (`gb`), including leave-one-family-out variants
  - z-score (`z_orig`), robust z-score using median/MAD (`rz`), and a budget ratio against the benign 99th percentile (`bud`)
  - Thresholds are calibrated at `alpha = 0.01` over several consecutive-window settings.
- **Metrics:**
  - Availability: on-time legitimate delivery within a 0.2 s deadline (`avail_deadline`) and total delivered (`avail_any`)
  - Failure breakdown: blocked, throttled by byte/rate limits, rejected, buffer drops, late delivery
  - Latency (p50, p99), CPU utilization, peak memory
  - Detection: true positive rate, false positive rate, time to detect
- **Statistics:** 95% t-based confidence intervals, paired differences, Holm-corrected p-values, and TOST equivalence testing at ±5%.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

No GPU is required. The script also runs on Google Colab.

## Usage

```bash
# Quick smoke test (2 seeds, shorter run, reduced grids)
python 6.v5guard_eval_v4.py --quick --out results_quick

# Full run (default 10 seeds)
python 6.v5guard_eval_v4.py --seeds 10 --out results

# Recommended for a final run (at least 20 seeds)
python 6.v5guard_eval_v4.py --seeds 20 --workers 8 --out results_final

# Run only some experiments
python 6.v5guard_eval_v4.py --exps matrix,base --out results_partial

# Override simulator costs with values measured on a real broker
python 6.v5guard_eval_v4.py --costs my_costs.json --out results_measured
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--seeds` | `10` (`2` with `--quick`) | Number of random worlds |
| `--out` | `results` | Output directory |
| `--workers` | CPU count | Number of parallel worker processes |
| `--quick` | off | Fewer seeds, shorter run, reduced grids |
| `--exps` | `matrix,sweep,cap,confirm,sens,base` | Comma-separated list of experiments to run |
| `--parse-mode` | `partial` | Parsing cost model: `full`, `partial`, or `header` |
| `--costs` | none | JSON file with measured parameters that override the simulator defaults |
| `--fixed-workload` | off | Disables per-seed randomization of the workload |

Each run writes a `costs_template.json` to the output directory. You can fill it in with measured values and pass it back through `--costs`.

## Output

Each run writes CSV tables and PNG figures to the `--out` directory.

**Tables**

- `params.csv`, `seed_params.csv`, `costs_template.json`
- `table_matrix.csv`, `table_paired_vs_B0.csv`, `table_benign_overhead.csv`
- `table_baselines.csv`, `table_tuning_heldout.csv`
- `table_cap_sweep.csv`, `table_cap_verdict.csv`, `table_confirm_sweep.csv`
- `table_ablation_loadaware.csv`
- `table_sensitivity.csv`, `table_sensitivity_ranking.csv`
- `table_bound_validation.csv`, `table_sweep.csv`

**Figures:** PNG plots for the sweeps, cap sweep, confirmation sweep, sensitivity analysis, and bound validation.

## Key default parameters

| Parameter | Default | Meaning |
|---|---|---|
| `T` | 60 s | Simulation length |
| `dt` | 0.1 s | Time step |
| `n_legit` | 50 | Number of legitimate clients |
| `cap` | 5000 | Capacity |
| `deadline` | 0.2 s | On-time delivery deadline |
| `alpha` | 0.01 | Detection threshold calibration level |
| `consec` | 3 | Consecutive windows required |
| `packet_cap` | 8192 | Packet size limit |
| `ip_conn_limit` | 3 | Connection limit per IP |
| `cost_sigma` | 0.3 | Spread of randomized operation costs |
| `equiv_margin_pct` | 5.0 | TOST equivalence margin (%) |

Attacker population sizes: `B_matrix = 40`, `B_hi = 320`, `B_syb = 1000`.

## Limitations

- The results are simulated. Real-broker validation is still needed.
- The default seed count is small. Use at least 20 seeds for a final run.
- Several ranges and operationalizations (for example the load-aware threshold and the B6 retry cost) are the author's own design choices.

## License

No license has been specified yet. Add a `LICENSE` file to define the terms of use.

## Citation

If you use this work, please cite the repository (add a paper reference here if one is available).
