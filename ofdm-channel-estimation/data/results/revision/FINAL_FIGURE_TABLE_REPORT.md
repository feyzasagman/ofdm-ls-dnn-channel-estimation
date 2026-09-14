# Final Figure and Table Report

Generated: 2026-09-08T09:38:30.779364+00:00

## Archived old figures/tables

- **Location:** `data/results/archive_pre_revision/`
- **Files moved:** 99 (see `archive_pre_revision/ARCHIVE_LOG.md`)
- Includes legacy MSE/NMSE/BER plots, improvement-percentage figures, seed bar charts,
  old comparison CSV/JSON/NPZ, pre-revision training logs/models, and methodology diagrams.

## New figures generated

| Figure | PNG/PDF | Source data |
| --- | --- | --- |
| `fig_mse_awgn` | `data\results\revision\final_figures\fig_mse_awgn.png` | `revision/mse_nmse/aggregate_awgn.json` |
| `fig_nmse_awgn` | `data\results\revision\final_figures\fig_nmse_awgn.png` | `revision/mse_nmse/aggregate_awgn.json` |
| `fig_ber_awgn` | `data\results\revision\final_figures\fig_ber_awgn.png` | `revision/ber/aggregate_awgn.json` |
| `fig_high_snr_awgn` | `data\results\revision\final_figures\fig_high_snr_awgn.png` | `revision/mse_nmse/aggregate_awgn.json` |
| `fig_training_awgn` | `data\results\revision\final_figures\fig_training_awgn.png` | `revision/training/history_awgn_seed42.json` |
| `fig_sample_awgn` | `data\results\revision\final_figures\fig_sample_awgn.png` | `revision/raw/awgn_ns100_seed42.npz + model + metadata` |
| `fig_mse_rayleigh` | `data\results\revision\final_figures\fig_mse_rayleigh.png` | `revision/mse_nmse/aggregate_rayleigh.json` |
| `fig_nmse_rayleigh` | `data\results\revision\final_figures\fig_nmse_rayleigh.png` | `revision/mse_nmse/aggregate_rayleigh.json` |
| `fig_ber_rayleigh` | `data\results\revision\final_figures\fig_ber_rayleigh.png` | `revision/ber/aggregate_rayleigh.json` |
| `fig_high_snr_rayleigh` | `data\results\revision\final_figures\fig_high_snr_rayleigh.png` | `revision/mse_nmse/aggregate_rayleigh.json` |
| `fig_training_rayleigh` | `data\results\revision\final_figures\fig_training_rayleigh.png` | `revision/training/history_rayleigh_seed42.json` |
| `fig_sample_rayleigh` | `data\results\revision\final_figures\fig_sample_rayleigh.png` | `revision/raw/rayleigh_ns100_seed42.npz + model + metadata` |
| `fig_mse_rician` | `data\results\revision\final_figures\fig_mse_rician.png` | `revision/mse_nmse/aggregate_rician.json` |
| `fig_nmse_rician` | `data\results\revision\final_figures\fig_nmse_rician.png` | `revision/mse_nmse/aggregate_rician.json` |
| `fig_ber_rician` | `data\results\revision\final_figures\fig_ber_rician.png` | `revision/ber/aggregate_rician.json` |
| `fig_high_snr_rician` | `data\results\revision\final_figures\fig_high_snr_rician.png` | `revision/mse_nmse/aggregate_rician.json` |
| `fig_training_rician` | `data\results\revision\final_figures\fig_training_rician.png` | `revision/training/history_rician_seed42.json` |
| `fig_sample_rician` | `data\results\revision\final_figures\fig_sample_rician.png` | `revision/raw/rician_ns100_seed42.npz + model + metadata` |
| `fig_inference_time` | `data\results\revision\final_figures\fig_inference_time.png` | `revision/complexity/complexity_*_seed*.json` |

## New tables generated

| Table | Files | Source |
| --- | --- | --- |
| Table 1 — System parameters | `final_tables/table01_*` | Code (`OFDMParams`, pilots) + `metadata/split_awgn_seed42.json` |
| Table 2 — DNN config | `final_tables/table02_*` | `model.py`, `training/history_awgn_seed42.json`, `complexity/complexity_awgn_seed42.json` |
| Tables 3–5 — MSE/NMSE | `final_tables/table03–05_*` | `mse_nmse/aggregate_{awgn,rayleigh,rician}.json` |
| Tables 6–8 — BER | `final_tables/table06–08_*` | `ber/aggregate_*.json` + `ber/ber_*_seed42.json` (total bits) |
| Table 9 — SNR gains | `final_tables/table09_*` | `snr_gain/snr_gain_*.json` |
| Table 10 — High-SNR | `final_tables/table10_*` | `mse_nmse/comparison_*_seed{42,123,999}.json` |
| Table 11 — Complexity | `final_tables/table11_*` | `complexity/complexity_*_seed*.json` |

## Zero-BER plotting treatment

- Raw BER values in tables/JSON are unchanged (including exact zeros).
- Log-scale BER figures apply `ber_for_log_plot()` with floor ε=1e-6 **for visualization only**.

## Missing data

- Target BER = 1e-3 SNR gain: **not estimable within simulated SNR range** (all channels).
- Training curves: representative **seed=42** only (not seed-averaged).

## Recommended manuscript figure order

1. System diagram (from archive: `archive_pre_revision/figures/ofdm_system_diagram.png`)
2. DNN architecture (from archive: `archive_pre_revision/figures/dnn_diagram.png`)
3. `fig_sample_awgn`, `fig_sample_rayleigh`, `fig_sample_rician`
4. `fig_mse_{awgn,rayleigh,rician}`
5. `fig_nmse_{awgn,rayleigh,rician}`
6. `fig_ber_{awgn,rayleigh,rician}`
7. `fig_high_snr_rayleigh`, `fig_high_snr_rician`
8. `fig_training_{awgn,rayleigh,rician}`
9. `fig_inference_time` (optional; Table 11 is primary)

## Replacement mapping (old → new)

| Old (archived) | New (final) |
| --- | --- |
| `figures/snr_vs_mse*.png` (linear/old) | `final_figures/fig_mse_*.png` |
| `figures/snr_vs_nmse*.png` | `final_figures/fig_nmse_*.png` |
| `figures/snr_vs_ber*.png` | `final_figures/fig_ber_*.png` |
| `figures/improvement_percentage_*.png` | **Removed** — not included in final set |
| `figures/awgn_seed_mean_std_bar.png` | **Removed** — replaced by multi-seed curves |
| `figures/example_channel_estimate_*.png` | `final_figures/fig_sample_*.png` |
| `figures/training_vs_validation_loss_*.png` | `final_figures/fig_training_*.png` |
| `metrics/comparison_per_snr*.csv/json` | `final_tables/table03–08_*.csv` |
| `metrics/improvement_percentage_*.csv` | **Removed** |
