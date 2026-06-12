# Enhancing Least Square Channel Estimation Using Deep Learning

Python reimplementation and extension of the LS-DNN channel estimator from the IEEE VTC 2020 paper [*Enhancing Least Square Channel Estimation Using Deep Learning*](https://ieeexplore.ieee.org/document/9128876).

This repository contains:

| Folder | Description |
|--------|-------------|
| **`ofdm-channel-estimation/`** | Modern, end-to-end **Python** pipeline (main project) |
| `MatLab_Codes/` | Original MATLAB OFDM simulation from the paper authors |
| `Python_Codes/` | Original Keras training/testing scripts from the paper authors |

> The `ofdm-channel-estimation` module is a clean-room Python rewrite with modular source code, pytest integration tests, multi-channel support (AWGN / Rayleigh / Rician), and reproducible experiment scripts.

## Highlights

- **OFDM modem** with QPSK data and comb pilots
- **Classical estimators**: Least Squares (LS) and MMSE
- **Deep learning**: fully-connected DNN that refines LS estimates (residual learning)
- **Channel models**: AWGN, flat Rayleigh fading, flat Rician fading
- **Benchmarking**: MSE / NMSE comparison across SNR (−10 … 30 dB)
- **Reproducibility**: fixed seeds, multi-seed aggregation, metric export (CSV / JSON)

## Quick start

```bash
cd ofdm-channel-estimation
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pytest tests/ -v
```

### Full experiment pipeline (example: Rayleigh)

```bash
# 1. Generate raw + processed datasets
python scripts/01_generate_data.py --channel-type rayleigh --samples-per-snr 100 --seed 42

# 2. Train DNN
python scripts/02_train_dnn.py --processed-path data/processed/rayleigh_ns100_seed42_residual_processed.npz

# 3. Compare LS vs MMSE vs LS+DNN
python scripts/03_run_comparison.py ^
  --raw-path data/raw/rayleigh_ns100_seed42.npz ^
  --processed-path data/processed/rayleigh_ns100_seed42_residual_processed.npz ^
  --model-path data/results/logs/rayleigh_ns100_seed42_residual_processed_dnn.keras ^
  --tag rayleigh

# 4. Plot SNR curves
python scripts/04_plot_results.py --input-csv data/results/metrics/comparison_per_snr_rayleigh.csv

# 5–10. Additional analysis plots (overall bars, seed stats, training history, etc.)
python scripts/06_plot_overall_comparison.py
python scripts/10_plot_improvement_percentage.py
```

Repeat with `--channel-type awgn` or `--channel-type rician` for other channel models.

## Sample results (seed = 42, 100 samples/SNR)

| Channel  | LS MSE | MMSE MSE | LS+DNN MSE |
|----------|--------|----------|------------|
| AWGN     | 1.38   | 1.35     | **0.13**   |
| Rayleigh | 1.39   | 1.35     | **0.25**   |
| Rician   | 1.31   | 1.28     | **0.19**   |

Full per-SNR metrics are stored under `ofdm-channel-estimation/data/results/metrics/`.

## Project structure

```
ofdm-channel-estimation/
├── scripts/          # Numbered experiment pipeline (01–10)
├── src/
│   ├── core/         # OFDM params, modem, pilots, metrics
│   ├── channels/     # AWGN, Rayleigh, Rician
│   ├── estimators/   # LS, MMSE
│   ├── dataset/      # Data generation & preprocessing
│   └── dnn/          # Model, training, evaluation
├── tests/            # Integration tests
├── data/
│   ├── raw/          # Generated NPZ (gitignored)
│   ├── processed/    # DNN-ready NPZ (gitignored)
│   └── results/      # Metrics, logs, figures
└── requirements.txt
```

## Citation

If you use the original method, please cite the paper:

```bibtex
@inproceedings{el2020enhancing,
  title={Enhancing Least Square Channel Estimation Using Deep Learning},
  booktitle={2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring)},
  year={2020},
  organization={IEEE}
}
```

## License

The `ofdm-channel-estimation` Python code is released under the [MIT License](LICENSE).

The `MatLab_Codes/` and `Python_Codes/` directories contain the original implementation accompanying the IEEE paper. Refer to the paper and original authors for usage terms.

## Acknowledgements

Based on the LS-DNN channel estimation approach proposed in the IEEE VTC 2020 paper. Original MATLAB and Keras code is included for reference.
