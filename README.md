# OFDM Channel Estimation: LS, Simplified-MMSE, LMMSE-flat, and LS+DNN

This repository provides an independently implemented and reproducible SISO-OFDM simulation framework for comparing LS, Simplified-MMSE, LMMSE-flat, and residual LS+DNN channel estimation methods under AWGN, single-tap Rayleigh flat fading, and single-tap Rician flat fading.

| Folder | Description |
| --- | --- |
| **`ofdm-channel-estimation/`** | Main Python package: simulation, training, evaluation, and revision manuscript outputs |
| `MatLab_Codes/` | Legacy MATLAB reference materials not used by the main revision pipeline |
| `Python_Codes/` | Legacy Python/Keras reference materials not used by the main revision pipeline |

Full documentation, system parameters, metrics, and limitations are in [`ofdm-channel-estimation/README.md`](ofdm-channel-estimation/README.md).

## Highlights

- **SISO-OFDM** with QPSK and comb pilots (64 subcarriers, 16 pilots)
- **Estimators:** LS, Simplified-MMSE (Wiener-shrinkage), LMMSE-flat, residual LS+DNN
- **Channels:** AWGN reference, single-tap Rayleigh flat fading, single-tap Rician flat fading (\(K = 6\) dB) — no frequency-selective multipath
- **Metrics:** MSE, NMSE, end-to-end BER, SNR gain at target BER, high-SNR behavior, inference complexity
- **Reproducibility:** SNR-stratified splits, seeds 42 / 123 / 999, exported JSON/CSV under `data/results/revision/`

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

### Major-revision experiment pipeline

```bash
cd ofdm-channel-estimation

# Regenerate datasets, models, and revision metrics (optional flags: --skip-generate, --skip-train)
python scripts/20_run_revision_pipeline.py

# Build final manuscript figures and tables
python scripts/21_build_final_figures_tables.py
```

Legacy scripts `01`–`10` under `ofdm-channel-estimation/scripts/` target an older `data/raw` layout; new work should use scripts **20** and **21**.

## Repository layout (main package)

```
ofdm-channel-estimation/
├── scripts/          # Pipelines 20–21 (revision) and legacy 01–10
├── src/              # channels, core, dataset, dnn, estimators, revision
├── tests/
├── data/results/revision/   # BER, MSE/NMSE, SNR gain, final_figures, final_tables, …
└── requirements.txt
```

Generated raw NPZ, processed tensors, and trained `.keras` models are gitignored and reproduced by the revision pipeline.

## Related work

Gizzini, F., et al., *“Enhancing Least Square Channel Estimation Using Deep Learning,”* IEEE VTC2020-Spring, 2020.  
DOI: [10.1109/VTC2020-Spring48590.2020.9128890](https://doi.org/10.1109/VTC2020-Spring48590.2020.9128890)

No source code from Gizzini et al. or any other cited study was reused or adapted in this repository. The simulation, data-generation, training, and evaluation pipeline was independently implemented.

The `MatLab_Codes/` and `Python_Codes/` folders are legacy reference materials and are not imported or called by the `ofdm-channel-estimation/` revision pipeline.

## Citation

If you use this repository, please cite the associated manuscript once it is published.

For the LS+DNN line of work in the literature, you may also cite Gizzini et al. (VTC 2020) as above.

## Authors / license

Author names and affiliations are given in the associated manuscript. This repository does not include a separate `AUTHORS` file.

No `LICENSE` file is currently present at the repository root; do not assume a specific open-source license unless one is added explicitly.
