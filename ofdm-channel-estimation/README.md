# OFDM Channel Estimation: LS, Simplified-MMSE, LMMSE and LS+DNN

Reproducible **SISO-OFDM** simulation and benchmarking of **pilot-based channel estimation** under **QPSK**, comparing classical estimators and a **deep-learning-assisted residual LS+DNN** refinement in **AWGN**, **single-tap Rayleigh flat fading**, and **single-tap Rician flat fading** scenarios.

The main contribution of this repository is not a new neural architecture proposal, but a **shared, transparent, and reproducible experimental framework** in which **LS**, **Simplified-MMSE (Wiener-shrinkage)**, **LMMSE-flat**, and **LS+DNN** are evaluated under identical data splits, metrics, and reporting conventions.

---

## Project Overview

Four channel estimators are implemented and compared end-to-end:

| Estimator | Role |
| --- | --- |
| **LS** | Least-squares estimate from comb pilots, with interpolation to all subcarriers |
| **Simplified-MMSE** | Per-pilot Wiener shrinkage + interpolation (not full covariance LMMSE) |
| **LMMSE-flat** | Joint scalar LMMSE under single-tap flat-fading assumptions |
| **LS+DNN** | Residual refinement of the LS estimate |

**LS+DNN (residual form):**

\[
\hat{H}_{\mathrm{LS+DNN}} = \hat{H}_{\mathrm{LS}} + f_\theta(\hat{H}_{\mathrm{LS}})
\]

The network predicts a complex residual (stored as 128 real-valued outputs); at inference, the residual is added back to the LS channel estimate. Training targets are built in residual mode in `src/dataset/preprocess.py`.

---

## System Configuration

Parameters below match `src/core/ofdm_params.py`, `src/core/pilots.py`, and the revision dataset generator defaults (`src/dataset/generate_dataset.py`, `scripts/20_run_revision_pipeline.py`).

| Parameter | Value |
| --- | --- |
| Subcarriers \(N\) | 64 |
| Cyclic prefix length | 16 |
| Modulation | QPSK (`modulation_order = 4`) |
| Pilot spacing | 4 subcarriers |
| Pilot indices | 0, 4, 8, …, 60 |
| Number of pilots | 16 |
| Data subcarriers | 48 |
| Pilot symbol | \(1 + 0j\) |
| SNR range | \(-10\) to \(30\) dB, step 5 dB |
| Samples per SNR | 100 |
| Random seeds (experiments) | 42, 123, 999 |
| Rician \(K\)-factor | 6 dB (`src/channels/rician.py`) |

---

## Channel Models

| Scenario | Implementation |
| --- | --- |
| **AWGN** | Unit-gain reference channel with AWGN (`src/channels/awgn.py`) |
| **Rayleigh** | Single-tap Rayleigh **flat** fading (`src/channels/rayleigh.py`) |
| **Rician** | Single-tap Rician **flat** fading, \(K = 6\) dB (`src/channels/rician.py`) |

**Not used:** frequency-selective multipath, power delay profiles (PDP), or multi-tap TDL models. All fading scenarios are **single-tap flat** (constant \(|H|\) across subcarriers for each OFDM symbol).

---

## Dataset and Reproducibility

- **Split strategy:** SNR-stratified train / validation / test split (`src/core/split.py`, `split_strategy: "snr_stratified"`).
- **Per SNR (100 samples):** 72 train / 8 validation / 20 test (ratios `test_ratio=0.2`, `val_ratio=0.1` of the remaining pool).
- **Per channel scenario (9 SNR points):** 648 train / 72 validation / 180 test total.
- Each generated sample uses an **independent channel and noise realization** (new draw per sample in `generate_raw_dataset`).
- **Normalization** for DNN inputs/targets: mean and standard deviation are **fit on training data only**, then applied to validation and test (`preprocess_for_dnn`, `normalize=True`).
- **Seeds:** `set_global_seed` (`src/core/seed.py`) fixes Python, NumPy, and TensorFlow seeds; experiments use **42, 123, 999** (`DEFAULT_SEEDS` in `src/core/split.py`).

---

## LS+DNN Architecture

Verified from `src/dnn/model.py`, `src/dnn/train.py`, and revision complexity tables (`data/results/revision/final_tables/table02_dnn_configuration.md`).

| Item | Setting |
| --- | --- |
| Input dimension | 128 (real/imag parts of LS estimate, 64 subcarriers) |
| Hidden layers | Dense 256 → ReLU → Dense 256 → ReLU |
| Output dimension | 128 (residual, real/imag) |
| Learning paradigm | Residual learning around LS |
| Trainable parameters | 131,712 |
| Optimizer | Adam |
| Learning rate | \(10^{-3}\) |
| Loss | MSE |
| Batch size | 64 |
| Max epochs | 60 (revision pipeline default) |
| Early stopping | Patience 8 on `val_loss`, best weights restored |
| Model scope | **One DNN per channel scenario** (AWGN / Rayleigh / Rician), trained across all SNRs in that scenario |
| Not used | Separate DNN per SNR |

---

## Evaluation Metrics

| Metric | Description |
| --- | --- |
| **MSE** | Mean squared error of complex channel estimate vs. true channel |
| **NMSE** | Normalized MSE (\(\mathrm{MSE} / \mathbb{E}[|H|^2]\) style reporting in aggregates) |
| **BER** | End-to-end QPSK BER after ZF equalization on data subcarriers (`src/core/ber_evaluation.py`) |
| **SNR gain at target BER** | LS vs. LS+DNN horizontal gap at fixed BER (`src/core/snr_gain.py`, Table 9) |
| **High-SNR error floor** | Analysis in 15–30 dB band (`src/core/high_snr_analysis.py`) |
| **Complexity** | Per-estimator inference time and parameter count (`src/core/complexity.py`, Table 11) |

**BER test budget (per seed, per SNR, per channel):**

- 48 data subcarriers × 2 bits/symbol (QPSK) = **96 bits** per test sample  
- 20 test samples per SNR → **1,920 bits** per SNR per seed  
- Three seeds → **5,760 bits** per SNR per channel when aggregating across seeds (see per-seed BER JSON under `data/results/revision/ber/`).

---

## Important Scientific Findings

Results summarized here come from the committed revision aggregates (`data/results/revision/mse_nmse/`, `ber/`, `snr_gain/`). They are **scenario-dependent**; LS+DNN should **not** be described as uniformly superior.

- **Low-to-moderate SNR:** LS+DNN often improves channel estimation and BER relative to LS alone.
- **High SNR:** An **error-floor** effect can appear for LS+DNN (see high-SNR figures and Table 10).
- **Rayleigh at 30 dB (aggregated MSE, seeds 42/123/999):** LS+DNN mean MSE is about **8.6×** higher than LS (\(\approx 5.89\times 10^{-3}\) vs. \(\approx 6.83\times 10^{-4}\)).
- **Rician at 30 dB:** LS+DNN mean MSE is about **2.76×** higher than LS (\(\approx 1.87\times 10^{-3}\) vs. \(\approx 6.79\times 10^{-4}\)).
- **LMMSE-flat** is a strong **model-based** baseline under the implemented single-tap flat-channel assumptions.
- **Simplified-MMSE** tends to approach LS at high SNR in the reported curves.

**SNR gain at target BER (LS+DNN vs. LS, aggregated curves — Table 9):**

| Channel | Target BER | SNR gain (dB) |
| --- | --- | --- |
| AWGN | \(10^{-1}\) | 1.78 |
| AWGN | \(10^{-2}\) | 1.46 |
| Rayleigh | \(10^{-1}\) | 1.68 |
| Rayleigh | \(10^{-2}\) | 0.90 |
| Rician | \(10^{-1}\) | 1.56 |
| Rician | \(10^{-2}\) | 1.31 |

At **target BER \(= 10^{-3}\)**, a reliable SNR gain is **not estimable** within the simulated SNR range (\(-10\) to \(30\) dB) for any channel (Table 9).

---

## Repository Structure

```
ofdm-channel-estimation/
├── scripts/                 # Experiment and figure scripts (01–10 legacy; 20–21 revision; 25–30 figures)
├── src/
│   ├── channels/            # AWGN, Rayleigh, Rician flat fading
│   ├── core/                # OFDM modem, pilots, split, BER, SNR gain, complexity
│   ├── dataset/             # Raw dataset generation and DNN preprocessing
│   ├── dnn/                 # Model definition and training
│   ├── estimators/          # LS, Simplified-MMSE, LMMSE-flat
│   ├── revision/            # Revision aggregates, comparison, plotting helpers
│   ├── experiments/         # Additional experiment helpers
│   └── utils/
├── tests/
│   ├── test_estimators.py
│   └── test_revision.py
├── data/
│   └── results/
│       └── revision/        # Major-revision outputs (see below)
│           ├── ber/
│           ├── mse_nmse/
│           ├── high_snr/
│           ├── complexity/
│           ├── snr_gain/
│           ├── training/
│           ├── metadata/
│           ├── final_figures/
│           └── final_tables/
└── requirements.txt
```

**Generated artifacts (not committed to Git):**

- `data/raw/`, `data/processed/` (legacy layout)
- `data/results/revision/raw/`, `processed/`, `models/` (revision NPZ and `.keras` checkpoints)

These are listed in the repository `.gitignore` and can be regenerated with the revision pipeline.

---

## How to Run

### Environment

```bash
cd ofdm-channel-estimation
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Requirements:** Python 3.10+, TensorFlow 2.12+, NumPy, SciPy, Matplotlib, scikit-learn, pytest (see `requirements.txt`).

### Tests

```bash
pytest tests/ -v
# or
pytest tests/test_revision.py -v
```

### Major revision pipeline (recommended)

From `ofdm-channel-estimation/`:

```bash
# Full regeneration: datasets, training, metrics, plots (all channels, seeds 42/123/999)
python scripts/20_run_revision_pipeline.py

# Archive legacy figures/tables and build manuscript final_figures / final_tables
python scripts/21_build_final_figures_tables.py
```

Useful flags for `20_run_revision_pipeline.py`:

- `--skip-generate` — reuse existing raw NPZ under `data/results/revision/raw/`
- `--skip-train` — reuse existing trained models under `data/results/revision/models/`
- `--channels awgn rayleigh rician` and `--seeds 42 123 999` (defaults)

### Legacy single-channel scripts

Scripts `01_generate_data.py` through `10_plot_improvement_percentage.py` support an older `data/raw` layout. The **revision manuscript results** are produced via scripts **20** and **21**.

---

## Revision Outputs

Under `data/results/revision/` (after running script 20):

| Folder | Contents |
| --- | --- |
| `ber/` | Per-seed BER JSON + `aggregate_{channel}.json` |
| `mse_nmse/` | Per-seed comparison JSON + aggregated MSE/NMSE |
| `high_snr/` | High-SNR error-floor summaries |
| `complexity/` | Inference timing and parameter counts |
| `snr_gain/` | Target-BER SNR gain estimates |
| `training/` | Training history JSON per channel/seed |
| `metadata/` | Split reports, sample-channel figure metadata |
| `figures/` | Intermediate revision plots |

After script 21:

| Folder | Contents |
| --- | --- |
| `final_figures/` | Manuscript-ready PNG/PDF/SVG (MSE, NMSE, BER, samples, training, etc.) |
| `final_tables/` | Tables 01–11 (CSV + Markdown) |

---

## Computational Complexity

Classical estimators have **no trainable parameters**. LS+DNN: **131,712** trainable parameters (Table 11).

**Mean inference time** (revision Table 11, one reference environment — **hardware/software dependent**):

| Estimator | Mean time | Notes |
| --- | --- | --- |
| LS | 0.021 ms | \(\approx 2.12 \times 10^{-5}\) s |
| Simplified-MMSE | 0.031 ms | \(\approx 3.07 \times 10^{-5}\) s |
| LMMSE-flat | 0.009 ms | \(\approx 9.02 \times 10^{-6}\) s |
| LS+DNN | 62.1 ms | \(\approx 6.21 \times 10^{-2}\) s |

Environment recorded in `data/results/revision/final_tables/table11_computational_complexity.md`.

---

## Related Work

Gizzini, F., et al., *“Enhancing Least Square Channel Estimation Using Deep Learning,”* IEEE VTC2020-Spring, 2020.  
DOI: [10.1109/VTC2020-Spring48590.2020.9128890](https://doi.org/10.1109/VTC2020-Spring48590.2020.9128890)

**Relationship to this repository:**

- This codebase does **not** reuse or adapt the original Gizzini et al. implementation.
- Simulation, training, and evaluation pipelines were implemented **independently** in Python for reproducible comparison of LS, Simplified-MMSE, LMMSE-flat, and LS+DNN.

The repository root may also contain `MatLab_Codes/` and `Python_Codes/` from the paper authors for historical reference; the `ofdm-channel-estimation/` package is a separate implementation.

---

## Reproducibility Note

- Random seeds are fixed per run (`set_global_seed`).
- Train/validation/test splits are **deterministic** given seed and SNR-stratified policy.
- Large generated files (raw NPZ, processed NPZ, trained models) are **excluded from Git** but can be recreated with `scripts/20_run_revision_pipeline.py`.
- Committed revision **metrics, tables, and final figures** under `data/results/revision/` reflect the last pipeline run documented in the project.

---

## Limitations

- **Single-tap flat fading only** — no frequency-selective multipath.
- **No Doppler** or time-varying channel within an OFDM symbol.
- **LMMSE-flat** uses a **zero-mean** prior (\(\sigma_h^2 = 1\)); it does **not** explicitly incorporate a non-zero Rician LoS mean in the prior (see `src/estimators/lmmse_flat_estimator.py`).
- **LS+DNN** can exhibit **high-SNR error-floor** behavior relative to LS or LMMSE-flat.
- **Complexity timings** depend on CPU, TensorFlow build, and system load.
- **Simplified-MMSE** is a pilot-wise shrinkage + interpolation heuristic, not full pilot-domain matrix LMMSE.

---

## Citation

If you use this repository, please cite the associated manuscript once it is published. A formal BibTeX entry will be added after publication.

For the original LS+DNN concept, cite Gizzini et al. (VTC 2020) as in **Related Work** above.

---

## Authors / License

Author names and affiliations are given in the associated **FUJECE manuscript**; this repository does not include a separate `AUTHORS` file.

No `LICENSE` file is currently present in the repository root; do not assume a specific open-source license unless one is added explicitly.

---

## Acknowledgements

Conceptual inspiration from the LS+DNN channel estimation line of work (Gizzini et al., IEEE VTC 2020). Original MATLAB/Keras artifacts in sibling folders are provided for bibliographic comparison only.
