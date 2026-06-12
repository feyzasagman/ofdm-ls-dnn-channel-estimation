# OFDM Channel Estimation (LS + DNN)

End-to-end Python pipeline for OFDM channel estimation using classical LS/MMSE estimators and a deep neural network refinement stage.

See the [repository root README](../README.md) for setup, citation, and full documentation.

## Requirements

- Python 3.10+
- TensorFlow 2.12+
- NumPy, SciPy, Matplotlib, scikit-learn, pytest

```bash
pip install -r requirements.txt
```

## Scripts

| Script | Purpose |
|--------|---------|
| `01_generate_data.py` | Generate raw NPZ datasets and preprocess for DNN |
| `02_train_dnn.py` | Train the LS-DNN regressor |
| `03_run_comparison.py` | Benchmark LS vs MMSE vs LS+DNN |
| `04_plot_results.py` | Plot SNR vs MSE/NMSE curves |
| `05_aggregate_seed_results.py` | Aggregate metrics across seeds (42, 123, 999) |
| `06_plot_overall_comparison.py` | Bar chart across channel types |
| `07_plot_seed_mean_std.py` | Mean ± std over seeds |
| `08_plot_training_history.py` | Training / validation loss curves |
| `09_plot_example_channel_estimate.py` | Visualize a single channel estimate |
| `10_plot_improvement_percentage.py` | DNN improvement over LS/MMSE (%) |

## Tests

```bash
pytest tests/ -v
```

## Key design choices

- **Residual mode** (default): DNN learns `true_channel − ls_estimate`, then adds the correction back at inference.
- **Input features**: real/imaginary parts of the LS estimate (128 dims for 64 subcarriers).
- **Architecture**: Dense(256) → ReLU → Dense(256) → ReLU → Dense(output).
- **SNR range**: −10 to 30 dB in 5 dB steps.
