"""Run full major-revision experiment pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ber_evaluation import evaluate_ber_on_test_split
from src.core.channel_models import channel_metadata
from src.core.complexity import build_complexity_report, save_complexity_report
from src.core.high_snr_analysis import analyze_high_snr_error_floor
from src.core.seed import set_global_seed
from src.core.snr_gain import compute_snr_gain_at_target_ber
from src.core.split import DEFAULT_SEEDS
from src.dataset.generate_dataset import generate_raw_dataset
from src.dataset.preprocess import preprocess_for_dnn
from src.dnn.train import train_dnn
from src.revision.aggregate import aggregate_ber_results, aggregate_per_snr_metrics, save_csv, save_json
from src.revision.comparison import compare_estimators_on_test, predict_dnn_test_channels
from src.revision.figures import plot_ber_log, plot_mse_nmse_log, plot_sample_channel, plot_training_history
from src.revision.paths import ensure_revision_dirs

CHANNELS = ("awgn", "rayleigh", "rician")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Major revision reproducible experiment pipeline.")
    parser.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=CHANNELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-samples-per-snr", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def run_single_seed_channel(
    channel: str,
    seed: int,
    dirs: dict[str, Path],
    n_samples_per_snr: int,
    epochs: int,
    patience: int,
    skip_train: bool,
    skip_generate: bool,
) -> dict:
    set_global_seed(seed)
    tag = f"{channel}_seed{seed}"

    raw_path = dirs["raw"] / f"{channel}_ns{n_samples_per_snr}_seed{seed}.npz"
    if not skip_generate or not raw_path.exists():
        raw_path = generate_raw_dataset(
            channel_type=channel,
            n_samples_per_snr=n_samples_per_snr,
            random_seed=seed,
            output_path=raw_path,
        )

    processed_path = dirs["processed"] / f"{raw_path.stem}_residual_processed.npz"
    split_report_path = dirs["metadata"] / f"split_{tag}.json"
    if not processed_path.exists():
        preprocess_for_dnn(
            raw_dataset_path=raw_path,
            residual_mode=True,
            normalize=True,
            random_seed=seed,
            output_path=processed_path,
            split_report_path=split_report_path,
        )

    processed = dict(np.load(processed_path, allow_pickle=False))
    raw = dict(np.load(raw_path, allow_pickle=False))
    test_indices = np.asarray(processed["test_indices"], dtype=int)

    model_path = dirs["models"] / f"{processed_path.stem}_dnn.keras"
    history_path = dirs["training"] / f"history_{tag}.json"
    if not skip_train or not model_path.exists():
        train_out = train_dnn(
            processed_dataset_path=processed_path,
            epochs=epochs,
            patience=patience,
            random_seed=seed,
            model_path=model_path,
            history_path=history_path,
        )
        model_path = train_out["model_path"]
        history_path = train_out["history_path"]

    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    plot_training_history(history, channel, seed, dirs["figures"])

    model = keras.models.load_model(model_path)
    dnn_pred = predict_dnn_test_channels(model, processed)
    comparison = compare_estimators_on_test(raw, test_indices, dnn_pred)

    meta = {
        "seed": seed,
        "channel_type": channel,
        **channel_metadata(channel, k_factor_db=float(raw.get("k_factor_db", 6.0))),
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "model_path": str(model_path),
        "split_strategy": "snr_stratified",
    }
    mse_out = dirs["mse_nmse"] / f"comparison_{tag}.json"
    save_json({"metadata": meta, **comparison}, mse_out)

    ber_result = evaluate_ber_on_test_split(
        raw_npz_path=str(raw_path),
        test_indices=test_indices,
        pilot_indices=np.asarray(raw["pilot_indices"], dtype=int),
        pilot_symbols=np.asarray(raw["pilot_symbols"], dtype=np.complex128),
        n_subcarriers=int(raw["n_subcarriers"]),
        dnn_pred=dnn_pred,
    )
    ber_out = dirs["ber"] / f"ber_{tag}.json"
    save_json({"metadata": meta, **ber_result}, ber_out)

    complexity = build_complexity_report(
        rx_freq=np.asarray(raw["rx_freq"][0]),
        pilot_indices=np.asarray(raw["pilot_indices"], dtype=int),
        pilot_symbols=np.asarray(raw["pilot_symbols"], dtype=np.complex128),
        n_subcarriers=int(raw["n_subcarriers"]),
        noise_variance=float(raw["noise_variance"][0]),
        model=model,
        x_sample=np.asarray(processed["x_test"][0]),
    )
    save_complexity_report(complexity, dirs["complexity"] / f"complexity_{tag}.json")

    high_snr = analyze_high_snr_error_floor(comparison["per_snr"], channel, seed)
    save_json(high_snr, dirs["high_snr"] / f"high_snr_{tag}.json")

    # Sample channel visualization at 10 dB if available
    snr_all = np.asarray(raw["snr_db"])
    candidates = np.where(snr_all == 10.0)[0]
    if candidates.size > 0:
        sample_idx = int(candidates[0])
        plot_meta = {
            **meta,
            "snr_db": 10.0,
            "sample_index": sample_idx,
            "flat_fading": True,
            "caption_note": "Flat fading: |H| is constant across subcarriers.",
        }
        plot_sample_channel(
            true_h=np.asarray(raw["true_channel"][sample_idx]),
            ls_h=np.asarray(raw["ls_estimate"][sample_idx]),
            metadata=plot_meta,
            output_dir=dirs["figures"],
        )
        save_json(plot_meta, dirs["metadata"] / f"sample_channel_{tag}.json")

    return {
        "seed": seed,
        "channel": channel,
        "comparison": comparison,
        "ber": ber_result,
        "history": history,
    }


def aggregate_channel(channel: str, seed_results: list[dict], dirs: dict[str, Path]) -> None:
    mse_keys = [
        "ls_mse",
        "ls_nmse",
        "simplified_mmse_mse",
        "simplified_mmse_nmse",
        "lmmse_flat_mse",
        "lmmse_flat_nmse",
        "ls_dnn_mse",
        "ls_dnn_nmse",
    ]
    mse_agg = aggregate_per_snr_metrics(
        [r["comparison"] for r in seed_results],
        mse_keys,
    )
    save_json(mse_agg, dirs["mse_nmse"] / f"aggregate_{channel}.json")
    save_csv(mse_agg["per_snr"], dirs["mse_nmse"] / f"aggregate_{channel}.csv")

    ber_agg = aggregate_ber_results([r["ber"] for r in seed_results])
    save_json(ber_agg, dirs["ber"] / f"aggregate_{channel}.json")
    save_csv(ber_agg["per_snr"], dirs["ber"] / f"aggregate_{channel}.csv")

    plot_mse_nmse_log(mse_agg, channel, dirs["figures"], metric="mse")
    plot_mse_nmse_log(mse_agg, channel, dirs["figures"], metric="nmse")
    plot_ber_log(ber_agg, channel, dirs["figures"])

    # SNR gain from aggregated BER curves
    snr = np.asarray([r["snr_db"] for r in ber_agg["per_snr"]])
    ber_ls = np.asarray([r["ls_ber_mean"] for r in ber_agg["per_snr"]])
    ber_dnn = np.asarray([r["ls_dnn_ber_mean"] for r in ber_agg["per_snr"]])
    snr_gain = compute_snr_gain_at_target_ber(snr, ber_ls, ber_dnn)
    save_json(
        {"channel": channel, "n_seeds": ber_agg["n_seeds"], "snr_gain": snr_gain},
        dirs["snr_gain"] / f"snr_gain_{channel}.json",
    )


def main() -> None:
    args = parse_args()
    dirs = ensure_revision_dirs(PROJECT_ROOT)
    all_results: dict[str, list[dict]] = {c: [] for c in args.channels}

    for channel in args.channels:
        for seed in args.seeds:
            print(f"=== Running {channel} seed={seed} ===")
            result = run_single_seed_channel(
                channel=channel,
                seed=seed,
                dirs=dirs,
                n_samples_per_snr=args.n_samples_per_snr,
                epochs=args.epochs,
                patience=args.patience,
                skip_train=args.skip_train,
                skip_generate=args.skip_generate,
            )
            all_results[channel].append(result)

        if all_results[channel]:
            aggregate_channel(channel, all_results[channel], dirs)

    save_json(
        {
            "channels": args.channels,
            "seeds": args.seeds,
            "output_root": str(dirs["root"]),
            "note": "All results generated by re-running experiments; no manual CSV/JSON edits.",
        },
        dirs["metadata"] / "revision_run_summary.json",
    )
    print(f"Revision pipeline complete. Outputs: {dirs['root']}")


if __name__ == "__main__":
    main()
