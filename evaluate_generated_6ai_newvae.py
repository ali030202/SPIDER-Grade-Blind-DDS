"""Evaluate a generated 21-frame LCOVT run with the frozen six AI raters.

The input is the generator's trajectory_metrics_all.csv. This script does not
train or recalibrate a rater. Each architecture is the equal-probability mean
of its three frozen seeds; the six architecture raters are then compared using
hard majority vote, mean expected grade, and mean probability.

Run on the SPIDER server where the generated PNGs and rater checkpoints exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error

import train_multi_ai_raters_6arch_3seed_native_160x224 as raters


VERSION = "GENERATED_6AI_NEWVAE_FROZEN_EVAL_V1"
MODELS = (
    "resnet50", "densenet201", "convnext_tiny", "coatnet_0",
    "swin_tiny", "efficientnet_b3",
)
SEEDS = (2026, 2027, 2028)
GRADES = np.arange(1, 6, dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_stat_signature(frame: pd.DataFrame) -> str:
    """Invalidate seed caches when generated PNGs change at the same paths."""
    digest = hashlib.sha256()
    for raw_path in frame["step_file_path"]:
        path = Path(raw_path)
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "metadata_index", "patient_id", "sample_id", "source_grade",
        "level_index", "step_index", "continuous_grade", "step_file_path",
        "is_integer_anchor",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Trajectory CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Trajectory CSV is empty")

    for column in ("metadata_index", "source_grade", "level_index", "step_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["continuous_grade"] = pd.to_numeric(frame["continuous_grade"], errors="raise")
    frame["is_integer_anchor"] = frame["is_integer_anchor"].astype(str).str.lower().eq("true")
    frame = frame.sort_values(["metadata_index", "step_index"]).reset_index(drop=True)

    if frame.duplicated(["metadata_index", "step_index"]).any():
        raise ValueError("Duplicate source/step rows in trajectory CSV")
    counts = frame.groupby("metadata_index").size()
    if not counts.eq(21).all():
        raise ValueError(f"Expected 21 frames per source; first exceptions: {counts[counts.ne(21)].head().to_dict()}")
    anchors = frame[frame["is_integer_anchor"]].groupby("metadata_index").size()
    if not anchors.reindex(counts.index, fill_value=0).eq(5).all():
        raise ValueError("Expected five integer anchors per source")
    if not frame["source_grade"].between(1, 5).all():
        raise ValueError("source_grade must be 1..5")
    integer_q = frame.loc[frame["is_integer_anchor"], "continuous_grade"].to_numpy(float)
    if not np.allclose(integer_q, np.rint(integer_q), atol=1e-6):
        raise ValueError("An integer anchor has a non-integer requested grade")
    if not frame.groupby("metadata_index")["patient_id"].nunique().eq(1).all():
        raise ValueError("A source has multiple patient_id values")
    if not frame.groupby("metadata_index")["source_grade"].nunique().eq(1).all():
        raise ValueError("A source has multiple source_grade values")

    paths = frame["step_file_path"].map(Path)
    missing_paths = [str(p) for p in paths if not p.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} generated images are missing; examples: {missing_paths[:5]}")
    frame["resolved_image_path"] = frame["step_file_path"]
    frame["grade"] = frame["source_grade"]  # Dataset requires a label; inference does not use it.
    return frame


def predict_seed(
    model_key: str,
    seed: int,
    run_dir: Path,
    cache_dir: Path,
    input_hash: str,
    loader,
    frame: pd.DataFrame,
    device: torch.device,
    amp: bool,
    image_height: int,
    image_width: int,
    force: bool,
) -> tuple[np.ndarray, dict]:
    checkpoint_path = run_dir / model_key / f"seed_{seed}" / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    stat = checkpoint_path.stat()
    tag = f"{VERSION}|{input_hash}|{stat.st_size}|{stat.st_mtime_ns}|{image_height}x{image_width}"
    cache_path = cache_dir / f"{model_key}_seed_{seed}.npz"
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            probs = cached["probs"]
            if str(cached["tag"].item()) == tag and probs.shape == (len(frame), 5):
                print(f"[cache] {model_key} seed={seed}", flush=True)
                return probs, {"model": model_key, "seed": seed, "checkpoint": str(checkpoint_path), "cached": True}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preprocessing = checkpoint.get("input_preprocessing", {})
    if int(preprocessing.get("image_height", -1)) != image_height or int(preprocessing.get("image_width", -1)) != image_width:
        raise ValueError(f"Input geometry differs from frozen checkpoint: {checkpoint_path}")
    temperature = float(checkpoint["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Invalid validation-set temperature: {checkpoint_path}")

    model, timm_name = raters.create_classifier(
        model_key=model_key, pretrained=False,
        image_height=image_height, image_width=image_width,
    )
    if timm_name != checkpoint.get("timm_model_name"):
        raise ValueError(f"timm model mismatch for {checkpoint_path}: {timm_name}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    logits, _, meta = raters.collect_logits(
        model, loader, device, amp,
        desc=f"generated {model_key} seed={seed}",
    )
    if not np.array_equal(meta["image_path"].to_numpy(), frame["step_file_path"].to_numpy()):
        raise RuntimeError("Prediction rows do not match trajectory rows")
    probs = raters.logits_to_probs(logits, temperature).astype(np.float32)
    if probs.shape != (len(frame), 5) or not np.isfinite(probs).all():
        raise RuntimeError(f"Invalid probabilities from {checkpoint_path}")
    np.savez_compressed(cache_path, probs=probs, tag=np.asarray(tag))
    del model, logits, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, {
        "model": model_key, "seed": seed, "checkpoint": str(checkpoint_path),
        "temperature_from_real_validation": temperature, "cached": False,
    }


def add_aggregations(frame: pd.DataFrame, architecture_probs: dict[str, np.ndarray]) -> pd.DataFrame:
    out = frame[[
        "metadata_index", "patient_id", "sample_id", "source_grade", "level_index",
        "step_index", "continuous_grade", "step_file_path", "is_integer_anchor",
    ]].copy()
    target = np.rint(out["continuous_grade"].to_numpy(float)).astype(int)
    is_integer = out["is_integer_anchor"].to_numpy(bool)
    source = out["source_grade"].to_numpy(int)
    out["target_integer_grade"] = np.where(is_integer, target, np.nan)
    out["is_native_anchor"] = is_integer & (target == source)
    out["is_counterfactual_anchor"] = is_integer & (target != source)
    out["edit_direction"] = np.where(target > source, "higher_grade", np.where(target < source, "lower_grade", "native"))

    probs = np.stack([architecture_probs[name] for name in MODELS], axis=1)
    hard = probs.argmax(axis=2).astype(int) + 1
    for index, name in enumerate(MODELS):
        out[f"{name}__hard_grade"] = hard[:, index]
        out[f"{name}__expected_grade"] = probs[:, index, :] @ GRADES
        for grade in range(1, 6):
            out[f"{name}__p_G{grade}"] = probs[:, index, grade - 1]

    soft = probs.mean(axis=1)
    out["probability_consensus_grade"] = soft.argmax(axis=1) + 1
    out["CONSENSUS__hard_grade"] = out["probability_consensus_grade"]
    out["CONSENSUS__expected_grade"] = soft @ GRADES
    for grade in range(1, 6):
        out[f"CONSENSUS__p_G{grade}"] = soft[:, grade - 1]

    votes = np.empty(len(out), dtype=int)
    for index, row in enumerate(hard):
        counts = np.bincount(row, minlength=6)[1:]
        tied = np.flatnonzero(counts == counts.max())
        votes[index] = int(tied[np.argmax(soft[index, tied])]) + 1
    out["majority_vote_grade"] = votes
    out["mean_expected_grade"] = (probs @ GRADES).mean(axis=1)
    out["mean_expected_grade_round"] = np.clip(np.rint(out["mean_expected_grade"]), 1, 5).astype(int)
    return out


def summarize_anchors(pred: pd.DataFrame, output_dir: Path) -> list[dict]:
    subsets = {
        "all_integer_anchors": pred["is_integer_anchor"],
        "native_anchors_only": pred["is_native_anchor"],
        "counterfactual_anchors_only": pred["is_counterfactual_anchor"],
    }
    methods = {
        "majority_vote": "majority_vote_grade",
        "mean_expected_grade_round": "mean_expected_grade_round",
        "probability_consensus": "probability_consensus_grade",
    }
    rows = []
    for subset_name, mask in subsets.items():
        part = pred.loc[mask]
        y = part["target_integer_grade"].to_numpy(int)
        for method_name, column in methods.items():
            yhat = part[column].to_numpy(int)
            rows.append({
                "subset": subset_name, "method": method_name, "n": len(part),
                "accuracy": accuracy_score(y, yhat),
                "qwk": cohen_kappa_score(y, yhat, weights="quadratic"),
                "hard_grade_mae": mean_absolute_error(y, yhat),
                "within_one_grade_accuracy": np.mean(np.abs(yhat - y) <= 1),
            })
    pd.DataFrame(rows).to_csv(output_dir / "three_aggregation_comparison_summary.csv", index=False)

    counterfactual = pred.loc[subsets["counterfactual_anchors_only"]]
    target_rows = []
    for target, part in counterfactual.groupby("target_integer_grade", sort=True):
        truth = part["target_integer_grade"].to_numpy(int)
        hard = part["probability_consensus_grade"].to_numpy(int)
        target_rows.append({
            "target_grade": int(target), "n": len(part),
            "accuracy": np.mean(hard == truth),
            "mean_expected_grade": part["CONSENSUS__expected_grade"].mean(),
        })
    pd.DataFrame(target_rows).to_csv(output_dir / "counterfactual_by_target_grade.csv", index=False)

    cm = confusion_matrix(
        counterfactual["target_integer_grade"].to_numpy(int),
        counterfactual["probability_consensus_grade"].to_numpy(int),
        labels=[1, 2, 3, 4, 5],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="viridis")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=[f"G{i}" for i in range(1, 6)],
           yticklabels=[f"G{i}" for i in range(1, 6)], xlabel="AI predicted grade",
           ylabel="Requested grade", title="New VAE: counterfactual anchors")
    fig.tight_layout()
    fig.savefig(output_dir / "counterfactual_confusion_probability_consensus.png", dpi=160)
    plt.close(fig)
    return rows


def summarize_trajectories(pred: pd.DataFrame, output_dir: Path) -> dict:
    rows = []
    for source_index, part in pred.groupby("metadata_index", sort=True):
        part = part.sort_values("step_index")
        q = part["continuous_grade"].to_numpy(float)
        severity = part["CONSENSUS__expected_grade"].to_numpy(float)
        diff = np.diff(severity)
        rows.append({
            "metadata_index": source_index, "patient_id": part["patient_id"].iloc[0],
            "source_grade": int(part["source_grade"].iloc[0]),
            "spearman": float(spearmanr(q, severity).statistic),
            "continuous_mae": float(np.mean(np.abs(severity - q))),
            "linear_slope": float(np.polyfit(q, severity, 1)[0]),
            "strict_adjacent_step_rate": float(np.mean(diff > 0)),
            "fully_strict_trajectory": bool(np.all(diff > 0)),
            "endpoint_gain_G5_minus_G1": float(severity[-1] - severity[0]),
        })
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(output_dir / "trajectory_metrics_per_source_consensus.csv", index=False)
    summary = {
        "n_sources": int(len(trajectory)), "n_frames": int(len(pred)),
        "mean_spearman": float(trajectory["spearman"].mean()),
        "mean_continuous_mae": float(trajectory["continuous_mae"].mean()),
        "mean_linear_slope": float(trajectory["linear_slope"].mean()),
        "mean_strict_adjacent_step_rate": float(trajectory["strict_adjacent_step_rate"].mean()),
        "fully_strict_trajectory_fraction": float(trajectory["fully_strict_trajectory"].mean()),
        "positive_endpoint_fraction": float(np.mean(trajectory["endpoint_gain_G5_minus_G1"] > 0)),
    }
    (output_dir / "trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    means = pred.groupby("continuous_grade", sort=True)["CONSENSUS__expected_grade"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(means.index, means.values, marker="o", label="6-AI probability consensus")
    ax.plot([1, 5], [1, 5], "--", color="gray", label="ideal requested grade")
    ax.set(xlabel="Requested grade q", ylabel="Mean AI expected grade", title="New VAE: generated trajectory response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mean_predicted_severity_vs_q.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--raters-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute existing per-seed prediction caches")
    parser.add_argument("--preflight-only", action="store_true", help="Check inputs without loading models")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be nonnegative")

    generated_dir = args.generated_dir.expanduser().resolve()
    raters_dir = args.raters_dir.expanduser().resolve()
    input_csv = generated_dir / "trajectory_metrics_all.csv"
    launch_path = raters_dir / "launch_manifest.json"
    missing_inputs = [path for path in (input_csv, launch_path) if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "Required input file(s) missing:\n" +
            "\n".join(f"  {path}" for path in missing_inputs)
        )
    frame = load_trajectory(input_csv)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    height = int(launch["train_config"]["image_height"])
    width = int(launch["train_config"]["image_width"])
    strict_size = bool(launch["train_config"]["strict_source_size"])
    if tuple(launch["models"]) != MODELS or tuple(launch["seeds"]) != SEEDS:
        raise ValueError("Rater manifest differs from expected frozen 6-architecture/3-seed protocol")
    for name in MODELS:
        for seed in SEEDS:
            path = raters_dir / name / f"seed_{seed}" / "checkpoints" / "best.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
    print(f"[{VERSION}] generated={generated_dir}")
    print(f"sources={frame.metadata_index.nunique()} frames={len(frame)} integer_anchors={int(frame.is_integer_anchor.sum())}")
    print(f"raters={raters_dir} geometry={height}x{width} strict_source_size={strict_size}")
    if args.preflight_only:
        return

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.out.expanduser().resolve()
    if output_dir in (generated_dir, raters_dir):
        raise ValueError("Output directory must differ from generated images and frozen rater runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "seed_probability_cache"
    cache_dir.mkdir(exist_ok=True)
    dataset = raters.PfirrmannDataset(
        frame, raters.build_transforms(height, width, train=False, strict_source_size=strict_size)
    )
    loader = raters.build_eval_loader(dataset, args.batch_size, args.num_workers, seed=2026)
    csv_hash = sha256_file(input_csv)
    image_hash = image_stat_signature(frame)
    input_hash = hashlib.sha256(f"{csv_hash}|{image_hash}".encode("utf-8")).hexdigest()
    architecture_probs = {}
    audit = []
    for name in MODELS:
        seed_probs = []
        for seed in SEEDS:
            probs, record = predict_seed(
                name, seed, raters_dir, cache_dir, input_hash, loader, frame,
                device, not args.no_amp, height, width, args.force,
            )
            seed_probs.append(probs)
            audit.append(record)
        architecture_probs[name] = np.mean(np.stack(seed_probs, axis=0), axis=0)
        print(f"[done] {name}: 3 seeds", flush=True)

    predictions = add_aggregations(frame, architecture_probs)
    predictions.to_csv(output_dir / "generated_predictions_all_frames_6ai_newvae.csv", index=False)
    anchor_rows = summarize_anchors(predictions, output_dir)
    trajectory_summary = summarize_trajectories(predictions, output_dir)
    manifest = {
        "script_version": VERSION,
        "generated_dir": str(generated_dir), "raters_dir": str(raters_dir),
        "output_dir": str(output_dir), "trajectory_csv_sha256": csv_hash,
        "generated_image_stat_signature": image_hash,
        "n_frames": len(frame), "n_sources": int(frame.metadata_index.nunique()),
        "n_patient_ids": int(frame.patient_id.nunique()),
        "n_integer_anchors": int(frame.is_integer_anchor.sum()),
        "models": list(MODELS), "seeds": list(SEEDS),
        "checkpoint_audit": audit,
        "temperature_policy": "Frozen from real validation; no calibration on generated images",
        "aggregation_policy": "Equal probability weight across seeds, then equal weight across architectures",
        "target_grade_note": "Requested generation condition, not clinical ground truth",
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(anchor_rows).to_string(index=False))
    print(json.dumps(trajectory_summary, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
