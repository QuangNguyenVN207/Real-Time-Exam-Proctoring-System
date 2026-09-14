from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "weights"
DEFAULT_DATASET = HERE / "dataset" / "cheating.csv"
DEFAULT_TRAINER_STATE = DEFAULT_MODEL_DIR / "checkpoint-423" / "trainer_state.json"
DEFAULT_OUTPUT_DIR = HERE.parent / "outputs" / "p1_phobert"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    df["source_row"] = df.index + 2
    df["text"] = df["text"].astype(str)
    df["normalized_text"] = df["text"].str.strip().str.lower()
    df["processed_text"] = df["text"].map(
        lambda value: word_tokenize(value.lower().strip(), format="text")
    )
    return df


def reproduce_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_indices, val_indices = train_test_split(
        df.index,
        test_size=0.15,
        random_state=42,
        stratify=df["label"],
    )
    return df.loc[train_indices].copy(), df.loc[val_indices].copy()


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def threshold_table(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = [classification_metrics(labels, scores, value / 100) for value in range(101)]
    for row in rows:
        row["threshold"] = round(row["threshold"], 2)
    return pd.DataFrame(rows)


def reproduce_weighted_trainer_eval_loss(
    labels: np.ndarray,
    scores: np.ndarray,
    class_counts: dict[int, int],
    batch_size: int = 16,
) -> float:
    """Match WeightedTrainer's per-batch CrossEntropyLoss aggregation."""
    total = sum(class_counts.values())
    class_weights = np.array(
        [total / (len(class_counts) * class_counts[index]) for index in range(2)]
    )
    batch_losses = []
    batch_sizes = []
    for start in range(0, len(labels), batch_size):
        batch_labels = labels[start : start + batch_size]
        batch_scores = np.clip(scores[start : start + batch_size], 1e-12, 1 - 1e-12)
        probabilities = np.where(batch_labels == 1, batch_scores, 1 - batch_scores)
        weights = class_weights[batch_labels]
        batch_losses.append(float(np.sum(-np.log(probabilities) * weights) / np.sum(weights)))
        batch_sizes.append(len(batch_labels))
    return float(np.average(batch_losses, weights=batch_sizes))


def choose_threshold(sweep: pd.DataFrame) -> float:
    """Maximize F1; break ties by choosing the value closest to 0.50."""
    best_f1 = sweep["f1"].max()
    candidates = sweep[np.isclose(sweep["f1"], best_f1)].copy()
    candidates["distance_to_0_5"] = (candidates["threshold"] - 0.5).abs()
    return float(candidates.sort_values(["distance_to_0_5", "threshold"]).iloc[0]["threshold"])


def infer_scores(
    model_dir: Path,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)
    model.eval()

    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            chunks.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(chunks)


def extract_epoch_metrics(trainer_state: dict) -> pd.DataFrame:
    history = trainer_state.get("log_history", [])
    train_rows = [row for row in history if "loss" in row and "step" in row]
    eval_rows = [row for row in history if "eval_loss" in row and "step" in row]
    result = []
    for eval_row in eval_rows:
        prior = [row for row in train_rows if row["step"] <= eval_row["step"]]
        result.append(
            {
                "epoch": float(eval_row["epoch"]),
                "step": int(eval_row["step"]),
                "train_loss_last_logged": float(prior[-1]["loss"]) if prior else np.nan,
                "eval_loss": float(eval_row["eval_loss"]),
                "eval_accuracy": float(eval_row["eval_accuracy"]),
                "eval_precision": float(eval_row["eval_precision"]),
                "eval_recall": float(eval_row["eval_recall"]),
                "eval_f1": float(eval_row["eval_f1"]),
            }
        )
    return pd.DataFrame(result)


def plot_training(epoch_df: pd.DataFrame, output_path: Path, best_epoch: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epoch_df["epoch"], epoch_df["train_loss_last_logged"], marker="o", label="Train loss (last logged)")
    axes[0].plot(epoch_df["epoch"], epoch_df["eval_loss"], marker="s", label="Validation loss")
    axes[0].axvline(best_epoch, color="#666666", linestyle="--", label=f"Best checkpoint (epoch {best_epoch:g})")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="PhoBERT training and validation loss")
    axes[0].legend(frameon=False)

    for column, label in (
        ("eval_accuracy", "Accuracy"),
        ("eval_precision", "Precision"),
        ("eval_recall", "Recall"),
        ("eval_f1", "F1"),
    ):
        axes[1].plot(epoch_df["epoch"], epoch_df[column], marker="o", label=label)
    axes[1].axvline(best_epoch, color="#666666", linestyle="--")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0.8, 1.01), title="Validation metrics by epoch")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_thresholds(sweep: pd.DataFrame, output_path: Path, selected: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for column, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")):
        ax.plot(sweep["threshold"], sweep[column], label=label)
    ax.axvline(0.60, color="#777777", linestyle=":", label="Runtime threshold 0.60")
    ax.axvline(selected, color="#111111", linestyle="--", label=f"Best validation F1: {selected:.2f}")
    ax.set(xlabel="PhoBERT cheating-probability threshold", ylabel="Score", ylim=(0, 1.02), title="Validation threshold sweep")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(labels: np.ndarray, scores: np.ndarray, output_path: Path) -> tuple[float, float, float, float]:
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    fnr = 1 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2)
    eer_threshold = float(roc_thresholds[eer_index])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle=":")
    axes[0].scatter([fpr[eer_index]], [tpr[eer_index]], color="#D55E00", label=f"Approx. EER = {eer:.3f}")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[0].legend(frameon=False)

    axes[1].plot(recall, precision, label=f"Average precision = {pr_auc:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.01), title="Precision-recall curve")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return roc_auc, pr_auc, eer, eer_threshold


def plot_confusion(metrics: dict, output_path: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels=["Normal", "Cheating"])
    ax.set_yticks([0, 1], labels=["Normal", "Cheating"])
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"Validation confusion matrix (threshold={metrics['threshold']:.2f})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the P1 PhoBERT validation outputs without retraining.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trainer-state", type=Path, default=DEFAULT_TRAINER_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    args.model_dir = args.model_dir.resolve()
    args.dataset = args.dataset.resolve()
    args.trainer_state = args.trainer_state.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_model_files = ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt", "bpe.codes"]
    missing = [name for name in required_model_files if not (args.model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model directory {args.model_dir}; missing: {missing}")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    df = load_dataset(args.dataset)
    train_df, val_df = reproduce_split(df)
    train_texts = set(train_df["normalized_text"])
    val_df["seen_in_train_exact"] = val_df["normalized_text"].isin(train_texts)
    val_df["cheating_probability"] = infer_scores(
        args.model_dir,
        val_df["processed_text"].tolist(),
        args.batch_size,
        device,
    )

    labels = val_df["label"].to_numpy()
    scores = val_df["cheating_probability"].to_numpy()
    class_counts = {int(key): int(value) for key, value in df["label"].value_counts().items()}
    reproduced_eval_loss = reproduce_weighted_trainer_eval_loss(
        labels,
        scores,
        class_counts,
        batch_size=16,
    )
    sweep = threshold_table(labels, scores)
    selected_threshold = choose_threshold(sweep)

    official_metrics = {
        "argmax_0_50": classification_metrics(labels, scores, 0.50),
        "runtime_0_60": classification_metrics(labels, scores, 0.60),
        "best_validation_f1": classification_metrics(labels, scores, selected_threshold),
    }
    clean_mask = ~val_df["seen_in_train_exact"].to_numpy()
    leakage_free_sensitivity = classification_metrics(
        labels[clean_mask],
        scores[clean_mask],
        selected_threshold,
    )

    val_df["prediction_0_50"] = (scores >= 0.50).astype(int)
    val_df["prediction_0_60"] = (scores >= 0.60).astype(int)
    val_df["prediction_selected"] = (scores >= selected_threshold).astype(int)
    val_df["error_selected"] = val_df["prediction_selected"] != val_df["label"]
    prediction_columns = [
        "source_row",
        "text",
        "processed_text",
        "label",
        "cheating_probability",
        "prediction_0_50",
        "prediction_0_60",
        "prediction_selected",
        "error_selected",
        "seen_in_train_exact",
    ]
    val_df[prediction_columns].sort_values("source_row").to_csv(
        args.output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sweep.to_csv(args.output_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")

    trainer_state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    epoch_df = extract_epoch_metrics(trainer_state)
    epoch_df.to_csv(args.output_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")
    best_epoch = float(epoch_df.loc[epoch_df["step"] == trainer_state["best_global_step"], "epoch"].iloc[0])
    model_metrics = official_metrics["argmax_0_50"]
    comparison_columns = ["accuracy", "precision", "recall", "f1"]
    epoch_df["model_log_max_abs_delta"] = epoch_df.apply(
        lambda row: max(
            [
                abs(float(row[f"eval_{name}"]) - float(model_metrics[name]))
                for name in comparison_columns
            ]
            + [abs(float(row["eval_loss"]) - reproduced_eval_loss)]
        ),
        axis=1,
    )
    closest_epoch_row = epoch_df.sort_values("model_log_max_abs_delta").iloc[0]
    inferred_weight_epoch = float(closest_epoch_row["epoch"])
    inferred_weight_step = int(closest_epoch_row["step"])
    metric_match_delta = float(closest_epoch_row["model_log_max_abs_delta"])
    epoch_df.to_csv(args.output_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")

    roc_auc, pr_auc, eer, eer_threshold = plot_roc_pr(labels, scores, args.output_dir / "roc_pr_curves.png")
    plot_training(epoch_df, args.output_dir / "training_curves.png", best_epoch)
    plot_thresholds(sweep, args.output_dir / "threshold_sweep.png", selected_threshold)
    plot_confusion(official_metrics["best_validation_f1"], args.output_dir / "confusion_matrix.png")

    summary = {
        "scope": {
            "fine_tuned_component": "vinai/phobert-base-v2 binary text classifier",
            "asr_component": "vinai/PhoWhisper-small pretrained; not evaluated by this script",
            "validation_protocol": "Exact train_test_split(test_size=0.15, random_state=42, stratify=label) from train_phobert.py",
            "threshold_selection_warning": "The best threshold is descriptive because it is selected and measured on the same validation split; it is not a locked-test estimate.",
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256_file(args.dataset),
            "rows": int(len(df)),
            "class_counts": {str(key): int(value) for key, value in df["label"].value_counts().sort_index().items()},
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "exact_duplicate_rows": int(df.duplicated(["normalized_text", "label"]).sum()),
            "validation_rows_seen_exactly_in_train": int(val_df["seen_in_train_exact"].sum()),
        },
        "training_log": {
            "requested_max_epochs": float(trainer_state["num_train_epochs"]),
            "stopped_epoch": float(trainer_state["epoch"]),
            "final_checkpoint_step": int(trainer_state["global_step"]),
            "best_checkpoint_step": int(trainer_state["best_global_step"]),
            "best_epoch": best_epoch,
            "best_logged_f1": float(trainer_state["best_metric"]),
            "best_model_checkpoint_original_path": trainer_state["best_model_checkpoint"],
            "available_weight_interpretation": {
                "inferred_epoch": inferred_weight_epoch,
                "inferred_step": inferred_weight_step,
                "reproduced_weighted_eval_loss": reproduced_eval_loss,
                "evidence": "The loaded model's weighted eval loss plus threshold-0.50 accuracy, precision, recall and F1 are compared with every logged epoch.",
                "maximum_absolute_metric_delta": metric_match_delta,
                "best_checkpoint_weight_available": True,
            },
        },
        "model": {
            "path": str(args.model_dir),
            "model_safetensors_sha256": sha256_file(args.model_dir / "model.safetensors"),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "device": str(device),
        },
        "validation": {
            "selected_threshold": selected_threshold,
            "metrics": official_metrics,
            "roc_auc": roc_auc,
            "average_precision": pr_auc,
            "approximate_eer": eer,
            "approximate_eer_threshold": eer_threshold,
            "sensitivity_excluding_exact_train_duplicates": leakage_free_sensitivity,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = official_metrics["best_validation_f1"]
    runtime_metrics = official_metrics["runtime_0_60"]
    report = f"""# Kết quả kiểm thử PhoBERT dùng cho thuyết minh

- Thành phần được fine-tune: `vinai/phobert-base-v2`, phân loại hai lớp Normal/Cheating.
- Thành phần nhận dạng tiếng nói: `vinai/PhoWhisper-small` pretrained, không fine-tune trong công trình này.
- Dữ liệu hợp lệ: {len(df)} câu; train {len(train_df)}, validation {len(val_df)} theo seed 42 và stratify.
- Huấn luyện đặt tối đa {trainer_state['num_train_epochs']:g} epoch và dừng sớm ở epoch {trainer_state['epoch']:g}. Checkpoint tốt nhất là step {trainer_state['best_global_step']} (epoch {best_epoch:g}), F1 log = {trainer_state['best_metric']:.4f}.
- Trọng số `model.safetensors` hiện có khớp weighted eval loss và toàn bộ bốn metric của epoch {inferred_weight_epoch:g}, step {inferred_weight_step} (sai khác lớn nhất {metric_match_delta:.2e}); đây là trọng số được kiểm thử và đóng gói.
- Cấu hình vận hành dùng threshold 0.60: accuracy {runtime_metrics['accuracy']:.4f}, precision {runtime_metrics['precision']:.4f}, recall {runtime_metrics['recall']:.4f}, F1 {runtime_metrics['f1']:.4f} (TP={runtime_metrics['tp']}, TN={runtime_metrics['tn']}, FP={runtime_metrics['fp']}, FN={runtime_metrics['fn']}).
- Sweep trên chính validation có điểm F1 cao nhất tại {selected_threshold:.2f}: accuracy {metrics['accuracy']:.4f}, precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, F1 {metrics['f1']:.4f}. Điểm này chỉ là kết quả phân tích độ nhạy, không thay thế cấu hình vận hành 0.60.
- ROC AUC = {roc_auc:.4f}; average precision = {pr_auc:.4f}; EER xấp xỉ = {eer:.4f}.
- Có {int(val_df['seen_in_train_exact'].sum())} hàng validation trùng nguyên văn với train. Sau khi loại các hàng đó để phân tích độ nhạy (không phải retrain), F1 = {leakage_free_sensitivity['f1']:.4f} trên {leakage_free_sensitivity['n']} câu.

Lưu ý: threshold tốt nhất được chọn và đo trên cùng validation split, vì vậy chỉ dùng để giải thích lựa chọn cấu hình. Không gọi đây là kết quả locked test hay khả năng tổng quát trên dữ liệu độc lập.
"""
    (args.output_dir / "report_snippet_vi.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
