import os
import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from underthesea import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "vinai/phobert-base-v2"

DATA_PATH = os.path.join(
    CURRENT_DIR,
    "dataset",
    "cheating.csv"
)

OUTPUT_DIR = os.path.join(
    CURRENT_DIR,
    "weights"
)

# Tăng lên 128 để bắt được những câu nói dài của sinh viên
MAX_LEN = 128 

# ==========================================================
# DATASET
# ==========================================================

class PhoBertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )
        return item

    def __len__(self):
        return len(self.labels)

# ==========================================================
# WEIGHTED TRAINER
# ==========================================================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(
            class_weights,
            dtype=torch.float
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    # 1. FIX: Làm sạch dữ liệu và xử lý đúng nhãn 0, 1
    df = df.dropna(subset=["text", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    
    # Chỉ giữ lại nhãn 0 (Normal) và 1 (Cheating)
    df = df[df["label"].isin([0, 1])]

    print(f"Tổng số mẫu hợp lệ: {len(df)}")
    print(df['label'].value_counts())

    # 2. Tiền xử lý văn bản: Tách từ tiếng Việt
    df["text"] = df["text"].astype(str).apply(
        lambda x: word_tokenize(x, format="text")
    )

    # 3. Tính toán trọng số lớp (Class weight) để chống mất cân bằng
    classes = np.unique(df["label"])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df["label"]
    )
    print("Class weights:", weights)

    # 4. Chia tập Train/Val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.15,
        random_state=42,
        stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    val_encodings = tokenizer(
        val_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    train_dataset = PhoBertDataset(train_encodings, train_labels.tolist())
    val_dataset = PhoBertDataset(val_encodings, val_labels.tolist())

    print("=" * 60)
    print("Loading PhoBERT...")
    print("=" * 60)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=15,             # Tăng lên 15 để học kỹ tập dữ liệu nhỏ
        warmup_steps=0,                  # Không warmup (bắt đầu học max tốc độ ngay từ đầu)
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4)
        ]
    )

    print("=" * 60)
    print("Start Training...")
    print("=" * 60)

    trainer.train()

    print("=" * 60)
    print("Evaluating...")
    print("=" * 60)

    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n[✓] Saved final model and tokenizer to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()