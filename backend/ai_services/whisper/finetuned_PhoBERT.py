import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# 1. Tải dữ liệu từ file CSV của bạn
df = pd.read_csv("data/phong_thi_data.csv")
dataset = Dataset.from_pandas(df)

# 2. Tải Tokenizer và Model PhoBERT (bản base)
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Khởi tạo mô hình cho bài toán phân loại 2 nhãn (0 và 1)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Chuyển đổi văn bản thành dạng số (Tensor) cho AI hiểu
def tokenize_function(examples):
    # Cắt ngắn hoặc đệm thêm để mọi câu đều có độ dài 128 token
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Chia tập dữ liệu: 80% để học (train) và 20% để thi thử (test)
split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 4. Cấu hình các tham số quá trình học
training_args = TrainingArguments(
    output_dir="./phobert_phong_thi_checkpoints", 
    eval_strategy="epoch",      # Kiểm tra trình độ sau mỗi vòng lặp
    learning_rate=2e-5,               # Tốc độ học (Nên để nhỏ với PhoBERT)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,               # Lặp lại việc học qua toàn bộ dữ liệu 4 lần
    weight_decay=0.01,
)

# 5. Khởi động bộ huấn luyện
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("[INFO] Bắt đầu huấn luyện PhoBERT...")
trainer.train()

# 6. Xuất xưởng mô hình
trainer.save_model("./phobert_gian_lan_final")
tokenizer.save_pretrained("./phobert_gian_lan_final")
print("[INFO] Đã lưu mô hình thành công! Hãy tích hợp vào AudioPipeline.")