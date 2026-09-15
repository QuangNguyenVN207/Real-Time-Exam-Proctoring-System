import matplotlib.pyplot as plt
import numpy as np

# SỐ LIỆU THỰC TẾ TỪ LOG CỦA BẠN (Từ Epoch 1 đến Epoch 9)
epochs = np.arange(1, 10)

# Trích xuất giá trị eval_loss và eval_f1 tương ứng với từng Epoch nguyên (1.0, 2.0, ..., 9.0)
val_loss = [0.339, 0.181, 0.352, 0.328, 0.123, 0.286, 0.410, 0.347, 0.317]
val_f1 =   [0.916, 0.955, 0.916, 0.929, 0.957, 0.942, 0.929, 0.942, 0.942]

# Trích xuất train_loss gần nhất với mỗi mốc kết thúc Epoch
train_loss = [0.377, 0.197, 0.033, 0.013, 0.005, 0.003, 0.002, 0.002, 0.002] 

fig, ax1 = plt.subplots(figsize=(8, 5))

# Vẽ đường Loss (Trục trái)
color1 = 'tab:red'
color2 = 'tab:orange'
ax1.set_xlabel('Epochs', fontweight='bold')
ax1.set_ylabel('Loss', color='black', fontweight='bold')
ax1.plot(epochs, train_loss, color=color1, marker='o', linewidth=2, label='Training Loss')
ax1.plot(epochs, val_loss, color=color2, marker='s', linewidth=2, label='Validation Loss')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(epochs)
ax1.grid(True, linestyle='--', alpha=0.6)

# Đánh dấu điểm Best Model (Early Stopping Checkpoint)
ax1.axvline(x=5, color='gray', linestyle=':', linewidth=2)
ax1.text(5.1, 0.2, 'Best Checkpoint\n(Epoch 5)', color='gray', fontweight='bold')

# Vẽ đường F1-Score (Trục phải)
ax2 = ax1.twinx()  
color3 = 'tab:blue'
ax2.set_ylabel('Macro F1-Score', color=color3, fontweight='bold')
ax2.plot(epochs, val_f1, color=color3, marker='^', linewidth=2, linestyle='-', label='Validation F1')
ax2.tick_params(axis='y', labelcolor=color3)
ax2.set_ylim(0.85, 1.0) # Scale lại trục để thấy rõ sự dao động đỉnh cao

# Gom chú thích (Legend)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

plt.title('PhoBERT Real-time Exam Intent Fine-Tuning', fontweight='bold')
plt.tight_layout()

# Lưu ra ảnh
plt.savefig('phobert_training_curve.png', dpi=300)
print("Đã lưu ảnh phobert_training_curve.png thành công!")