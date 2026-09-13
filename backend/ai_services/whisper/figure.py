import matplotlib.pyplot as plt
import numpy as np

# Số liệu thực tế từ lần chạy Benchmark VIVOS của bạn
accuracy = 75.25
wer = 24.75

# Thiết lập phong cách biểu đồ chuyên nghiệp
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ biểu đồ cột
categories = ['Độ chính xác\n(Accuracy)', 'Tỷ lệ lỗi từ\n(WER)']
values = [accuracy, wer]
colors = ['#27ae60', '#e74c3c']  # Xanh lá cho chính xác, Đỏ cho lỗi

bars = ax.bar(categories, values, color=colors, width=0.5)

# Trang trí số liệu trực tiếp lên cột
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, 
            f'{yval}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

# Căn chỉnh tiêu đề và trục
ax.set_ylim(0, 100)
ax.set_ylabel('Phần trăm (%)', fontsize=12, fontweight='bold')
ax.set_title('Đánh giá độ chính xác của AI PhoWhisper\n(Kiểm thử trên tập dữ liệu VIVOS)', 
             fontsize=14, fontweight='bold', pad=20)

# Lưu thành file ảnh nền trong suốt cực xịn
plt.tight_layout()
plt.savefig('vivos_benchmark_figure.png', dpi=300, transparent=False)
print("✅ Đã xuất thành công biểu đồ: vivos_benchmark_figure.png")

# Hiển thị lên màn hình
plt.show()