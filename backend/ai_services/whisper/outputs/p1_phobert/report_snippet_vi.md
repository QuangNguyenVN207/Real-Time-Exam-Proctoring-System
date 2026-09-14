# Kết quả kiểm thử PhoBERT dùng cho thuyết minh

- Thành phần được fine-tune: `vinai/phobert-base-v2`, phân loại hai lớp Normal/Cheating.
- Thành phần nhận dạng tiếng nói: `vinai/PhoWhisper-small` pretrained, không fine-tune trong công trình này.
- Dữ liệu hợp lệ: 443 câu; train 376, validation 67 theo seed 42 và stratify.
- Huấn luyện đặt tối đa 15 epoch và dừng sớm ở epoch 9. Checkpoint tốt nhất là step 235 (epoch 5), F1 log = 0.9577.
- Trọng số `model.safetensors` hiện có khớp weighted eval loss và toàn bộ bốn metric của epoch 5, step 235 (sai khác lớn nhất 3.63e-07); đây là trọng số được kiểm thử và đóng gói.
- Cấu hình vận hành dùng threshold 0.60: accuracy 0.9701, precision 0.9444, recall 1.0000, F1 0.9714 (TP=34, TN=31, FP=2, FN=0).
- Sweep trên chính validation có điểm F1 cao nhất tại 0.71: accuracy 0.9851, precision 0.9714, recall 1.0000, F1 0.9855. Điểm này chỉ là kết quả phân tích độ nhạy, không thay thế cấu hình vận hành 0.60.
- ROC AUC = 0.9724; average precision = 0.9393; EER xấp xỉ = 0.0152.
- Có 2 hàng validation trùng nguyên văn với train. Sau khi loại các hàng đó để phân tích độ nhạy (không phải retrain), F1 = 0.9851 trên 65 câu.

Lưu ý: threshold tốt nhất được chọn và đo trên cùng validation split, vì vậy chỉ dùng để giải thích lựa chọn cấu hình. Không gọi đây là kết quả locked test hay khả năng tổng quát trên dữ liệu độc lập.
