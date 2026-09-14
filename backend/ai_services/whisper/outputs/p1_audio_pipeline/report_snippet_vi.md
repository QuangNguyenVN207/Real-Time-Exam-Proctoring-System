# Kết quả kiểm thử pipeline audio

Pipeline sử dụng **PhoWhisper-small pretrained** để nhận dạng tiếng Việt và checkpoint **PhoBERT step 235 (epoch 5)** để phân loại transcript. Ngưỡng xác suất gian lận dùng khi vận hành là **0,60**; luật từ khóa mức `medium` hoặc `high` có thể kích hoạt cảnh báo trực tiếp.

Trên 40 mẫu audio duy nhất theo SHA-256 (21 gian lận, 19 bình thường), pipeline fusion đạt accuracy = precision = recall = F1 = **1.000** với ma trận TP=21, TN=19, FP=0, FN=0. Tỷ lệ xử lý thành công là **100.0%**. Hai đường dẫn trùng nội dung byte đã được loại khỏi metric chính.

Thử nghiệm tách thành phần cho thấy luật từ khóa đạt recall **0.714** (15/21 mẫu gian lận), trong khi PhoBERT và fusion đều đạt recall **1.000**. PhoBERT vì vậy bổ sung 6 mẫu không được luật từ khóa kích hoạt trong tập này.

Sweep trên cùng tập mẫu cho F1 tối đa trong khoảng ngưỡng **0.02–0.99**; ngưỡng 0,60 được giữ nguyên vì nằm trong vùng ổn định và là cấu hình vận hành đã chọn. Đây chỉ là phân tích độ nhạy, không phải tối ưu trên một tập test độc lập.

**Giới hạn:** tập thử nhỏ, các câu ngắn và phân tách khá dễ; nhãn chỉ cho hành vi cuối, không có transcript chuẩn để tính WER/CER. Vì vậy kết quả trên được dùng để xác nhận tính đúng đắn và khả năng tái lập của pipeline, không dùng để tuyên bố khả năng tổng quát hóa. Thời gian CPU trung bình **15.69 giây/tệp** trong lần chạy này cũng chưa đáp ứng suy luận thời gian thực nếu xử lý tuần tự trên CPU.
