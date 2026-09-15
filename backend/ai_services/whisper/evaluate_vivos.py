import os
import time
from jiwer import wer
# Xóa dòng cũ và thay bằng dòng này:
from backend.ai_services.whisper.audio_pipeline import AudioPipeline

def run_benchmark(vivos_test_dir, max_files=50):
    """
    Hàm đo lường độ chính xác của hệ thống AI Giám thị bằng VIVOS dataset.
    - vivos_test_dir: Đường dẫn đến thư mục 'test' của VIVOS vừa giải nén.
    - max_files: Số lượng file muốn test (Để 50 file chạy cho nhanh, nếu muốn test toàn bộ thì để None).
    """
    prompts_file = os.path.join(vivos_test_dir, "prompts.txt")
    waves_dir = os.path.join(vivos_test_dir, "waves")

    if not os.path.exists(prompts_file):
        print(f"❌ Không tìm thấy file prompts.txt tại {prompts_file}")
        return

    # Khởi tạo siêu AI của bạn (Gồm cả bộ lọc AGC, Bandpass, VAD và PhoWhisper)
    print("⏳ Đang nạp hệ thống AI Giám thị...")
    pipeline = AudioPipeline()
    print("✅ Nạp xong! Bắt đầu chấm điểm...\n")

    ground_truths = []
    predictions = []
    
    # 1. Đọc file đáp án chuẩn của VIVOS
    with open(prompts_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    count = 0
    start_time = time.time()

    # 2. Bắt đầu vòng lặp cho AI nghe và chấm điểm
    for line in lines:
        if max_files and count >= max_files:
            break
            
        parts = line.strip().split(" ", 1)
        if len(parts) < 2:
            continue
            
        audio_id = parts[0]
        true_text = parts[1].lower() # Chữ chuẩn từ VIVOS
        
        # Tìm đường dẫn file âm thanh tương ứng (VIVOS chia theo thư mục tên loa/speaker)
        speaker_id = audio_id.split("_")[0]
        audio_path = os.path.join(waves_dir, speaker_id, f"{audio_id}.wav")
        
        if not os.path.exists(audio_path):
            continue

        try:
            # Cho AI của bạn xử lý file âm thanh
            result = pipeline.process(audio_path)
            
            # Lấy kết quả AI nghe được (chuyển về chữ thường để so sánh công bằng)
            pred_text = result.get("transcription", "").lower()
            
            # Lưu lại để chấm điểm tổng
            ground_truths.append(true_text)
            predictions.append(pred_text)
            
            # In ra màn hình để theo dõi trực tiếp
            print(f"[{count+1}] File: {audio_id}")
            print(f"   Chuẩn : {true_text}")
            print(f"   AI nghe: {pred_text}")
            print("-" * 50)
            
            count += 1
            
        except Exception as e:
            print(f"Lỗi đọc file {audio_id}: {e}")

    # 3. Tổng kết và tính điểm WER
    print("\n" + "="*50)
    print("📊 BÁO CÁO KẾT QUẢ BENCHMARK (ĐO LƯỜNG ĐỘ CHÍNH XÁC)")
    print("="*50)
    
    if len(ground_truths) > 0:
        error_rate = wer(ground_truths, predictions)
        accuracy = (1 - error_rate) * 100
        
        print(f"Tổng số file đã test : {count} files")
        print(f"Thời gian chạy       : {time.time() - start_time:.2f} giây")
        print(f"Tỷ lệ lỗi từ (WER)   : {error_rate:.4f} ({error_rate * 100:.2f}%)")
        print(f"⭐ ĐỘ CHÍNH XÁC ƯỚC TÍNH: {accuracy:.2f}%")
        
        if accuracy > 85:
            print("🚀 Đánh giá: Xuất sắc! Phù hợp để làm hệ thống nhận diện thực tế.")
        elif accuracy > 70:
            print("👍 Đánh giá: Tốt! Nhưng có thể tối ưu thêm thuật toán lọc nhiễu AGC.")
        else:
            print("⚠️ Đánh giá: AI đang gặp khó khăn. Cần kiểm tra lại bộ lọc đầu vào.")
    else:
        print("Không có dữ liệu để chấm điểm.")

if __name__ == "__main__":
    # Thay đường dẫn này bằng thư mục 'test' trong thư mục VIVOS bạn vừa giải nén
    VIVOS_TEST_FOLDER = r"D:\vivos\test" 
    
    # Đang set max_files = 50 để chạy thử cho nhanh. Nếu muốn test thật, hãy xóa tham số này đi.
    run_benchmark(VIVOS_TEST_FOLDER, max_files=20)