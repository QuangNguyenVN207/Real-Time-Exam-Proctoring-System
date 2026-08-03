import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize

class PhobertService:
    def __init__(self, model_path=None):
        print("[PhoBERT] Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PhoBERT] Device: {self.device}")
        
        # 1. Xử lý đường dẫn tuyệt đối để không bao giờ bị lỗi do khác thư mục đứng
        if model_path is None:
            # Lấy đường dẫn của chính file phobert_service.py, sau đó trỏ vào thư mục "weights"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "weights")
            
        # 2. Kiểm tra xem thư mục weights đã có file cấu hình (model đã được train) chưa
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            print(f"[PhoBERT] CẢNH BÁO: Chưa có model tại {model_path}!")
            print("[PhoBERT] Đang tải model gốc 'vinai/phobert-base-v2'...")
            model_path = "vinai/phobert-base-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # CHỈNH VỀ 2 NHÃN
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            
        self.model.eval()
        
        # CHỈNH LẠI LABEL MAP: 0 là Normal, 1 là Cheating
        self.id2label = {0: "Normal", 1: "Cheating"}
        self.MAX_LEN = 64
        print("[PhoBERT] Model loaded successfully!")

    def _preprocess(self, text):
        # CHÚ Ý: Phải tách từ giống hệt lúc train
        # "chỉ tao câu hai" -> "chỉ tao câu hai" (không có gạch dưới) 
        # -> word_tokenize format="text" -> "chỉ tao câu_hai"
        return word_tokenize(text.lower().strip(), format="text")

    def predict(self, text):
        """
        Nhận vào chuỗi văn bản (đã qua PhoWhisper), trả về phân loại.
        """
        if not text:
            return {"label": "Normal", "confidence": 1.0, "all_probs": {}}

        processed_text = self._preprocess(text)
        
        inputs = self.tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.MAX_LEN
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Áp dụng softmax để quy đổi logits thành % xác suất
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Lấy nhãn có xác suất cao nhất
        predicted_class_id = int(np.argmax(probs))
        predicted_label = self.id2label[predicted_class_id]
        confidence = float(probs[predicted_class_id])
        
        # Đóng gói toàn bộ xác suất để Decision Fusion tự quyết định
        all_probs = {self.id2label[i]: float(probs[i]) for i in range(len(self.id2label))}
        
        return {
            "text_processed": processed_text,
            "label": predicted_label,
            "confidence": confidence,
            "all_probs": all_probs
        }

# --- Test nhanh khi chạy trực tiếp file ---
if __name__ == "__main__":
    service = PhobertService()
    
    test_cases = [
        "thầy ơi cho em hỏi câu số 3",
        "ê nhắc tao câu năm m đáp án gì đấy",
        "hôm nay trời nóng quá"
    ]
    
    for t in test_cases:
        res = service.predict(t)
        print(f"\nText: {t}")
        print(f"Result: {res}")