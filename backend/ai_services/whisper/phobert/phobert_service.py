import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# optimum.intel chỉ import khi cần (máy NVIDIA không bắt buộc cài optimum[openvino])
from backend.core.device import resolve_device
from underthesea import word_tokenize

class PhobertService:
    def __init__(self, model_path=None):
        print("[PhoBERT] Loading model...")
        # NVIDIA -> PyTorch CUDA | Intel iGPU -> OpenVINO GPU | còn lại -> OpenVINO CPU (hoặc PyTorch CPU nếu thiếu optimum)
        # Ép thủ công: PROCTOR_DEVICE=cuda|intel|cpu
        info = resolve_device(None)
        self.device_info = info
        self._OVModel = None
        if info.kind == "cuda":
            self.run_mode = "cuda"
            self.device = torch.device("cuda")
        else:
            try:
                from optimum.intel import OVModelForSequenceClassification
                self._OVModel = OVModelForSequenceClassification
                self.run_mode = "openvino"
                self.device = "GPU" if info.kind == "intel" else "CPU"
            except Exception as e:
                print(f"[PhoBERT] Không dùng được optimum.intel ({e}) -> PyTorch CPU")
                self.run_mode = "torch_cpu"
                self.device = torch.device("cpu")

        print(f"[PhoBERT] Thiết bị được chọn: {self.device} (Mode: {self.run_mode})")
        
        # 1. Xử lý đường dẫn tuyệt đối để không bao giờ bị lỗi do khác thư mục đứng
        if model_path is None:
            # Lấy đường dẫn của chính file phobert_service.py, sau đó trỏ vào thư mục "weights"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "weights")
            
        # 2. Kiểm tra xem thư mục weights đã có file cấu hình (model đã được train) chưa
        config_file = os.path.join(model_path, "config.json")

        is_pretrained = not os.path.exists(config_file)
        if is_pretrained:
            print(f"[PhoBERT] CẢNH BÁO: Chưa có model tại {model_path}!")
            print("[PhoBERT] Đang tải model gốc 'vinai/phobert-base-v2'...")
            model_path = "vinai/phobert-base-v2"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # CHỈ truyền num_labels khi tải model gốc. Model đã train có sẵn num_labels trong config.json;
        # truyền num_labels=None làm transformers 5.x lỗi "'NoneType' object cannot be interpreted as an integer".
        load_kwargs = {"num_labels": 2} if is_pretrained else {}

        if self.run_mode == "openvino":
            self.model = self._load_openvino(model_path, is_pretrained, load_kwargs)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, **load_kwargs
            ).to(self.device)

        self.model.eval()
        
        # CHỈNH LẠI LABEL MAP: 0 là Normal, 1 là Cheating
        self.id2label = {0: "Normal", 1: "Cheating"}
        self.MAX_LEN = 64
        print("[PhoBERT] Model loaded successfully!")

    def _load_openvino(self, model_path, is_pretrained, load_kwargs):
        """Nạp bản OpenVINO. Export 1 lần rồi cache cạnh thư mục model (<model>_openvino) để khởi động nhanh."""
        ov_dir = None if is_pretrained else model_path.rstrip("/\\") + "_openvino"
        if ov_dir is not None:
            xml = os.path.join(ov_dir, "openvino_model.xml")
            if os.path.exists(xml):
                newest_src = max(
                    (os.path.getmtime(os.path.join(model_path, f))
                     for f in os.listdir(model_path)
                     if f.endswith((".safetensors", ".bin", ".json"))),
                    default=0,
                )
                if os.path.getmtime(xml) >= newest_src:  # model chưa bị train lại sau lần export
                    try:
                        return self._OVModel.from_pretrained(ov_dir, device=self.device, export=False)
                    except Exception as e:
                        print(f"[PhoBERT] Cache OpenVINO lỗi ({e}) -> export lại")
        model = self._OVModel.from_pretrained(
            model_path, device=self.device, export=True, **load_kwargs
        )
        if ov_dir is not None:
            try:
                model.save_pretrained(ov_dir)
                print(f"[PhoBERT] Đã cache bản OpenVINO tại {ov_dir}")
            except Exception as e:
                print(f"[PhoBERT] Không lưu được cache OpenVINO: {e}")
        return model

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
        )               # .to(self.device)
        # Nếu đang chạy CUDA, cần đẩy dữ liệu (inputs) lên card đồ họa
        if hasattr(self, 'run_mode') and self.run_mode == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

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