class DecisionFusionService:
    def __init__(
        self, 
        ai_cheating_threshold=0.60  # Ngưỡng để PhoBERT bắt thêm (vớt lỗi)
    ):
        self.ai_cheating = ai_cheating_threshold

    def fuse(self, transcript, rule_label, ai_result):
        ai_label = ai_result.get("label", "Normal")
        probs = ai_result.get("all_probs", {})
        
        final_label = "Normal"
        reason = ""

        # Đưa rule_label về chữ thường để dễ so sánh, triệt tiêu lỗi viết hoa/thường
        rule_lower = str(rule_label).lower()

        # =========================================================
        # CHIẾN THUẬT MỚI: TÔN TRỌNG TUYỆT ĐỐI KEYWORD DETECTOR
        # =========================================================
        # Hỗ trợ cả 2 chuẩn nhãn: "high/medium" của Keyword cũ, hoặc "cheating/suspicious"
        if rule_lower in ["high", "medium", "cheating", "suspicious"]:
            final_label = "Cheating"
            reason = f"Rule-based Triggered (Mức độ: {rule_label})."

        # =========================================================
        # CHỈ DÙNG PHO-BERT ĐỂ BẮT THÊM KHI KEYWORD BỎ LỌT
        # =========================================================
        elif rule_lower in ["safe", "low", "normal"]:
            # Nếu Keyword an toàn, hỏi ý kiến AI xem có ẩn ý không
            if probs.get("Cheating", 0) >= self.ai_cheating:
                final_label = "Cheating"
                reason = "AI Catch: Keyword bỏ sót, AI bọc lót thành công!"
            else:
                final_label = "Normal"
                reason = "Đồng thuận an toàn."

        return {
            "transcript": transcript,
            "final_label": final_label,
            "rule_label": rule_label,
            "ai_label": ai_label,
            "ai_probs": probs,
            "fusion_reason": reason
        }