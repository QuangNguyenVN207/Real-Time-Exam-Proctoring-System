import os
import cv2
import json
import time
import numpy as np
import sqlite3
import face_recognition

class FaceVerifier:
    def __init__(self, db_path='data/student_faces/', log_img_path='logs/images/', sqlite_db='logs/proctoring.db'):
        self.db_path = db_path
        self.log_img_path = log_img_path
        self.sqlite_db = sqlite_db
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Ngưỡng Cosine Similarity (Cần tinh chỉnh trong thực tế, thường > 0.6 là cùng một người)
        self.threshold = 0.6 
        
        self._init_directories()
        self._init_sqlite()
        self._load_database()

    def _init_directories(self):
        """Tạo các thư mục cần thiết nếu chưa tồn tại"""
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.log_img_path, exist_ok=True)

    def _init_sqlite(self):
        """Khởi tạo database SQLite để lưu log vi phạm"""
        os.makedirs(os.path.dirname(self.sqlite_db), exist_ok=True)
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                message TEXT,
                similarity REAL,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _load_database(self):
        """Trích xuất Face Vector từ ảnh thẻ trong thư mục (Chạy 1 lần khi khởi động)"""
        print("[INFO] Đang nạp cơ sở dữ liệu khuôn mặt...")
        for filename in os.listdir(self.db_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.db_path, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    # Lấy tên sinh viên từ tên file (vd: MSSV.jpg -> MSSV)
                    self.known_face_names.append(os.path.splitext(filename)[0])
        print(f"[INFO] Đã nạp {len(self.known_face_encodings)} khuôn mặt.")

    def _cosine_similarity(self, vec1, vec2):
        """Tính Cosine Similarity giữa 2 vector 128D"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

    def process_frame(self, frame_np):
        """
        Input: Khung hình cắt từ camera (Numpy array dạng BGR của OpenCV)
        Output: JSON string chứa kết quả
        """
        current_time = int(time.time())
        # face_recognition yêu cầu ảnh RGB
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        
        # Tìm các khuôn mặt trong khung hình
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            return json.dumps({
                "module": "face_verify",
                "status": "warning",
                "message": "Không tìm thấy khuôn mặt",
                "cosine_similarity": 0.0,
                "timestamp": current_time
            })

        # Xử lý khuôn mặt đầu tiên tìm thấy (giả định thi chỉ có 1 người/camera)
        current_encoding = face_encodings[0]
        
        max_similarity = 0.0
        best_match_name = "Unknown"

        # So sánh với database cục bộ
        for i, known_encoding in enumerate(self.known_face_encodings):
            sim = self._cosine_similarity(current_encoding, known_encoding)
            if sim > max_similarity:
                max_similarity = sim
                best_match_name = self.known_face_names[i]

        # Đánh giá kết quả
        if max_similarity >= self.threshold:
            status = "ok"
            message = f"Đúng thí sinh: {best_match_name}"
            # Không cần lưu log cho trường hợp bình thường
        else:
            status = "alert"
            message = "Xuất hiện người lạ"
            self._handle_violation(frame_np, max_similarity, message, current_time)

        # Trả về chuỗi JSON để core backend đẩy qua WebSocket
        result = {
            "module": "face_verify",
            "status": status,
            "message": message,
            "cosine_similarity": float(round(max_similarity, 4)),
            "timestamp": current_time
        }
        return json.dumps(result)

    def _handle_violation(self, frame_np, similarity, message, timestamp):
        """Lưu ảnh và ghi log vào SQLite khi có vi phạm"""
        img_filename = f"violation_{timestamp}.jpg"
        img_path = os.path.join(self.log_img_path, img_filename)
        
        # 1. Lưu ảnh cục bộ
        cv2.imwrite(img_path, frame_np)
        
        # 2. Ghi log vào SQLite
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO violations (timestamp, message, similarity, image_path)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, message, float(similarity), img_path))
        conn.commit()
        conn.close()