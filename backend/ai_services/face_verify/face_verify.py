import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from backend.core.config import settings

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class FaceVerifier:
    """Xác minh danh tính mọi khuôn mặt trong frame so với ảnh thẻ sinh viên.

    Chỉ trả về cảnh báo khi có khuôn mặt KHÔNG khớp với bất kỳ ai trong
    cơ sở dữ liệu (data/student_faces/) — sinh viên hợp lệ luôn được bỏ qua.
    """

    def __init__(self, db_path="data/student_faces/", similarity_threshold=None,
                 det_size=None, model_name=None):
        self.db_path = db_path
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.face_similarity_threshold
        )
        det_size = det_size or settings.face_det_size
        model_name = model_name or settings.face_model_name

        # RetinaFace (detection) + ArcFace (recognition, 512D, đã L2-normalize)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"], providers=providers)
        self._app.prepare(ctx_id=0, det_size=det_size)

        self._known_names: list[str] = []
        self._known_vectors: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._faiss_index = None
        self._load_database()

    def _load_database(self):
        """Quét db_path, trích Face Vector của từng sinh viên và nạp vào RAM."""
        os.makedirs(self.db_path, exist_ok=True)

        names, vectors = [], []
        for filename in sorted(os.listdir(self.db_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filepath = os.path.join(self.db_path, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"[face_verify][WARNING] Không đọc được ảnh: {filepath}")
                continue

            faces = self._app.get(image)
            if not faces:
                print(f"[face_verify][WARNING] Không tìm thấy khuôn mặt trong ảnh thẻ: {filepath}")
                continue

            # Ảnh thẻ lẽ ra chỉ có 1 người -> nếu detect nhầm nhiều mặt thì lấy mặt lớn nhất
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            names.append(os.path.splitext(filename)[0])
            vectors.append(face.normed_embedding)

        self._known_names = names
        self._known_vectors = np.array(vectors, dtype=np.float32) if vectors else np.empty((0, 512), dtype=np.float32)

        # Nếu số lượng sinh viên lớn, dùng FAISS (IndexFlatIP) để tìm kiếm nhanh hơn
        # thay vì nhân ma trận numpy tuần tự. Cosine similarity == inner product vì
        # normed_embedding đã được L2-normalize.
        if _HAS_FAISS and self._known_vectors.shape[0] > 0:
            self._faiss_index = faiss.IndexFlatIP(self._known_vectors.shape[1])
            self._faiss_index.add(self._known_vectors)
        else:
            self._faiss_index = None

        print(f"[face_verify] Đã nạp {len(self._known_names)} khuôn mặt từ {self.db_path}"
              + (" (dùng FAISS)" if self._faiss_index is not None else ""))

    def _best_match_score(self, query_vector: np.ndarray) -> float:
        """Điểm Cosine Similarity cao nhất giữa query_vector và toàn bộ DB."""
        if self._known_vectors.shape[0] == 0:
            return -1.0
        if self._faiss_index is not None:
            scores, _ = self._faiss_index.search(query_vector.reshape(1, -1), 1)
            return float(scores[0][0])
        return float(np.max(self._known_vectors @ query_vector))

    def verify_face(self, frame, timestamp):
        """
        Input: frame (Numpy array, BGR) và timestamp của khung hình.
        Output: None nếu mọi khuôn mặt đều khớp với DB (hoặc không có mặt nào),
                hoặc dict cảnh báo khi có khuôn mặt không khớp với ai trong DB.
        """
        faces = self._app.get(frame)
        if not faces:
            return None

        for face in faces:
            best_score = self._best_match_score(face.normed_embedding)
            if best_score < self.similarity_threshold:
                bbox = [int(v) for v in face.bbox]
                return {
                    "module": "face_verify",
                    "status": "alert",
                    "timestamp": timestamp,
                    "message": "unauthorized_person",
                    "details": {
                        "similarity_score": round(best_score, 4),
                        "unauthorized_bbox": bbox,
                    },
                }

        return None
