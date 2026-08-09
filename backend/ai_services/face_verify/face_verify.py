import os
import os.path as osp
import sys
import logging

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from insightface.utils import ensure_available

from backend.core.config import settings

ort.set_default_logger_severity(3)
logging.getLogger("insightface").setLevel(logging.ERROR)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


def _safe_print(message: str) -> None:
    """print() thường, nhưng không crash trên console Windows dùng codepage
    không phải UTF-8 (cp1252/cp437...) khi log có tiếng Việt có dấu."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(message.encode(encoding, errors="replace").decode(encoding))


_AVAILABLE_PROVIDERS = ort.get_available_providers()
if "CUDAExecutionProvider" in _AVAILABLE_PROVIDERS:
    _PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _DEVICE = "cuda"
else:
    _PROVIDERS = ["CPUExecutionProvider"]
    _DEVICE = "cpu"


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

        _safe_print(f"[face_verify] Inference device: {_DEVICE}"
              + (" (không tìm thấy CUDA, chạy trên CPU)" if _DEVICE == "cpu" else ""))

        # Chỉ nạp đúng 2 model cần dùng (det_10g=RetinaFace, w600k_r50=ArcFace)
        # thay vì insightface.app.FaceAnalysis (nạp cả 5 model trong bộ
        # buffalo_l rồi mới lọc bớt) — bộ đó có model landmark_3d_68 nặng
        # ~140MB không dùng tới, tốn RAM/thời gian khởi động vô ích.
        ctx_id = 0 if _DEVICE == "cuda" else -1
        model_dir = ensure_available(
            "models",
            model_name,
            root=os.path.expanduser("~/.insightface"),
        )

        self._det_model = model_zoo.get_model(osp.join(model_dir, "det_10g.onnx"), providers=_PROVIDERS)
        self._det_model.prepare(ctx_id, input_size=det_size, det_thresh=0.5)

        self._rec_model = model_zoo.get_model(osp.join(model_dir, "w600k_r50.onnx"), providers=_PROVIDERS)
        self._rec_model.prepare(ctx_id)

        self._known_names: list[str] = []
        self._known_vectors: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._faiss_index = None
        self._load_database()

    def _detect_faces(self, img: np.ndarray) -> list:
        """Chạy RetinaFace tìm bbox, rồi ArcFace trích vector cho từng mặt —
        tương đương FaceAnalysis.get() nhưng chỉ dùng 2 model đã nạp riêng."""
        bboxes, kpss = self._det_model.detect(img, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return []
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, 0:4], kps=(kpss[i] if kpss is not None else None), det_score=bboxes[i, 4])
            self._rec_model.get(img, face)
            faces.append(face)
        return faces

    def _load_database(self):
        """Quét db_path, trích Face Vector của từng sinh viên và nạp vào RAM."""
        os.makedirs(self.db_path, exist_ok=True)

        names, vectors = [], []
        for filename in sorted(os.listdir(self.db_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            filepath = os.path.join(self.db_path, filename)
            try:
                image = cv2.imread(filepath)
                if image is None:
                    _safe_print(f"[face_verify][WARNING] Không đọc được ảnh: {filepath}")
                    continue

                faces = self._detect_faces(image)
                if not faces:
                    _safe_print(f"[face_verify][WARNING] Không tìm thấy khuôn mặt trong ảnh thẻ: {filepath}")
                    continue

                # Ảnh thẻ lẽ ra chỉ có 1 người -> nếu detect nhầm nhiều mặt thì lấy mặt lớn nhất
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                names.append(os.path.splitext(filename)[0])
                vectors.append(face.normed_embedding)
            except Exception as e:
                _safe_print(f"[face_verify][ERROR] Lỗi khi xử lý {filepath}: {e}")
                continue

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

        _safe_print(f"[face_verify] Đã nạp {len(self._known_names)} khuôn mặt từ {self.db_path}"
              + (" (dùng FAISS)" if self._faiss_index is not None else ""))

    def _best_match(self, query_vector: np.ndarray) -> tuple[int | None, float]:
        """(index, score) của vector khớp cao nhất trong DB. index=None nếu DB rỗng."""
        if self._known_vectors.shape[0] == 0:
            return None, -1.0
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vector.reshape(1, -1), 1)
            return int(indices[0][0]), float(scores[0][0])
        sims = self._known_vectors @ query_vector
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> np.ndarray:
        """Cắt vùng bbox [x1,y1,x2,y2] ra khỏi frame, giới hạn trong biên ảnh."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return frame[y1:y2, x1:x2]

    def verify_face(self, frame, timestamp):
        """
        Input: frame (Numpy array, BGR) và timestamp của khung hình.
        Output: None nếu mọi khuôn mặt đều khớp với DB (hoặc không có mặt nào),
                hoặc dict cảnh báo khi có khuôn mặt không khớp với ai trong DB.
        """
        try:
            faces = self._detect_faces(frame)
            if not faces:
                return None

            for face in faces:
                _, best_score = self._best_match(face.normed_embedding)
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
        except Exception as e:
            # Frame lỗi/vỡ nét không được làm sập luồng AI đa luồng của server
            _safe_print(f"[face_verify][ERROR] verify_face thất bại: {e}")
            return None

    def identify(self, frame, bbox=None):
        """Nhận diện khuôn mặt khớp nhất trong DB (dùng để tự động gán student_id
        cho 1 track mới từ module pose_gaze/tracking — thay vì gán tay qua API).

        Input: frame (Numpy array, BGR). bbox=[x1,y1,x2,y2] tuỳ chọn để chỉ xét
               vùng của 1 người cụ thể (toạ độ từ TrackedPerson.bbox); None thì
               lấy khuôn mặt lớn nhất trong cả frame.
        Output: (student_id, similarity_score) nếu điểm >= threshold, ngược lại None.
        """
        try:
            region = self._crop(frame, bbox) if bbox is not None else frame
            if region.size == 0:
                return None

            faces = self._detect_faces(region)
            if not faces:
                return None

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            idx, score = self._best_match(face.normed_embedding)
            if idx is None or score < self.similarity_threshold:
                return None
            return self._known_names[idx], round(score, 4)
        except Exception as e:
            _safe_print(f"[face_verify][ERROR] identify thất bại: {e}")
            return None

    def verify_assigned_identity(self, frame, bbox, expected_student_id, timestamp):
        """Xác minh khuôn mặt trong bbox (toạ độ 1 track, từ pose_gaze/tracking)
        có ĐÚNG là expected_student_id đã được gán cho track đó không.

        Khác với verify_face (chỉ hỏi "có nằm trong DB không"), hàm này bắt được
        ca thi hộ giữa các sinh viên ĐỀU hợp lệ trong DB — ví dụ sinh viên B
        (không phải người lạ) ngồi vào bàn của sinh viên A. Trường hợp này lọt
        qua cả giám thị (không ai là người lạ) lẫn verify_face (B vẫn khớp DB,
        chỉ là khớp sai người).

        Output: None nếu đúng người được gán, dict cảnh báo "identity_mismatch"
                nếu không khớp (kể cả khi không có khuôn mặt nào được phát hiện
                trong bbox đó cũng được coi là bất thường và bỏ qua — trả None,
                vì có thể do góc quay/che khuất tạm thời, không đủ cơ sở kết luận).
        """
        try:
            if expected_student_id not in self._known_names:
                return None  # id chưa có ảnh gốc trong DB, không đủ cơ sở xác minh

            region = self._crop(frame, bbox)
            if region.size == 0:
                return None

            faces = self._detect_faces(region)
            if not faces:
                return None

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            query_vector = face.normed_embedding

            expected_idx = self._known_names.index(expected_student_id)
            expected_score = float(self._known_vectors[expected_idx] @ query_vector)
            if expected_score >= self.similarity_threshold:
                return None  # đúng người được gán cho track này

            best_idx, best_score = self._best_match(query_vector)
            best_name = self._known_names[best_idx] if best_idx is not None else None

            # Quy đổi bbox khuôn mặt (toạ độ trong vùng crop) về toạ độ frame gốc
            x1, y1 = int(bbox[0]), int(bbox[1])
            fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
            absolute_bbox = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]

            return {
                "module": "face_verify",
                "status": "alert",
                "timestamp": timestamp,
                "message": "identity_mismatch",
                "details": {
                    "expected_student_id": expected_student_id,
                    "matched_student_id": best_name,
                    "expected_score": round(expected_score, 4),
                    "matched_score": round(best_score, 4) if best_name is not None else None,
                    "bbox": absolute_bbox,
                },
            }
        except Exception as e:
            _safe_print(f"[face_verify][ERROR] verify_assigned_identity thất bại: {e}")
            return None
