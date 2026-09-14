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
elif "OpenVINOExecutionProvider" in _AVAILABLE_PROVIDERS:
    _PROVIDERS = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    _DEVICE = "gpu"  # Hoặc "intel_gpu"
else:
    _PROVIDERS = ["CPUExecutionProvider"]
    _DEVICE = "cpu"


class FaceVerifier:
    """Xác minh danh tính mọi khuôn mặt trong frame so với ảnh thẻ sinh viên.

    Chỉ trả về cảnh báo khi có khuôn mặt KHÔNG khớp với bất kỳ ai trong
    cơ sở dữ liệu (data/student_faces/) — sinh viên hợp lệ luôn được bỏ qua.
    """

    def __init__(self, db_path="data/student_faces/", similarity_threshold=None,
                 identity_margin_threshold=None, det_size=None, model_name=None,
                 model_root=None, gallery_path=None, detection_threshold=None):
        self.db_path = db_path
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.face_similarity_threshold
        )
        self.identity_margin_threshold = (
            identity_margin_threshold
            if identity_margin_threshold is not None
            else settings.face_identity_margin_threshold
        )
        det_size = det_size or settings.face_det_size
        model_name = model_name or settings.face_model_name
        detection_threshold = (
            detection_threshold
            if detection_threshold is not None
            else settings.face_detection_threshold
        )

        _safe_print(f"[face_verify] Inference device: {_DEVICE}"
              + (" (không tìm thấy CUDA, chạy trên CPU)" if _DEVICE == "cpu" else ""))

        # Chỉ nạp đúng 2 model cần dùng (det_10g=SCRFD, w600k_r50=ArcFace)
        # thay vì insightface.app.FaceAnalysis (nạp cả 5 model trong bộ
        # buffalo_l rồi mới lọc bớt) — bộ đó có model landmark_3d_68 nặng
        # ~140MB không dùng tới, tốn RAM/thời gian khởi động vô ích.
        # Thay vì chỉ check cuda cho ctx_id, đổi lại để OpenVINO nhận diện provider đúng cách
        ctx_id = 0 if _DEVICE in ["cuda", "gpu"] else -1
        model_dir = ensure_available(
            "models",
            model_name,
            root=os.path.expanduser(model_root or "~/.insightface"),
        )

        self._det_model = model_zoo.get_model(osp.join(model_dir, "det_10g.onnx"), providers=_PROVIDERS)
        self._det_model.prepare(
            ctx_id,
            input_size=det_size,
            det_thresh=detection_threshold,
        )

        self._rec_model = model_zoo.get_model(osp.join(model_dir, "w600k_r50.onnx"), providers=_PROVIDERS)
        self._rec_model.prepare(ctx_id)

        self._known_names: list[str] = []
        self._known_vectors: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._faiss_index = None
        default_gallery = osp.join(self.db_path, "gallery_embeddings.npz")
        selected_gallery = gallery_path or default_gallery
        if osp.isfile(selected_gallery):
            self._load_gallery(selected_gallery)
        else:
            if gallery_path is not None:
                _safe_print(
                    f"[face_verify][WARNING] Không tìm thấy gallery: {selected_gallery}; "
                    "chuyển sang ảnh trong DB."
                )
            self._load_database()

    def _detect_faces(self, img: np.ndarray) -> list:
        """Chạy SCRFD tìm bbox, rồi ArcFace trích vector cho từng mặt —
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

        matrix = (
            np.array(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, 512), dtype=np.float32)
        )
        self._set_database(names, matrix)

        _safe_print(f"[face_verify] Đã nạp {len(self._known_names)} khuôn mặt từ {self.db_path}"
              + (" (dùng FAISS)" if self._faiss_index is not None else ""))

    def _load_gallery(self, gallery_path: str) -> None:
        """Nạp centroid gallery do benchmark_video.py sinh ra.

        Mỗi centroid tổng hợp nhiều frame enrollment của đúng một actor. Đây là
        cách triển khai tương ứng với protocol P1; dữ liệu sinh trắc học vẫn ở
        file cục bộ và không được tự động commit.
        """
        try:
            with np.load(gallery_path, allow_pickle=False) as gallery:
                names = [str(value) for value in gallery["actor_ids"].tolist()]
                vectors = np.asarray(gallery["centroids"], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(names):
                raise ValueError("actor_ids và centroids không cùng số hàng")
            if not names or len(set(names)) != len(names):
                raise ValueError("actor_ids phải không rỗng và không trùng")
            if vectors.shape[1] != 512 or not np.isfinite(vectors).all():
                raise ValueError("centroids phải là ma trận float hữu hạn Nx512")
            self._set_database(names, vectors)
            _safe_print(
                f"[face_verify] Đã nạp gallery centroid cho {len(names)} người từ "
                f"{gallery_path}" + (" (dùng FAISS)" if self._faiss_index is not None else "")
            )
        except Exception as exc:
            raise ValueError(f"Gallery face không hợp lệ: {gallery_path}: {exc}") from exc

    def _set_database(self, names: list[str], vectors: np.ndarray) -> None:
        """Chuẩn hoá gallery một lần và dựng chỉ mục tìm kiếm cosine."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise ValueError("Gallery chứa vector có norm bằng 0")
            matrix = matrix / norms
        self._known_names = list(names)
        self._known_vectors = matrix

        # Nếu số lượng sinh viên lớn, dùng FAISS (IndexFlatIP) để tìm kiếm nhanh hơn
        # thay vì nhân ma trận numpy tuần tự. Cosine similarity == inner product vì
        # normed_embedding đã được L2-normalize.
        if _HAS_FAISS and self._known_vectors.shape[0] > 0:
            self._faiss_index = faiss.IndexFlatIP(self._known_vectors.shape[1])
            self._faiss_index.add(self._known_vectors)
        else:
            self._faiss_index = None

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

    def _best_two_matches(
        self,
        query_vector: np.ndarray,
    ) -> tuple[int | None, float, float]:
        """Trả về (top1 index, top1 score, top2 score).

        Khi DB chỉ có một người, top2=-1 nên margin không vô tình chặn danh
        tính duy nhất. Query được chuẩn hoá phòng khi caller cung cấp vector thô.
        """
        if self._known_vectors.shape[0] == 0:
            return None, -1.0, -1.0
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            return None, -1.0, -1.0
        query = query / norm
        count = min(2, self._known_vectors.shape[0])
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.reshape(1, -1), count)
            top1_index = int(indices[0][0])
            top1_score = float(scores[0][0])
            top2_score = float(scores[0][1]) if count == 2 else -1.0
            return top1_index, top1_score, top2_score
        scores = self._known_vectors @ query
        order = np.argsort(scores)[::-1]
        top1_index = int(order[0])
        top1_score = float(scores[top1_index])
        top2_score = float(scores[int(order[1])]) if count == 2 else -1.0
        return top1_index, top1_score, top2_score

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
                # print(f"[DEBUG FACE] Điểm số so khớp thực tế: {best_score} (Ngưỡng yêu cầu: {self.similarity_threshold})") # <--- Thêm dòng này
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
            idx, score, runner_up_score = self._best_two_matches(face.normed_embedding)
            margin = score - runner_up_score
            if (
                idx is None
                or score < self.similarity_threshold
                or margin < self.identity_margin_threshold
            ):
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
            query_vector = np.asarray(face.normed_embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query_vector))
            if query_norm <= 1e-12:
                return None
            query_vector = query_vector / query_norm

            expected_idx = self._known_names.index(expected_student_id)
            best_idx, best_score, runner_up_score = self._best_two_matches(query_vector)
            best_name = self._known_names[best_idx] if best_idx is not None else None
            expected_score = float(self._known_vectors[expected_idx] @ query_vector)
            margin = best_score - runner_up_score
            if (
                best_idx == expected_idx
                and best_score >= self.similarity_threshold
                and margin >= self.identity_margin_threshold
            ):
                return None  # đúng người, đủ điểm và không mơ hồ với runner-up
            if best_score < self.similarity_threshold:
                decision_reason = "low_similarity"
            elif margin < self.identity_margin_threshold:
                decision_reason = "ambiguous_identity"
            else:
                decision_reason = "wrong_top1_identity"

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
                    "runner_up_score": round(runner_up_score, 4) if best_name is not None else None,
                    "identity_margin": round(margin, 4) if best_name is not None else None,
                    "required_similarity": self.similarity_threshold,
                    "required_margin": self.identity_margin_threshold,
                    "decision_reason": decision_reason,
                    "bbox": absolute_bbox,
                },
            }
        except Exception as e:
            _safe_print(f"[face_verify][ERROR] verify_assigned_identity thất bại: {e}")
            return None
