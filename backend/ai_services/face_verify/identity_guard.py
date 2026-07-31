"""Cầu nối giữa face_verify và pose_gaze/tracking (TrackingManager).

Bối cảnh: phòng thi đã có giám thị kiểm soát cửa ra vào (không có "người lạ"
theo nghĩa tuyệt đối), nên câu hỏi "khuôn mặt này có nằm trong DB không"
(FaceVerifier.verify_face) gần như luôn đúng cho mọi người trong phòng và ít
giá trị. Rủi ro thật sự giám thị khó bắt được là THI HỘ: sinh viên B (cũng có
mặt hợp lệ trong CSDL) ngồi vào bàn của sinh viên A.

IdentityGuard giải quyết việc đó bằng cách gắn với track_id do module
pose_gaze/tracking (TrackingManager) theo dõi xuyên suốt buổi thi:
  - Track chưa có student_id -> tự nhận diện và gán (thay cho gán tay qua API
    PUT /api/pose-gaze/sessions/{id}/tracks/{track_id}/assignment).
  - Track đã có student_id -> liên tục kiểm tra khuôn mặt hiện tại có đúng là
    student_id đã gán không; sai thì trả cảnh báo "identity_mismatch".

Lưu ý triển khai thực tế: TrackingManager giữ state trong RAM theo session
(self._sessions), nên phải truyền vào ĐÚNG instance đang được
backend/api/pose_gaze_routes.py dùng (không tự tạo TrackingManager mới ở đây,
nếu không 2 bên sẽ không thấy chung state).
"""

from backend.ai_services.face_verify.face_verify import FaceVerifier


class IdentityGuard:
    def __init__(self, face_verifier: FaceVerifier, tracking_manager):
        self._face = face_verifier
        self._tracking = tracking_manager

    def sync(self, session_id: str, frame, timestamp: float) -> list[dict]:
        """Gọi sau mỗi lần TrackingManager.process_detections() cập nhật track
        cho session_id đó. Tự gán danh tính cho track mới, đối chiếu track đã
        gán. Trả về danh sách dict cảnh báo identity_mismatch (rỗng nếu ổn)."""
        packet = self._tracking.get_packet(session_id)
        alerts: list[dict] = []

        for track in packet.tracks:
            if not track.is_present:
                continue
            bbox = track.bbox.to_list()

            if track.student_id is None:
                self._try_auto_assign(session_id, track.track_id, bbox, frame)
            else:
                alert = self._face.verify_assigned_identity(frame, bbox, track.student_id, timestamp)
                if alert is not None:
                    alert["details"]["track_id"] = track.track_id
                    alerts.append(alert)

        return alerts

    def _try_auto_assign(self, session_id: str, track_id: int, bbox, frame) -> None:
        match = self._face.identify(frame, bbox)
        if match is None:
            return  # chưa nhận diện đủ tin cậy -> chờ frame sau, không đoán bừa

        student_id, _score = match
        try:
            self._tracking.assign_student(session_id, track_id=track_id, student_id=student_id)
        except Exception as e:
            # Ví dụ student_id này đã gán cho track khác đang hiển thị -> bỏ qua,
            # không phải lỗi nghiêm trọng, thử lại ở frame sau.
            print(f"[identity_guard][WARNING] Không gán được {student_id} cho track {track_id}: {e}")
