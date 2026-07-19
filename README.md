# Hệ thống giám sát phòng thi bằng AI (Exam Proctoring System)

## Tóm tắt dự án

[cite_start]Đây là một hệ thống giám sát thi trực tuyến toàn diện, hoạt động trong thời gian thực (realtime)[cite: 242]. Hệ thống tích hợp 4 module AI cốt lõi chạy đồng thời nhằm tự động phát hiện các hành vi gian lận trong phòng thi: 
* [cite_start]**Giám sát tư thế và hướng nhìn** (MediaPipe Pose + Face Mesh)[cite: 243].
* [cite_start]**Phát hiện người lạ thay thi** (FaceNet/face_recognition)[cite: 243].
* [cite_start]**Nhận diện điện thoại hoặc tài liệu trên bàn** (YOLOv8)[cite: 243].
* [cite_start]**Phân tích âm thanh/lời nói bất thường** (VAD + Whisper AI)[cite: 243].

[cite_start]Thách thức cốt lõi và trọng tâm kỹ thuật của dự án nằm ở khả năng xử lý đồng thời (Concurrency) và tối ưu hiệu suất, đảm bảo tốc độ xử lý nhanh, không gây giật lag hay tụt FPS[cite: 244]. [cite_start]Hệ thống được thiết kế theo kiến trúc Micro-Monolith, sử dụng các công nghệ hiện đại nhất hiện nay bao gồm FastAPI cho Backend, Streamlit cho Frontend, cùng mạng WebRTC và WebSocket để giao tiếp dữ liệu[cite: 245].

---

## Mục lục

* [Tóm tắt dự án](#tóm-tắt-dự-án)
* [Cấu trúc thư mục & chức năng từng file](#cấu-trúc-thư-mục--chức-năng-từng-file)

---

## Cấu trúc thư mục & chức năng từng file

[cite_start]Dự án được tổ chức theo mô hình Micro-Monolith (tách biệt rõ Frontend và Backend nhưng vẫn nằm trong cùng một kho lưu trữ), giúp dễ dàng quản lý, gỡ lỗi và mở rộng[cite: 246].

```bash
exam_proctoring_system/
├── backend/
│   ├── main.py                  
│   ├── api/                     
│   │   ├── endpoints_webrtc.py  
│   │   └── endpoints_ws.py      
│   ├── ai_services/             
│   │   ├── pose_gaze.py         
│   │   ├── object_detect.py     
│   │   ├── face_verify.py       
│   │   └── audio_whisper.py     
│   └── core/                    
│       └── config.py            
├── frontend/
│   ├── app.py                   
│   └── components/              
│       ├── student_view.py      
│       └── proctor_dashboard.py 
├── weights/                     
│   ├── yolov8_finetuned.pt      
│   └── whisper_tiny.pt          
├── data/
│   └── student_faces/           
├── .env                         
├── requirements.txt
└── README.md
```
### Chi tiết chức năng từng File / Thư mục

| File / Thư mục | Chức năng chi tiết |
| :--- | :--- |
| **`backend/`** | [cite_start]Thư mục chứa toàn bộ logic xử lý của FastAPI, nơi quản lý kết nối và điều phối dữ liệu[cite: 271]. |
| `backend/main.py` | [cite_start]Tệp gốc để khởi chạy máy chủ Backend[cite: 272]. |
| **`backend/api/`** | [cite_start]Nơi định nghĩa các luồng mở kết nối mạng, nhận video frame và gửi cảnh báo[cite: 273]. |
| `api/endpoints_webrtc.py` | [cite_start]Quản lý kết nối WebRTC để truyền luồng video từ thí sinh lên server mượt mà[cite: 274]. |
| `api/endpoints_ws.py` | [cite_start]Quản lý kết nối WebSocket để server bắn các cảnh báo gian lận ngược về dạng Text/JSON[cite: 275]. |
| **`backend/ai_services/`** | [cite_start]Chứa các script chạy độc lập từng module AI, giúp dễ dàng bảo trì hoặc thay thế mô hình[cite: 276]. |
| `ai_services/pose_gaze.py` | [cite_start]Module tính góc Euler từ MediaPipe để tìm hướng nhìn của sinh viên[cite: 277]. |
| `ai_services/object_detect.py` | [cite_start]Module chạy mô hình YOLOv8 phát hiện vật thể[cite: 278]. |
| `ai_services/face_verify.py` | [cite_start]Module xử lý nhận diện danh tính và so sánh khuôn mặt[cite: 279]. |
| `ai_services/audio_whisper.py` | [cite_start]Module xử lý Audio Chunking, VAD và mồi cho Whisper AI[cite: 280]. |
| `backend/core/config.py` | [cite_start]Chứa các biến cấu hình và tham số hệ thống[cite: 281]. |
| **`frontend/`** | [cite_start]Chứa mã nguồn Streamlit để hiển thị giao diện luồng giám sát[cite: 282]. |
| `frontend/app.py` | [cite_start]Tệp gốc khởi chạy ứng dụng web giao diện[cite: 283]. |
| **`frontend/components/`** | [cite_start]Chứa các module giao diện như màn hình làm bài của thí sinh và bảng điều khiển (Dashboard) của giám thị[cite: 284]. |
| **`weights/`** | [cite_start]Nơi lưu trữ các file trọng số (model weights) tĩnh của AI, giúp tách biệt code và dữ liệu mô hình[cite: 285]. |
| **`data/student_faces/`** | [cite_start]Thư mục lưu trữ ảnh gốc hoặc Face Vector đã trích xuất sẵn của thí sinh, dùng để so sánh Cosine Similarity tìm người lạ[cite: 286]. |
| `.env` | [cite_start]Tệp tin ẩn dùng để lưu trữ các biến môi trường bảo mật (Mật khẩu, Port, API Key)[cite: 287]. |
| `requirements.txt` | [cite_start]Danh sách liệt kê các thư viện Python (dependencies) cần thiết để chạy dự án[cite: 288]. |
| `README.md` | [cite_start]Tệp tài liệu hiện tại chứa mô tả và hướng dẫn về dự án[cite: 289]. |