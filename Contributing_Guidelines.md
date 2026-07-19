# 📚 Quy Trình & Hướng Dẫn Đóng Góp Code (Contributing Guidelines)

Tài liệu này quy định cấu trúc thư mục, chiến lược quản lý nhánh (Git Workflow) và các quy chuẩn bắt buộc khi làm việc nhóm trên repository này.

---

## 1. 🗂 Cấu trúc Repository (Monorepo)

<!-- Dự án được tổ chức theo dạng Monorepo, bao gồm các thư mục chính sau:

*   **`/frontend`**: Chứa source code giao diện UI/UX (Ví dụ: React, Vue, Next.js...).
*   **`/backend`**: Chứa API và logic hệ thống lõi (Ví dụ: Node.js, Go, Spring Boot...).
*   **`/ai-core`**: Chứa các model AI và pipeline xử lý dữ liệu multimedia (Ví dụ: Python, PyTorch, TensorFlow...).
*   **`/docs`**: Chứa tài liệu dự án, bao gồm tài liệu API, kiến trúc hệ thống và hướng dẫn cài đặt. -->

---

## 2. 🌿 Quản lý Nhánh (Git Workflow)

> ⚠️ **Quy tắc TỐI THƯỢNG:** Tuyệt đối không ai trong team được phép `push` code trực tiếp lên các nhánh chính. Mọi thay đổi đều phải thông qua Pull Request (PR).

Hệ thống nhánh được cấu trúc như sau:

*   `main`: Nhánh chứa source code ổn định nhất, **chỉ dành cho môi trường Production**.
*   `develop`: Nhánh hội tụ code. Đây là nơi các tính năng mới được gộp vào để kiểm thử (Testing/Staging) trước khi được phát hành ra `main`.
*   `feature/*`: Nhánh phát triển tính năng mới. Luôn được tách ra từ nhánh `develop`.
*   `bugfix/*`: Nhánh sửa lỗi thông thường trong quá trình phát triển (tách ra từ `develop`).
*   `hotfix/*`: Nhánh sửa lỗi khẩn cấp, nghiêm trọng đang xảy ra trên môi trường thực tế (tách thẳng từ `main`).

---

## 3. 🏷 Quy tắc Đặt tên & Commit Code

Việc tuân thủ quy tắc đặt tên giúp lịch sử Git trở nên minh bạch, nhìn vào là biết ngay ai đang làm gì và sửa đổi phần nào.

### Quy tắc đặt tên nhánh
Cú pháp chuẩn: `[loại-nhánh]/[tên-người-dùng]-[tên-tính-năng]`

*Ví dụ:*
*   `feature/dat-upload-video`
*   `bugfix/hoa-fix-login-error`

### Quy tắc Commit Message (Conventional Commits)
Mỗi lần commit cần bắt buộc bắt đầu bằng một tiền tố (prefix) để bộ lọc log hoạt động hiệu quả.

| Tiền tố | Mục đích sử dụng | Ví dụ |
| :--- | :--- | :--- |
| **`feat:`** | Thêm một tính năng mới. | `feat: thêm API nhận diện khuôn mặt` |
| **`fix:`** | Sửa một lỗi (bug). | `fix: sửa lỗi crash khi chưa nhập email` |
| **`docs:`** | Cập nhật, viết thêm tài liệu. | `docs: cập nhật hướng dẫn chạy file docker` |
| **`refactor:`** | Tối ưu lại code, không làm thay đổi logic/tính năng. | `refactor: tách hàm xử lý ảnh sang file util` |
| **`chore:`** | Cập nhật thư viện, cấu hình (không ảnh hưởng code chính). | `chore: cập nhật phiên bản react lên 18` |

---

## 4. 🔄 Quy trình Làm việc Hàng ngày (Daily Flow)

Để đảm bảo source code luôn sạch và không gây xung đột (conflict) với các thành viên khác, bạn cần tuân thủ chặt chẽ luồng làm việc sau:

**Bước 1: Cập nhật code mới nhất**
Đầu ngày làm việc, hãy kéo code mới nhất từ nhánh hội tụ về máy local.
```bash
git checkout develop
git pull origin develop
```
**Bước 2: Tạo nhánh làm việc mới**
Tạo một nhánh feature hoặc bugfix mới từ nhánh develop để bắt đầu task của bạn.

```bash
git checkout -b feature/<tên-người-dùng>-<tên-tính-năng>
```
**Bước 3: Code, Commit và Push**
Lưu lại các thay đổi và đẩy nhánh của bạn lên Remote Repository.

```bash
git add <các-file-hoặc-thư-mục-đã-sửa>
git commit -m "feat: [Mô tả ngắn gọn về phần code vừa viết]"
git push origin feature/<tên-người-dùng>-<tên-tính-năng>
```
**Bước 4: Tạo Pull Request (PR)**

Truy cập vào giao diện web của repository (GitHub/GitLab/Bitbucket).

Tạo PR yêu cầu gộp nhánh feature/... của bạn vào nhánh develop.

**Bước 5: Code Review (Bắt buộc)**

Yêu cầu ít nhất 1 hoặc 2 thành viên khác trong team review đoạn code của bạn.

Chỉ khi người review đồng ý và nhấn "Approve", tiến trình mới được tiếp tục.

**Bước 6: Merge (Gộp code)**
Người quản lý (hoặc chính bạn, sau khi được Approve) sẽ thực hiện thao tác merge code vào nhánh develop trên giao diện web.

🛠 Xử lý gộp code thủ công (Fallback)
Lưu ý: Chỉ sử dụng cách này trong giai đoạn đầu nếu dự án chưa thiết lập chặn merge trực tiếp trên repository. Phải thao tác cẩn thận để không ghi đè code của người khác.

```bash
git checkout develop
git pull origin develop
git merge feature/<tên-của-bạn>-<tên-task>
git push origin develop
```
💡 Các mẹo hữu ích cho Git
Kiểm tra nhánh hiện tại: Sử dụng lệnh git branch. Nhánh bạn đang đứng sẽ có dấu * và được tô màu xanh lá.

Thói quen tốt: Trước khi gõ bất kỳ lệnh git add hay git commit nào, hãy LUÔN LUÔN chạy lệnh git status để chắc chắn bạn đang ở đúng nhánh và kiểm tra lại danh sách các file sắp được lưu.