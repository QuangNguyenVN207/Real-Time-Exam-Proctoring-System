# Paper count experiment (khong dung paper_id)

## Muc tieu

Phien ban nay giu `person_id` on dinh, nhung khong tao va khong tracking
`paper_id`. Quyet dinh cheat sheet chi dua tren tong so to giay dang thay.

## Luong xu ly

1. YOLO lay cac bbox co nhan `cheat_sheet`, `paper_unknown`, `test_paper`;
   `book` duoc chuan hoa thanh `cheat_sheet` nhu phien ban hien tai.
2. Cac bbox trung/nam trong nhau duoc gom thanh mot paper observation. Dieu
   nay tranh mot to giay bi dem 2-3 lan do full-frame, person ROI va model COCO.
3. Trong 3 giay SETUP, he thong lay mode (gia tri xuat hien nhieu nhat) lam
   `baseline_count`.
4. Sau khi ARMED, mot so luong moi phai lap lai 3 lan object inference lien
   tiep moi duoc cong nhan. Mot frame mo/nhoe khong tao alert.
5. Neu count tang, bbox moi duoc so khop voi snapshot baseline trong frame.
   Paper moi duoc gan cho `person_id` co bbox gan nhat.
6. Neu count quay ve baseline trong 3 inference lien tiep, alert duoc xoa.

Khong co `paper_id`: `observation_index` chi la so thu tu hien thi trong frame
hien tai va co the thay doi o frame ke tiep.

## Chay hai video

Tu thu muc `develop-paper-count`:

```powershell
python -m backend.ai_services.object_detect.test_video_paper_count `
  "C:\Users\Admin\Desktop\Presentation\develop\data\smartphone.mp4" `
  "C:\Users\Admin\Desktop\Presentation\develop\data\cheatsheet.mp4" `
  --setup-seconds 3 --frame-stride 3
```

Ket qua nam trong `data/paper_count_results/outputs`.

## Gioi han can biet

- Thay mot to cu bang mot to moi ma tong count khong doi se khong bi phat hien.
- Hai to che phu hoan toan va YOLO chi tra mot bbox thi khong the dem thanh hai.
- Camera phai thay ro mat ban trong giai doan SETUP de hoc baseline dung.
- Xac nhan 3 inference giup giam false alert, doi lai alert tre khoang 0.6-0.9
  giay voi cau hinh test hien tai.
