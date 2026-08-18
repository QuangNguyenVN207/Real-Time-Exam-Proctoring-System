# Train lai model paper cho logic dem

## Model khuyen nghi

Dung mot model rieng `YOLO26s-seg` chi co mot class `paper`.

- Giu `best (1).pt` + YOLOv8 pretrained cho smartphone, earphone va smartwatch.
- `YOLO26s-seg` chi chay trong person/desk/lap ROI de tach tung to giay.
- Khong resume `best (1).pt`: YOLOv8 detect va YOLO26 segment khac kien truc,
  khac head va khac task.

## Buoc 1: trich frame can annotate

```powershell
python -m backend.ai_services.object_detect.prepare_paper_segmentation_frames `
  "C:\Users\Admin\Desktop\Presentation\develop\data\smartphone.mp4" `
  "C:\Users\Admin\Desktop\Presentation\develop\data\cheatsheet.mp4" `
  --sample-fps 1 `
  --hard-fps 3 `
  --hard-interval cheatsheet:24:31 `
  --hard-interval cheatsheet:40:68
```

## Buoc 2: annotate instance polygon

Upload thu muc `data/paper_segmentation_annotation/images` len Roboflow, CVAT
hoac Ultralytics Platform.

Quy tac quan trong:

1. Chi mot class: `paper`.
2. Moi to vat ly la mot polygon instance rieng, ke ca de thi hop le.
3. To tren dui va to tren ban phai co hai mask rieng, du chung/che nhau.
4. Danh dau ca giay gap, giay bi tay che va chi lo mot phan.
5. Anh khong co paper phai giu lai lam negative image voi label rong.
6. Chia train/val/test theo video hoac doan thoi gian, khong chia ngau nhien
   cac frame lien tiep de tranh data leakage.

Export dinh dang `YOLOv8 Segmentation`, tao Kaggle Dataset va attach no vao
notebook `notebooks/train_paper_count_yolo26s_seg_kaggle.ipynb`.

## Du lieu toi thieu

- Nen co it nhat 500 paper instances thuc te.
- It nhat 150 instances paper tren dui/duoi ban.
- It nhat 100 anh hard-negative: mat ban trang, tay, quan ao, mep ban.
- Val/test phai co cac clip khong xuat hien trong train.

Neu chi train lai public Illegal-Tools dataset, loi paper tren dui se khong
duoc sua vi dataset do khong chua day du tinh huong nay.
