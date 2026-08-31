# Train model earphone rieng

## Model

Dung `YOLO26s` detection, mot class duy nhat la `earphone`.

- Map ca `Earphone`, `earbud`, `headphone` va `headset` ve `earphone`.
- Train o `imgsz=960` va multi-scale vi earphone thuong rat nho.
- Giu model nay rieng, khong ghi de `best (1).pt`.
- Output la `yolo26s_earphone_best.pt`.

Notebook:

`notebooks/train_earphone_yolo26s_kaggle.ipynb`

## Chay tren Kaggle

1. Upload notebook len Kaggle.
2. Add Input: `ahmedezzat02/datazeft`.
3. Neu co dataset earphone tu camera that, attach them dataset do. Notebook se
   tu dong gop neu ten dataset co `ear`, `headset` hoac `earbud`.
4. Bat GPU va Internet.
5. Run all cells.
6. Download `yolo26s_earphone_best.pt` trong Kaggle Output.

## Du lieu custom nen bo sung

- Earphone o tai nhin nghieng, bi toc/non che.
- Mot ben tai nghe, tai nghe co day va khong day.
- Vat the rat nho trong camera 1080p toan phong.
- Negative: bong tai, kinh, toc, ngon tay, but, logo tren ao/non.

Chia train/val/test theo video; khong chia ngau nhien frame lien tiep.
