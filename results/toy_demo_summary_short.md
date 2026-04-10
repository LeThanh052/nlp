# Toy Demo Summary

## Tong quan

- So cau: `8`
- So mau train: `6`
- So mau test: `2`
- `max_words = 1000`
- `max_len = 10`

## Train/Test split

Train texts:

- `co phieu ngan hang tang manh`
- `ca si ra mat album moi`
- `thi truong chung khoan giam diem`
- `loi nhuan doanh nghiep vuot ky vong`
- `doi tuyen bong da thang tran`
- `bo phim moi dat doanh thu phong ve cao`

Test texts:

- `du lich he rat nhon nhip`
- `doanh thu quy nay tang cao`

## Vi du sequence va padding

Vi du sequence train:

```text
[4, 5, 6, 7, 8, 9]
```

Vi du sequence test:

```text
[1, 1, 1, 1, 1, 1]
```

Nhan xet:

- So `1` la token `<OOV>`, tuc tu khong co trong vocabulary cua tap train.

Vi du sau padding:

```text
[4, 5, 6, 7, 8, 9, 0, 0, 0, 0]
```

## Ket qua CNN va LSTM

| Model | Accuracy | Precision lop 1 | Recall lop 1 | F1 lop 1 |
|---|---:|---:|---:|---:|
| CNN | 0.5 | 0.5 | 1.0 | 0.6667 |
| LSTM | 0.5 | 0.5 | 1.0 | 0.6667 |

## Confusion matrix

CNN:

```text
[[0, 1],
 [0, 1]]
```

LSTM:

```text
[[0, 1],
 [0, 1]]
```

## Nhan xet nhanh

- Ca CNN va LSTM deu cho accuracy `0.5`.
- Cả hai mo hinh deu du doan 2 mau test thanh lop `1`.
- Ket qua nay khong bat thuong vi du lieu toy qua nho.
- Muc tieu cua toy demo la hieu quy trinh:
  - train/test split
  - tokenizer
  - texts to sequences
  - padding
  - embedding
  - CNN/LSTM

## Du doan thu

Van ban moi:

- `gia vang hom nay tang manh`
- `tran chung ket bong da rat hap dan`

Xac suat:

| Text | CNN | LSTM |
|---|---:|---:|
| gia vang hom nay tang manh | 0.6939 | 0.6773 |
| tran chung ket bong da rat hap dan | 0.6262 | 0.6217 |

## Ket luan

- Toy demo chi dung de minh hoa flow cua deep learning cho text classification.
- Khong nen dung ket qua nay de ket luan mo hinh nao tot hon han.
- Muon danh gia nghiem tuc can dung bo du lieu lon hon, nhu phan mo rong voi dataset that.

## File lien quan

- File day du: [deep_learning_summary.json](d:\nlp\results\deep_learning_summary.json)
- File bao ve: [bao_ve_phan_a_b.md](d:\nlp\results\bao_ve_phan_a_b.md)
- JSON rut gon: [toy_demo_summary_short.json](d:\nlp\results\toy_demo_summary_short.json)
