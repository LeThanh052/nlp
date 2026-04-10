# Deep Learning Part C Summary

## Tong quan

- Dataset: `D:\nlp\vietnamese_news_2000_new.csv`
- So dong goc: `1949`
- So dong sau lam sach: `1934`
- Phan bo nhan:
  - Lop `0`: `1246`
  - Lop `1`: `688`
- Tokenizer: `underthesea`
- `max_words = 20000`
- `max_len = 300`
- Train size: `1547`
- Test size: `387`

## Ket qua deep learning

| Model | Accuracy | Precision lop 1 | Recall lop 1 | F1 lop 1 |
|---|---:|---:|---:|---:|
| CNN | 0.9819 | 0.9645 | 0.9855 | 0.9749 |
| LSTM | 0.9432 | 0.9754 | 0.8623 | 0.9154 |

## Confusion matrix

CNN:

```text
[[244, 5],
 [  2, 136]]
```

LSTM:

```text
[[246, 3],
 [ 19, 119]]
```

## So sanh voi pipeline co dien

| Model | Accuracy | F1 lop 1 |
|---|---:|---:|
| LogisticRegression + TF-IDF Bigram | 0.9974 | 0.9964 |
| LogisticRegression + Word2Vec Mean | 0.9974 | 0.9964 |
| CNN | 0.9819 | 0.9749 |
| LSTM | 0.9432 | 0.9154 |

## Nhan xet nhanh

- Trong 2 mo hinh deep learning, `CNN` tot hon `LSTM` tren dataset nay.
- CNN co `F1 lop 1 = 0.9749`, cao hon ro so voi LSTM `0.9154`.
- Tuy nhien, ca CNN va LSTM deu chua vuot duoc pipeline co dien.
- Hien tai, mo hinh tot nhat van la `LogisticRegression + TF-IDF Bigram` va `LogisticRegression + Word2Vec Mean`.

## Du doan thu

Van ban mau:

- `gia vang va chung khoan hom nay tang manh sau phien giao dich tich cuc`
- `tran derby toi nay thu hut dong dao nguoi ham mo bong da`

Xac suat du doan:

| Text | CNN | LSTM |
|---|---:|---:|
| Kinh te/Kinh doanh | 0.8280 | 0.0603 |
| Khong phai Kinh te/Kinh doanh | 0.3365 | 0.0603 |

## File lien quan

- File day du: [deep_learning_part_c_summary.json](d:\nlp\results\deep_learning_part_c_summary.json)
- File rut gon JSON: [deep_learning_part_c_summary_short.json](d:\nlp\results\deep_learning_part_c_summary_short.json)
- Code: [deep_learning_part_c_real_dataset.py](d:\nlp\src\deep_learning_part_c_real_dataset.py)
