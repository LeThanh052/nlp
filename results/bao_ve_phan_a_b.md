# Bao Ve Phan A Va Phan B

## 1. Muc tieu

Trong bai nay, em chuyen tu pipeline co dien `TF-IDF/Word2Vec + classifier` sang pipeline deep learning:

`text -> tokenize -> sequence -> padding -> embedding -> CNN/LSTM -> output`

Phan A dung du lieu toy de hieu ro flow.  
Phan B giu nguyen dau vao, chi thay than mo hinh tu CNN sang LSTM de so sanh.

Code chinh nam o [deep_learning_assignment.py](d:\nlp\src\deep_learning_assignment.py#L1).  
Ket qua duoc luu trong [deep_learning_summary.json](d:\nlp\results\deep_learning_summary.json) va [deep_learning_explanations.md](d:\nlp\results\deep_learning_explanations.md).

---

## 2. Phan A - Demo CNN Don Gian De Minh Hoa Flow

### 2.1. Du lieu toy

Du lieu toy gom 8 cau ngan, chia thanh 2 lop:

- Lop `1`: Kinh te/Kinh doanh
- Lop `0`: Khong phai Kinh te/Kinh doanh

Vi du:

- `co phieu ngan hang tang manh`
- `thi truong chung khoan giam diem`
- `doanh thu quy nay tang cao`
- `doi tuyen bong da thang tran`
- `ca si ra mat album moi`

Muc tieu cua phan nay khong phai toi uu do chinh xac, ma la hieu ro luong xu ly du lieu.

### 2.2. Chia train/test

Doan code:

```python
x_train, x_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)
```

Giai thich:

- `train_test_split(...)` chia du lieu thanh tap train va tap test.
- Tap train duoc dung de hoc tham so cua mo hinh.
- Tap test duoc dung de danh gia kha nang du doan tren du lieu chua tung gap.
- Neu khong tach train/test, mo hinh co the hoc thuoc du lieu va ket qua danh gia se khong trung thuc.
- `stratify=labels` giup giu ty le 2 lop gan nhu khong doi giua train va test.

Ket qua chia trong bai nay:

- Train gom 6 cau
- Test gom 2 cau

Theo ket qua luu trong [deep_learning_summary.json](d:\nlp\results\deep_learning_summary.json), 2 cau test la:

- `du lich he rat nhon nhip`
- `doanh thu quy nay tang cao`

### 2.3. Tao tokenizer

Doan code:

```python
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
```

Giai thich:

- Token la don vi tu/cum tu trong cau sau khi tach.
- `Tokenizer(...)` tao bo anh xa tu token sang so nguyen.
- `fit_on_texts(x_train)` cho tokenizer hoc tu vung tu tap train.
- Chi duoc fit tren train de tranh ro ri thong tin tu test vao qua trinh hoc.
- Tap test phai dung lai chinh tokenizer cua train de dam bao cung he quy chieu token id.
- `oov_token="<OOV>"` duoc dung cho nhung tu xuat hien o test nhung khong co trong train.

Vi du trong ket qua:

- `doanh -> 3`
- `tang -> 8`
- `cao -> 39`
- Cac tu la nhu `du lich he rat nhon nhip` khong co trong train nen thanh `1`, tuc `<OOV>`.

### 2.4. Chuyen text thanh sequence so nguyen

Doan code:

```python
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
```

Giai thich:

- `texts_to_sequences(...)` bien moi cau thanh danh sach cac token id.
- Vi du cau `co phieu ngan hang tang manh` duoc doi thanh:

```text
[4, 5, 6, 7, 8, 9]
```

- Cau test `du lich he rat nhon nhip` tro thanh:

```text
[1, 1, 1, 1, 1, 1]
```

Do phan lon tu cua cau nay khong xuat hien trong tap train.

Sequence khac BoW o cho:

- Sequence giu lai thu tu xuat hien cua token
- BoW chi dem tan suat, khong giu thu tu

### 2.5. Padding

Doan code:

```python
x_train_pad = pad_sequences(
    x_train_seq,
    maxlen=max_len,
    padding="post",
    truncating="post",
)
x_test_pad = pad_sequences(
    x_test_seq,
    maxlen=max_len,
    padding="post",
    truncating="post",
)
```

Giai thich:

- Deep learning can dau vao co cung kich thuoc.
- Cac cau co do dai khac nhau, nen can `pad_sequences(...)` de bo sung so `0` vao cuoi cau ngan.
- `max_len=10` nghia la moi cau sau cung se co do dai 10.
- `padding="post"` them so `0` o cuoi.
- `truncating="post"` cat bot o cuoi neu cau qua dai.

Vi du:

```text
[4, 5, 6, 7, 8, 9]
```

sau padding thanh:

```text
[4, 5, 6, 7, 8, 9, 0, 0, 0, 0]
```

### 2.6. Xay dung mo hinh CNN

Doan code:

```python
def build_cnn_model(vocab_size, max_len, embedding_dim=64):
    return Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Conv1D(filters=128, kernel_size=3, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
```

Giai thich tung thanh phan:

- `Embedding(...)`: bien moi token id thanh mot vector dac trung hoc duoc.
- `Conv1D(...)`: hoc cac pattern cuc bo tren chuoi token, vi du cac cum 3 tu di lien nhau.
- `GlobalMaxPooling1D()`: lay tin hieu manh nhat tren toan bo cau, giup giam kich thuoc va giu feature quan trong nhat.
- `Dense(64, relu)`: hoc tiep cach ket hop cac feature.
- `Dense(1, sigmoid)`: tra ra xac suat thuoc lop 1.

Truc giac:

- CNN phu hop de bat cac mau cum tu ngan.
- Neu trong cau co cac pattern nhu `co phieu tang`, `thi truong chung khoan`, `doanh thu cao`, CNN co the hoc nhung pattern nay.

### 2.7. Compile va train

Doan code:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train_pad,
    np.array(y_train),
    validation_split=0.2,
    epochs=12,
    batch_size=2,
    verbose=0,
)
```

Giai thich:

- `optimizer="adam"` la bo toi uu pho bien, hoi tu nhanh va de dung.
- `loss="binary_crossentropy"` phu hop voi bai toan phan loai nhi phan.
- `metrics=["accuracy"]` de theo doi do chinh xac trong qua trinh train.
- `epochs=12` la so lan hoc qua toan bo tap train.
- `batch_size=2` nghia la moi lan cap nhat trong so, mo hinh nhin 2 mau.

### 2.8. Danh gia va du doan thu

Doan code:

```python
pred_probs = model.predict(x_test_pad, verbose=0).reshape(-1)
pred_binary = (pred_probs >= 0.5).astype(int)
```

Giai thich:

- `predict(...)` tra ve xac suat.
- Neu xac suat `>= 0.5` thi du doan lop `1`.
- Neu `< 0.5` thi du doan lop `0`.

Ket qua toy demo:

- CNN accuracy: `0.5`
- Precision lop 1: `0.5`
- Recall lop 1: `1.0`
- F1 lop 1: `0.6667`

Confusion matrix:

```text
[[0, 1],
 [0, 1]]
```

Nhan xet:

- Mo hinh du doan ca 2 mau test deu la lop `1`.
- Trong 2 mau test, 1 mau dung, 1 mau sai, nen accuracy bang `0.5`.
- Ket qua nay khong bat thuong vi du lieu toy qua nho, rat de bi anh huong boi cach chia train/test.

### 2.9. Tra loi 7 cau hoi bat buoc trong bao cao

1. `Tokenizer.fit_on_texts(...)` lam gi?  
   Hoc tu vung tren tap train va gan moi token mot id.

2. `texts_to_sequences(...)` lam gi?  
   Doi moi cau thanh danh sach so nguyen theo vocabulary da hoc.

3. `pad_sequences(...)` lam gi?  
   Dua moi sequence ve cung do dai de dua vao neural network.

4. `Embedding(...)` dong vai tro gi?  
   Hoc vector bieu dien cho tung token thay vi dung vector thu cong.

5. `Conv1D(...)` dang hoc gi theo truc giac?  
   Hoc cac pattern cuc bo tren chuoi token, thuong la cum tu ngan.

6. `GlobalMaxPooling1D()` giup gi?  
   Giu lai tin hieu manh nhat, giam kich thuoc va giam so tham so.

7. Vi sao pipeline nay khac TF-IDF + Logistic/SVM?  
   Vi deep learning hoc truc tiep tren sequence va tu hoc bieu dien; con TF-IDF la bieu dien co dinh truoc khi dua vao classifier.

---

## 3. Phan B - Doi Tu CNN Sang LSTM

### 3.1. Muc tieu

Phan B giu nguyen toan bo pipeline dau vao nhu Phan A, chi thay phan than mo hinh tu CNN sang LSTM.

Y nghia:

- Sinh vien thay duoc input pipeline co the giu nguyen
- Phan thay doi lon nhat nam o kieu mo hinh xu ly sequence

### 3.2. Nhung phan giu nguyen giua CNN va LSTM

Nhung phan sau duoc giu nguyen:

- Cung bo du lieu toy
- Cung cach train/test split
- Cung tokenizer
- Cung `max_words`
- Cung `max_len`
- Cung cach `texts_to_sequences`
- Cung `pad_sequences`
- Cung dau ra `Dense(1, activation="sigmoid")`

Noi cach khac, dau vao va cach chuan bi du lieu khong doi, chi doi kien truc xu ly sequence.

### 3.3. Doan code LSTM

```python
def build_lstm_model(vocab_size, max_len, embedding_dim=64):
    return Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
```

Giai thich:

- `Embedding(...)` van dong vai tro bien token id thanh vector.
- `LSTM(64)` doc lan luot tung token trong chuoi va giu thong tin theo thu tu.
- `Dense(32, relu)` hoc them tren dac trung sequence.
- `Dense(1, sigmoid)` tra ve xac suat lop 1.

### 3.4. Truc giac so sanh CNN va LSTM

CNN:

- Manh o viec bat pattern cuc bo
- Nhin tot cac cum tu ngan xuat hien lien tiep
- Thuong nhanh hon

LSTM:

- Manh o viec doc chuoi theo thu tu
- Co kha nang nho thong tin truoc do trong cau
- Phu hop khi thu tu va ngu canh dai han quan trong

### 3.5. Ket qua toy demo

Ket qua tu [deep_learning_summary.json](d:\nlp\results\deep_learning_summary.json):

| Mo hinh | Accuracy | Precision lop 1 | Recall lop 1 | F1 lop 1 |
|---|---:|---:|---:|---:|
| CNN | 0.5 | 0.5 | 1.0 | 0.6667 |
| LSTM | 0.5 | 0.5 | 1.0 | 0.6667 |

Nhan xet:

- Tren bo du lieu toy rat nho, CNN va LSTM cho ket qua giong nhau.
- Khong nen ket luan mo hinh nao tot hon han.
- Ket qua bi anh huong rat manh boi:
  - kich thuoc du lieu qua nho
  - chi co 2 mau test
  - rat nhieu token test khong co trong train

### 3.6. Cach tra loi cau hoi phan tich cua giao vien

1. Phan nao giu nguyen giua CNN va LSTM?  
   Train/test split, tokenizer, sequence, padding, embedding, output sigmoid.

2. Phan nao thay doi?  
   Than mo hinh: `Conv1D + GlobalMaxPooling1D` duoc thay bang `LSTM`.

3. Theo truc giac, CNN phu hop gi?  
   Hoc cac pattern cuc bo nhu cum tu ngan.

4. Theo truc giac, LSTM phu hop gi?  
   Hoc phu thuoc theo chuoi va thu tu xuat hien.

5. CNN va LSTM co khac nhau khong?  
   Ve kien truc thi khac, nhung tren toy demo ket qua chua the hien ro su khac biet.

6. Co nen ket luan mo hinh nao tot hon han khong?  
   Khong. Vi du lieu toy qua nho, ket qua chua du on dinh de rut ra ket luan tong quat.

---

## 4. Diem Co The Noi Them Neu Bi Hoi Sau

### 4.1. Vi sao toy demo ket qua thap?

- Du lieu qua nho
- Tap test chi co 2 cau
- Nhieu tu trong test khong co trong train nen bi doi thanh `<OOV>`
- Muc tieu toy demo la hieu flow, khong phai toi uu mo hinh

### 4.2. Vi sao phai fit tokenizer tren train truoc?

Neu fit tren ca test, vocabulary se hoc truoc thong tin tu tap test, dan den danh gia khong con cong bang.

### 4.3. Vi sao sequence khac TF-IDF?

- Sequence giu thu tu token
- TF-IDF mat thu tu, chi giu tam quan trong theo tan suat

### 4.4. Vi sao can padding?

Vi neural network can tensor co kich thuoc dong nhat. Cac cau dai ngan khac nhau nen phai dua ve cung do dai.

---

## 5. Ket Luan Ngan Cho Phan A Va B

Phan A giup em hieu ro toan bo flow cua text classification bang deep learning, tu tokenizer den sequence, padding va CNN.  
Phan B cho thay co the giu nguyen pipeline dau vao va chi thay doi kien truc xu ly sequence tu CNN sang LSTM.  
Tren bo du lieu toy, ket qua cua 2 mo hinh chua du de ket luan mo hinh nao tot hon. Dieu quan trong nhat cua 2 phan nay la nam vung quy trinh va y nghia cua tung buoc.

---

## 6. File Tham Khao

- Code: [deep_learning_assignment.py](d:\nlp\src\deep_learning_assignment.py#L1)
- Tong hop ket qua: [deep_learning_summary.json](d:\nlp\results\deep_learning_summary.json)
- Giai thich ngan: [deep_learning_explanations.md](d:\nlp\results\deep_learning_explanations.md)
