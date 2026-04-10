# Deep Learning Notes

## Phan A - Giai thich bat buoc

1. `Tokenizer.fit_on_texts(...)` hoc tu vung tren tap train va gan moi token mot id so nguyen.
2. `texts_to_sequences(...)` bien moi cau thanh danh sach id theo tu vung da hoc.
3. `pad_sequences(...)` cat/do them 0 de moi cau co cung do dai, nhom du lieu thanh tensor deu.
4. `Embedding(...)` hoc vector dac trung dac cho tung token thay vi dung one-hot hay TF-IDF co dinh.
5. `Conv1D(...)` hoc cac mau cum tu cuc bo, vi du nhung chuoi token ngan xuat hien lien tiep.
6. `GlobalMaxPooling1D()` giu lai dac trung manh nhat tren toan cau, giam kich thuoc dau ra.
7. Pipeline nay khac TF-IDF + Logistic/SVM vi mo hinh hoc bieu dien va mau chuoi truc tiep tu sequence, khong can vector hoa co dinh truoc.

## Phan B - So sanh CNN va LSTM

1. Giua CNN va LSTM, phan giu nguyen la: train/test split, tokenizer, sequence, padding, embedding, dau ra sigmoid.
2. Phan thay doi nam o than mo hinh: CNN dung `Conv1D + GlobalMaxPooling1D`, LSTM dung `LSTM`.
3. Ve truc giac, CNN manh o viec bat pattern cuc bo; LSTM manh o viec ghi nho thu tu va phu thuoc theo chuoi.
4. Ket qua co the khac nhau do cach mo hinh doc sequence khac nhau.
5. Voi bo toy rat nho, khong nen ket luan mo hinh nao tot hon han vi do on dinh thap va de bi anh huong boi cach chia mau.

## Phan mo rong voi du lieu that

1. Du lieu dau vao duoc ghep tu `title + desc + text`, lam sach, tach tu tieng Viet roi moi dua vao tokenizer.
2. CNN/LSTM dung sequence da padding, khong dung BoW/TF-IDF co dinh.
3. Ket qua deep learning can so sanh voi `results/test_metrics.csv` de doi chieu voi TF-IDF va Word2Vec trung binh.
