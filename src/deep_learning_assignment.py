import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_CANDIDATES = [
    ROOT / "vietnamese_news_2000_new.csv",
    ROOT / "vietnamese-news.csv",
]
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)

if DATA_PATH is None:
    searched = "\n".join(str(path) for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Khong tim thay file du lieu. Da kiem tra cac duong dan sau:\n"
        f"{searched}"
    )

os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"thông tin doanh nghiệp\s*[-–]\s*sản phẩm", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vietnamese(text):
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if word_tokenize is not None:
        return word_tokenize(cleaned, format="text").split()

    return cleaned.split()


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


def build_lstm_model(vocab_size, max_len, embedding_dim=64):
    return Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )


def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_predictions(y_true, y_pred_binary):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred_binary)), 4),
        "precision_class_1": round(float(precision_score(y_true, y_pred_binary, zero_division=0)), 4),
        "recall_class_1": round(float(recall_score(y_true, y_pred_binary, zero_division=0)), 4),
        "f1_class_1": round(float(f1_score(y_true, y_pred_binary, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred_binary).tolist(),
        "classification_report": classification_report(
            y_true, y_pred_binary, zero_division=0, output_dict=True
        ),
    }


def train_tokenizer_and_pad(x_train, x_test, num_words, max_len):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding="post", truncating="post")
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding="post", truncating="post")

    return tokenizer, x_train_seq, x_test_seq, x_train_pad, x_test_pad


def run_single_model(model_name, builder, vocab_size, max_len, x_train_pad, x_test_pad, y_train, y_test, epochs, batch_size):
    set_seed()
    model = compile_model(builder(vocab_size, max_len))
    history = model.fit(
        x_train_pad,
        np.array(y_train),
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    pred_probs = model.predict(x_test_pad, verbose=0).reshape(-1)
    pred_binary = (pred_probs >= 0.5).astype(int)
    metrics = evaluate_predictions(y_test, pred_binary)
    metrics["history"] = {
        "loss": [round(float(v), 4) for v in history.history["loss"]],
        "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
        "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
        "val_accuracy": [round(float(v), 4) for v in history.history["val_accuracy"]],
    }
    metrics["test_probabilities"] = [round(float(v), 4) for v in pred_probs]
    metrics["test_predictions"] = pred_binary.tolist()
    metrics["model_name"] = model_name
    return model, metrics


def load_real_dataset():
    df = pd.read_csv(DATA_PATH)
    df["input_text"] = (
        df["title"].fillna("") + " " + df["desc"].fillna("") + " " + df["text"].fillna("")
    )
    df["clean_text"] = df["input_text"].apply(clean_text)
    df["body_clean"] = df["text"].fillna("").apply(clean_text)
    df = df.drop_duplicates(subset=["url"]).copy()
    df = df[df["body_clean"].str.len() >= 120].copy()
    df = df.reset_index(drop=True)
    df["tokens"] = df["clean_text"].apply(tokenize_vietnamese)
    df["tokenized_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
    return df


def load_classic_results():
    metrics_path = RESULTS_DIR / "test_metrics.csv"
    if not metrics_path.exists():
        return {}

    metrics_df = pd.read_csv(metrics_path)
    best_classic = metrics_df.iloc[0].to_dict()
    tfidf_rows = metrics_df[metrics_df["vectorizer"].astype(str).str.contains("TF-IDF", regex=False)]
    word2vec_rows = metrics_df[metrics_df["vectorizer"].astype(str).str.contains("Word2Vec", regex=False)]

    return {
        "best_classic": best_classic,
        "best_tfidf": tfidf_rows.iloc[0].to_dict() if not tfidf_rows.empty else None,
        "best_word2vec": word2vec_rows.iloc[0].to_dict() if not word2vec_rows.empty else None,
    }


def run_toy_demo():
    texts = [
        "cổ phiếu ngân hàng tăng mạnh",
        "thị trường chứng khoán giảm điểm",
        "doanh thu quý này tăng cao",
        "lợi nhuận doanh nghiệp vượt kỳ vọng",
        "đội tuyển bóng đá thắng trận",
        "ca sĩ ra mắt album mới",
        "du lịch hè rất nhộn nhịp",
        "bộ phim mới đạt doanh thu phòng vé cao",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    max_words = 1000
    max_len = 10
    tokenizer, x_train_seq, x_test_seq, x_train_pad, x_test_pad = train_tokenizer_and_pad(
        x_train,
        x_test,
        num_words=max_words,
        max_len=max_len,
    )

    cnn_model, cnn_metrics = run_single_model(
        "CNN",
        build_cnn_model,
        max_words,
        max_len,
        x_train_pad,
        x_test_pad,
        y_train,
        y_test,
        epochs=12,
        batch_size=2,
    )
    lstm_model, lstm_metrics = run_single_model(
        "LSTM",
        build_lstm_model,
        max_words,
        max_len,
        x_train_pad,
        x_test_pad,
        y_train,
        y_test,
        epochs=12,
        batch_size=2,
    )

    new_texts = [
        "giá vàng hôm nay tăng mạnh",
        "trận chung kết bóng đá rất hấp dẫn",
    ]
    new_seq = tokenizer.texts_to_sequences(new_texts)
    new_pad = pad_sequences(new_seq, maxlen=max_len, padding="post", truncating="post")

    cnn_pred_new = cnn_model.predict(new_pad, verbose=0).reshape(-1)

    lstm_pred_new = lstm_model.predict(new_pad, verbose=0).reshape(-1)

    return {
        "max_words": max_words,
        "max_len": max_len,
        "train_texts": x_train,
        "test_texts": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "word_index": tokenizer.word_index,
        "x_train_seq": x_train_seq,
        "x_test_seq": x_test_seq,
        "x_train_pad": x_train_pad.tolist(),
        "x_test_pad": x_test_pad.tolist(),
        "cnn": cnn_metrics,
        "lstm": lstm_metrics,
        "new_text_predictions": {
            "texts": new_texts,
            "cnn_probabilities": [round(float(v), 4) for v in cnn_pred_new],
            "lstm_probabilities": [round(float(v), 4) for v in lstm_pred_new],
        },
    }


def run_real_dataset_demo():
    df = load_real_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df["tokenized_text"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    max_words = 20000
    max_len = 300
    tokenizer, x_train_seq, x_test_seq, x_train_pad, x_test_pad = train_tokenizer_and_pad(
        x_train.tolist(),
        x_test.tolist(),
        num_words=max_words,
        max_len=max_len,
    )

    cnn_model, cnn_metrics = run_single_model(
        "CNN",
        build_cnn_model,
        max_words,
        max_len,
        x_train_pad,
        x_test_pad,
        y_train.tolist(),
        y_test.tolist(),
        epochs=6,
        batch_size=32,
    )
    lstm_model, lstm_metrics = run_single_model(
        "LSTM",
        build_lstm_model,
        max_words,
        max_len,
        x_train_pad,
        x_test_pad,
        y_train.tolist(),
        y_test.tolist(),
        epochs=6,
        batch_size=32,
    )

    sample_texts = [
        "giá vàng và chứng khoán hôm nay tăng mạnh sau phiên giao dịch tích cực",
        "trận derby tối nay thu hút đông đảo người hâm mộ bóng đá",
    ]
    sample_tokenized = [" ".join(tokenize_vietnamese(text)) for text in sample_texts]
    sample_seq = tokenizer.texts_to_sequences(sample_tokenized)
    sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding="post", truncating="post")

    cnn_pred_new = cnn_model.predict(sample_pad, verbose=0).reshape(-1)

    lstm_pred_new = lstm_model.predict(sample_pad, verbose=0).reshape(-1)

    return {
        "dataset_path": str(DATA_PATH),
        "raw_rows": int(pd.read_csv(DATA_PATH).shape[0]),
        "clean_rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "tokenizer_used": "underthesea" if word_tokenize is not None else "split_fallback",
        "max_words": max_words,
        "max_len": max_len,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_sequence_example": x_train_seq[:2],
        "test_sequence_example": x_test_seq[:2],
        "train_padding_example": x_train_pad[:2].tolist(),
        "test_padding_example": x_test_pad[:2].tolist(),
        "cnn": cnn_metrics,
        "lstm": lstm_metrics,
        "sample_predictions": {
            "texts": sample_texts,
            "tokenized_texts": sample_tokenized,
            "cnn_probabilities": [round(float(v), 4) for v in cnn_pred_new],
            "lstm_probabilities": [round(float(v), 4) for v in lstm_pred_new],
        },
        "classic_comparison": load_classic_results(),
    }


def write_explanation_file():
    explanation = """# Deep Learning Notes

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
"""
    (RESULTS_DIR / "deep_learning_explanations.md").write_text(explanation, encoding="utf-8")


def main():
    set_seed()
    toy_summary = run_toy_demo()
    real_summary = run_real_dataset_demo()

    combined_summary = {
        "toy_demo": toy_summary,
        "real_dataset_demo": real_summary,
    }

    summary_path = RESULTS_DIR / "deep_learning_summary.json"
    summary_path.write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_explanation_file()

    print("\n===== TOY DEMO =====")
    print("CNN accuracy :", toy_summary["cnn"]["accuracy"])
    print("LSTM accuracy:", toy_summary["lstm"]["accuracy"])

    print("\n===== REAL DATASET =====")
    print("Dataset:", real_summary["dataset_path"])
    print("CNN accuracy :", real_summary["cnn"]["accuracy"])
    print("CNN F1 class 1 :", real_summary["cnn"]["f1_class_1"])
    print("LSTM accuracy:", real_summary["lstm"]["accuracy"])
    print("LSTM F1 class 1 :", real_summary["lstm"]["f1_class_1"])

    classic = real_summary["classic_comparison"]
    if classic.get("best_tfidf") is not None:
        print("\n===== SO SANH VOI PIPELINE CO DIEN =====")
        print(
            "Best TF-IDF:",
            classic["best_tfidf"]["experiment"],
            "F1 =",
            classic["best_tfidf"]["f1_class_1"],
        )
    if classic.get("best_word2vec") is not None:
        print(
            "Best Word2Vec:",
            classic["best_word2vec"]["experiment"],
            "F1 =",
            classic["best_word2vec"]["f1_class_1"],
        )

    print("\nDa luu ket qua vao:", RESULTS_DIR / "deep_learning_summary.json")
    print("Da luu ghi chu giai thich vao:", RESULTS_DIR / "deep_learning_explanations.md")


if __name__ == "__main__":
    main()
