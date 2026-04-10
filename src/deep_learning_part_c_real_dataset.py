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


def load_classic_results():
    metrics_path = RESULTS_DIR / "test_metrics.csv"
    if not metrics_path.exists():
        return {}

    metrics_df = pd.read_csv(metrics_path)
    tfidf_rows = metrics_df[metrics_df["vectorizer"].astype(str).str.contains("TF-IDF", regex=False)]
    word2vec_rows = metrics_df[metrics_df["vectorizer"].astype(str).str.contains("Word2Vec", regex=False)]

    return {
        "best_classic": metrics_df.iloc[0].to_dict(),
        "best_tfidf": tfidf_rows.iloc[0].to_dict() if not tfidf_rows.empty else None,
        "best_word2vec": word2vec_rows.iloc[0].to_dict() if not word2vec_rows.empty else None,
    }


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
    return model, metrics


def main():
    set_seed()

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

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train.tolist())

    x_train_seq = tokenizer.texts_to_sequences(x_train.tolist())
    x_test_seq = tokenizer.texts_to_sequences(x_test.tolist())

    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding="post", truncating="post")
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding="post", truncating="post")

    print("=== PHAN MO RONG - DATASET THAT ===")
    print("Dataset      :", DATA_PATH)
    print("So dong goc  :", pd.read_csv(DATA_PATH).shape[0])
    print("So dong sach :", len(df))
    print("Train size   :", len(x_train))
    print("Test size    :", len(x_test))
    print("Tokenizer    :", "underthesea" if word_tokenize is not None else "split() fallback")

    print("\nVi du sequence train dau tien:")
    print(x_train_seq[:2])
    print("\nVi du padding train dau tien:")
    print(x_train_pad[:2])

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
        "gia vang va chung khoan hom nay tang manh sau phien giao dich tich cuc",
        "tran derby toi nay thu hut dong dao nguoi ham mo bong da",
    ]
    sample_tokenized = [" ".join(tokenize_vietnamese(text)) for text in sample_texts]
    sample_seq = tokenizer.texts_to_sequences(sample_tokenized)
    sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding="post", truncating="post")

    cnn_pred_new = cnn_model.predict(sample_pad, verbose=0).reshape(-1)
    lstm_pred_new = lstm_model.predict(sample_pad, verbose=0).reshape(-1)

    classic_results = load_classic_results()

    summary = {
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
        "classic_comparison": classic_results,
    }

    output_path = RESULTS_DIR / "deep_learning_part_c_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== KET QUA CNN ===")
    print("Accuracy :", cnn_metrics["accuracy"])
    print("Precision:", cnn_metrics["precision_class_1"])
    print("Recall   :", cnn_metrics["recall_class_1"])
    print("F1-score :", cnn_metrics["f1_class_1"])

    print("\n=== KET QUA LSTM ===")
    print("Accuracy :", lstm_metrics["accuracy"])
    print("Precision:", lstm_metrics["precision_class_1"])
    print("Recall   :", lstm_metrics["recall_class_1"])
    print("F1-score :", lstm_metrics["f1_class_1"])

    if classic_results.get("best_tfidf") is not None:
        print("\n=== SO SANH VOI PIPELINE CO DIEN ===")
        print(
            "Best TF-IDF :",
            classic_results["best_tfidf"]["experiment"],
            "F1 =",
            classic_results["best_tfidf"]["f1_class_1"],
        )
    if classic_results.get("best_word2vec") is not None:
        print(
            "Best Word2Vec:",
            classic_results["best_word2vec"]["experiment"],
            "F1 =",
            classic_results["best_word2vec"]["f1_class_1"],
        )

    print("\nDa luu tong hop vao:", output_path)


if __name__ == "__main__":
    main()
