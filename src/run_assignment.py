import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None


RANDOM_STATE = 42

# Duong dan goc cua project
ROOT = Path(__file__).resolve().parent.parent

# File du lieu goc
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

# File du lieu sau khi tien xu ly
PROCESSED_PATH = ROOT / "processed" / "vietnamese_news_2000_processed.csv"

# Thu muc luu ket qua
RESULTS_DIR = ROOT / "results"

# Tao thu muc neu chua ton tai
os.makedirs(ROOT / "processed", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cac bien de gom ket qua cua tat ca thi nghiem
all_results = []
all_reports = {}
all_confusions = {}


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


def document_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def vectorize_documents(token_series, model):
    return np.vstack([document_vector(tokens, model) for tokens in token_series])


def evaluate_and_print(model_name, vectorizer_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    exp_name = f"{model_name} + {vectorizer_name}"

    print("\n" + "=" * 70)
    print(exp_name)
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report_text)

    all_results.append(
        {
            "experiment": exp_name,
            "model": model_name,
            "vectorizer": vectorizer_name,
            "accuracy": round(acc, 4),
            "precision_class_1": round(prec, 4),
            "recall_class_1": round(rec, 4),
            "f1_class_1": round(f1, 4),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )
    all_reports[exp_name] = report_dict
    all_confusions[exp_name] = cm.tolist()


def run_word2vec_models(experiment_name, model, train_tokens, test_tokens, y_train, y_test):
    print(f"\n===== WORD2VEC: {experiment_name.upper()} =====")
    print(
        "Thong so:",
        {
            "vector_size": model.vector_size,
            "window": model.window,
            "min_count": model.min_count,
            "sg": model.sg,
            "epochs": model.epochs,
        },
    )

    X_train_w2v = vectorize_documents(train_tokens, model)
    X_test_w2v = vectorize_documents(test_tokens, model)

    print("Shape train vector:", X_train_w2v.shape)
    print("Shape test vector :", X_test_w2v.shape)

    lr_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_w2v, y_train)
    y_pred_lr = lr_model.predict(X_test_w2v)
    evaluate_and_print("LogisticRegression", experiment_name, y_test, y_pred_lr)

    svm_model = LinearSVC(random_state=RANDOM_STATE)
    svm_model.fit(X_train_w2v, y_train)
    y_pred_svm = svm_model.predict(X_test_w2v)
    evaluate_and_print("LinearSVC", experiment_name, y_test, y_pred_svm)


# ==================== DOC DU LIEU ====================
print("\n===== DOC DU LIEU =====")
print("Nguon du lieu:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("So dong ban dau:", len(df))
print("So mau moi lop truoc khi lam sach:")
print(df["label"].value_counts())

# ==================== TIEN XU LY ====================
print("\n===== TIEN XU LY =====")
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

processed_df = df.drop(columns=["tokens"])
processed_df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")

print("So dong sau khi lam sach:", len(df))
print("So mau moi lop sau khi lam sach:")
print(df["label"].value_counts())
print("Da tach tu bang:", "underthesea" if word_tokenize is not None else "split() fallback")

# X la van ban, y la nhan phan loai
X = df["clean_text"]
y = df["label"]
token_sequences = df["tokens"]

# Chia du lieu train/test theo ti le 80/20, co giu nguyen ti le lop
X_train, X_test, y_train, y_test, tokens_train, tokens_test = train_test_split(
    X,
    y,
    token_sequences,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n===== CHIA TAP DU LIEU =====")
print("Train:", len(X_train))
print("Test :", len(X_test))

# ==================== BUOC 1: MULTINOMIAL NAIVE BAYES ====================
print("\n===== BUOC 1: MULTINOMIAL NAIVE BAYES =====")

bow_vectorizer_nb = CountVectorizer(ngram_range=(1, 1))
X_train_bow_nb = bow_vectorizer_nb.fit_transform(X_train)
X_test_bow_nb = bow_vectorizer_nb.transform(X_test)

nb_bow_model = MultinomialNB()
nb_bow_model.fit(X_train_bow_nb, y_train)
y_pred_nb_bow = nb_bow_model.predict(X_test_bow_nb)
evaluate_and_print("MultinomialNB", "BoW", y_test, y_pred_nb_bow)

tfidf_vectorizer_nb = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_nb = tfidf_vectorizer_nb.fit_transform(X_train)
X_test_tfidf_nb = tfidf_vectorizer_nb.transform(X_test)

nb_tfidf_model = MultinomialNB()
nb_tfidf_model.fit(X_train_tfidf_nb, y_train)
y_pred_nb_tfidf = nb_tfidf_model.predict(X_test_tfidf_nb)
evaluate_and_print("MultinomialNB", "TF-IDF", y_test, y_pred_nb_tfidf)

# ==================== BUOC 2: LOGISTIC REGRESSION ====================
print("\n===== BUOC 2: LOGISTIC REGRESSION =====")

bow_vectorizer_lr = CountVectorizer(ngram_range=(1, 1))
X_train_bow_lr = bow_vectorizer_lr.fit_transform(X_train)
X_test_bow_lr = bow_vectorizer_lr.transform(X_test)

lr_bow_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
lr_bow_model.fit(X_train_bow_lr, y_train)
y_pred_lr_bow = lr_bow_model.predict(X_test_bow_lr)
evaluate_and_print("LogisticRegression", "BoW", y_test, y_pred_lr_bow)

tfidf_vectorizer_lr = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_lr = tfidf_vectorizer_lr.fit_transform(X_train)
X_test_tfidf_lr = tfidf_vectorizer_lr.transform(X_test)

lr_tfidf_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
lr_tfidf_model.fit(X_train_tfidf_lr, y_train)
y_pred_lr_tfidf = lr_tfidf_model.predict(X_test_tfidf_lr)
evaluate_and_print("LogisticRegression", "TF-IDF", y_test, y_pred_lr_tfidf)

# ==================== BUOC 3: LINEAR SVM ====================
print("\n===== BUOC 3: LINEAR SVM =====")

bow_vectorizer_svm = CountVectorizer(ngram_range=(1, 1))
X_train_bow_svm = bow_vectorizer_svm.fit_transform(X_train)
X_test_bow_svm = bow_vectorizer_svm.transform(X_test)

svm_bow_model = LinearSVC(random_state=RANDOM_STATE)
svm_bow_model.fit(X_train_bow_svm, y_train)
y_pred_svm_bow = svm_bow_model.predict(X_test_bow_svm)
evaluate_and_print("LinearSVC", "BoW", y_test, y_pred_svm_bow)

tfidf_vectorizer_svm = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_svm = tfidf_vectorizer_svm.fit_transform(X_train)
X_test_tfidf_svm = tfidf_vectorizer_svm.transform(X_test)

svm_tfidf_model = LinearSVC(random_state=RANDOM_STATE)
svm_tfidf_model.fit(X_train_tfidf_svm, y_train)
y_pred_svm_tfidf = svm_tfidf_model.predict(X_test_tfidf_svm)
evaluate_and_print("LinearSVC", "TF-IDF", y_test, y_pred_svm_tfidf)

# ==================== BUOC 4: CAI TIEN 1 - CLASS WEIGHT ====================
print("\n===== BUOC 4: CAI TIEN 1 - CLASS WEIGHT =====")

tfidf_vectorizer_balanced = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_balanced = tfidf_vectorizer_balanced.fit_transform(X_train)
X_test_tfidf_balanced = tfidf_vectorizer_balanced.transform(X_test)

lr_balanced_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
lr_balanced_model.fit(X_train_tfidf_balanced, y_train)
y_pred_lr_balanced = lr_balanced_model.predict(X_test_tfidf_balanced)
evaluate_and_print("LogisticRegression_balanced", "TF-IDF", y_test, y_pred_lr_balanced)

# ==================== BUOC 5: CAI TIEN 2 - BIGRAM ====================
print("\n===== BUOC 5: CAI TIEN 2 - BIGRAM =====")

tfidf_vectorizer_bigram = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf_bigram = tfidf_vectorizer_bigram.fit_transform(X_train)
X_test_tfidf_bigram = tfidf_vectorizer_bigram.transform(X_test)

lr_bigram_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
lr_bigram_model.fit(X_train_tfidf_bigram, y_train)
y_pred_lr_bigram = lr_bigram_model.predict(X_test_tfidf_bigram)
evaluate_and_print("LogisticRegression", "TF-IDF Bigram", y_test, y_pred_lr_bigram)

# ==================== BUOC 6: WORD2VEC ====================
print("\n===== BUOC 6: WORD2VEC =====")

all_sentences = df["tokens"].tolist()

w2v_base = Word2Vec(
    sentences=all_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=0,
    epochs=10,
    seed=RANDOM_STATE,
)
run_word2vec_models(
    "Word2Vec Mean",
    w2v_base,
    tokens_train,
    tokens_test,
    y_train,
    y_test,
)

w2v_tuned = Word2Vec(
    sentences=all_sentences,
    vector_size=200,
    window=8,
    min_count=1,
    workers=4,
    sg=1,
    epochs=20,
    negative=10,
    sample=1e-4,
    seed=RANDOM_STATE,
)
run_word2vec_models(
    "Word2Vec Mean Tuned",
    w2v_tuned,
    tokens_train,
    tokens_test,
    y_train,
    y_test,
)

# ==================== LUU KET QUA ====================
results_df = pd.DataFrame(all_results).sort_values(by=["f1_class_1", "accuracy"], ascending=False)
results_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

with open(RESULTS_DIR / "test_reports.json", "w", encoding="utf-8") as f:
    json.dump(all_reports, f, ensure_ascii=False, indent=2)

with open(RESULTS_DIR / "test_confusions.json", "w", encoding="utf-8") as f:
    json.dump(all_confusions, f, ensure_ascii=False, indent=2)

best_tfidf_or_bow = results_df[
    results_df["vectorizer"].isin(["BoW", "TF-IDF", "TF-IDF Bigram"])
].iloc[0].to_dict()
best_word2vec = results_df[
    results_df["vectorizer"].str.contains("Word2Vec", regex=False)
].iloc[0].to_dict()

summary = {
    "raw_rows": int(pd.read_csv(DATA_PATH).shape[0]),
    "clean_rows": int(len(df)),
    "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
    "best_model": results_df.iloc[0].to_dict(),
    "best_tfidf_or_bow": best_tfidf_or_bow,
    "best_word2vec": best_word2vec,
    "word_tokenizer": "underthesea" if word_tokenize is not None else "split_fallback",
}

with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n===== TONG KET =====")
print(results_df.to_string(index=False))
print("\nBest TF-IDF/BoW:", best_tfidf_or_bow["experiment"])
print("Best Word2Vec :", best_word2vec["experiment"])
print("\nDa luu file ket qua vao thu muc results/")
print("Da luu file du lieu da xu ly vao:", PROCESSED_PATH)
