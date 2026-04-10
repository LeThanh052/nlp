import json
import os
import re
from pathlib import Path

# Thu vien xu ly bang du lieu
import pandas as pd
# Thu vien tach train/test
from sklearn.model_selection import train_test_split
# Hai cach bieu dien van ban: BoW va TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Ba mo hinh phan loai can so sanh
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# Cac metric danh gia mo hinh
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# Duong dan goc cua project
ROOT = Path(__file__).resolve().parent.parent
# File du lieu goc
DATA_PATH = ROOT / "vietnamese_news_2000_new.csv"
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
    # Dua van ban ve chu thuong
    text = str(text).lower()
    # Xoa link neu co
    text = re.sub(r"https?://\S+", " ", text)
    # Xoa cum boilerplate o cuoi bai bao
    text = re.sub(r"thông tin doanh nghiệp\s*[-–]\s*sản phẩm", " ", text)
    # Xoa ky tu dac biet
    text = re.sub(r"[^\w\s]", " ", text)
    # Chuan hoa khoang trang
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_and_print(model_name, vectorizer_name, y_true, y_pred):
    # Tinh cac chi so danh gia co ban
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    exp_name = f"{model_name} + {vectorizer_name}"

    # In ket qua ra man hinh
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

    # Luu ket qua tong hop de xuat ra CSV
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
    # Luu classification report tung mo hinh
    all_reports[exp_name] = report_dict
    # Luu confusion matrix tung mo hinh
    all_confusions[exp_name] = cm.tolist()


# ==================== DOC DU LIEU ====================
print("\n===== DOC DU LIEU =====")
df = pd.read_csv(DATA_PATH)
print("So dong ban dau:", len(df))
print("So mau moi lop truoc khi lam sach:")
print(df["label"].value_counts())

# ==================== TIEN XU LY ====================
print("\n===== TIEN XU LY =====")
# Gop title + desc + text thanh mot cot dau vao duy nhat
df["input_text"] = (
    df["title"].fillna("") + " " + df["desc"].fillna("") + " " + df["text"].fillna("")
)
# Lam sach van ban dau vao de dua vao mo hinh
df["clean_text"] = df["input_text"].apply(clean_text)
# Lam sach rieng phan text de loc bai qua ngan
df["body_clean"] = df["text"].fillna("").apply(clean_text)

# Bo bai bi trung URL
df = df.drop_duplicates(subset=["url"]).copy()
# Bo bai co phan noi dung qua ngan
df = df[df["body_clean"].str.len() >= 120].copy()
# Reset lai chi so sau khi loc
df = df.reset_index(drop=True)

# Luu file da xu ly
df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")

print("So dong sau khi lam sach:", len(df))
print("So mau moi lop sau khi lam sach:")
print(df["label"].value_counts())

# X la van ban, y la nhan phan loai
X = df["clean_text"]
y = df["label"]

# Chia du lieu train/test theo ti le 80/20, co giu nguyen ti le lop
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n===== CHIA TAP DU LIEU =====")
print("Train:", len(X_train))
print("Test :", len(X_test))

# ==================== BUOC 1: MULTINOMIAL NAIVE BAYES ====================
print("\n===== BUOC 1: MULTINOMIAL NAIVE BAYES =====")

# 1A. Naive Bayes + BoW
bow_vectorizer_nb = CountVectorizer(ngram_range=(1, 1))
X_train_bow_nb = bow_vectorizer_nb.fit_transform(X_train)
X_test_bow_nb = bow_vectorizer_nb.transform(X_test)

nb_bow_model = MultinomialNB()
nb_bow_model.fit(X_train_bow_nb, y_train)
y_pred_nb_bow = nb_bow_model.predict(X_test_bow_nb)
evaluate_and_print("MultinomialNB", "BoW", y_test, y_pred_nb_bow)

# 1B. Naive Bayes + TF-IDF
tfidf_vectorizer_nb = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_nb = tfidf_vectorizer_nb.fit_transform(X_train)
X_test_tfidf_nb = tfidf_vectorizer_nb.transform(X_test)

nb_tfidf_model = MultinomialNB()
nb_tfidf_model.fit(X_train_tfidf_nb, y_train)
y_pred_nb_tfidf = nb_tfidf_model.predict(X_test_tfidf_nb)
evaluate_and_print("MultinomialNB", "TF-IDF", y_test, y_pred_nb_tfidf)

# ==================== BUOC 2: LOGISTIC REGRESSION ====================
print("\n===== BUOC 2: LOGISTIC REGRESSION =====")

# 2A. Logistic Regression + BoW
bow_vectorizer_lr = CountVectorizer(ngram_range=(1, 1))
X_train_bow_lr = bow_vectorizer_lr.fit_transform(X_train)
X_test_bow_lr = bow_vectorizer_lr.transform(X_test)

lr_bow_model = LogisticRegression(max_iter=2000)
lr_bow_model.fit(X_train_bow_lr, y_train)
y_pred_lr_bow = lr_bow_model.predict(X_test_bow_lr)
evaluate_and_print("LogisticRegression", "BoW", y_test, y_pred_lr_bow)

# 2B. Logistic Regression + TF-IDF
tfidf_vectorizer_lr = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_lr = tfidf_vectorizer_lr.fit_transform(X_train)
X_test_tfidf_lr = tfidf_vectorizer_lr.transform(X_test)

lr_tfidf_model = LogisticRegression(max_iter=2000)
lr_tfidf_model.fit(X_train_tfidf_lr, y_train)
y_pred_lr_tfidf = lr_tfidf_model.predict(X_test_tfidf_lr)
evaluate_and_print("LogisticRegression", "TF-IDF", y_test, y_pred_lr_tfidf)

# ==================== BUOC 3: LINEAR SVM ====================
print("\n===== BUOC 3: LINEAR SVM =====")

# 3A. Linear SVM + BoW
bow_vectorizer_svm = CountVectorizer(ngram_range=(1, 1))
X_train_bow_svm = bow_vectorizer_svm.fit_transform(X_train)
X_test_bow_svm = bow_vectorizer_svm.transform(X_test)

svm_bow_model = LinearSVC()
svm_bow_model.fit(X_train_bow_svm, y_train)
y_pred_svm_bow = svm_bow_model.predict(X_test_bow_svm)
evaluate_and_print("LinearSVC", "BoW", y_test, y_pred_svm_bow)

# 3B. Linear SVM + TF-IDF
tfidf_vectorizer_svm = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_svm = tfidf_vectorizer_svm.fit_transform(X_train)
X_test_tfidf_svm = tfidf_vectorizer_svm.transform(X_test)

svm_tfidf_model = LinearSVC()
svm_tfidf_model.fit(X_train_tfidf_svm, y_train)
y_pred_svm_tfidf = svm_tfidf_model.predict(X_test_tfidf_svm)
evaluate_and_print("LinearSVC", "TF-IDF", y_test, y_pred_svm_tfidf)

# ==================== BUOC 4: CAI TIEN 1 - CLASS WEIGHT ====================
print("\n===== BUOC 4: CAI TIEN 1 - CLASS WEIGHT =====")

# Tang trong so cho lop 1 de giam anh huong mat can bang
tfidf_vectorizer_balanced = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf_balanced = tfidf_vectorizer_balanced.fit_transform(X_train)
X_test_tfidf_balanced = tfidf_vectorizer_balanced.transform(X_test)

lr_balanced_model = LogisticRegression(max_iter=2000, class_weight="balanced")
lr_balanced_model.fit(X_train_tfidf_balanced, y_train)
y_pred_lr_balanced = lr_balanced_model.predict(X_test_tfidf_balanced)
evaluate_and_print("LogisticRegression_balanced", "TF-IDF", y_test, y_pred_lr_balanced)

# ==================== BUOC 5: CAI TIEN 2 - BIGRAM ====================
print("\n===== BUOC 5: CAI TIEN 2 - BIGRAM =====")

# Dung them cum 2 tu de mo hinh hoc duoc ngu canh tot hon
tfidf_vectorizer_bigram = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf_bigram = tfidf_vectorizer_bigram.fit_transform(X_train)
X_test_tfidf_bigram = tfidf_vectorizer_bigram.transform(X_test)

lr_bigram_model = LogisticRegression(max_iter=2000)
lr_bigram_model.fit(X_train_tfidf_bigram, y_train)
y_pred_lr_bigram = lr_bigram_model.predict(X_test_tfidf_bigram)
evaluate_and_print("LogisticRegression", "TF-IDF Bigram", y_test, y_pred_lr_bigram)

# ==================== LUU KET QUA ====================
# Sap xep ket qua theo F1 lop 1 va accuracy
results_df = pd.DataFrame(all_results).sort_values(by=["f1_class_1", "accuracy"], ascending=False)
results_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

# Luu classification report tung mo hinh
with open(RESULTS_DIR / "test_reports.json", "w", encoding="utf-8") as f:
    json.dump(all_reports, f, ensure_ascii=False, indent=2)

# Luu confusion matrix tung mo hinh
with open(RESULTS_DIR / "test_confusions.json", "w", encoding="utf-8") as f:
    json.dump(all_confusions, f, ensure_ascii=False, indent=2)

# Tong hop thong tin quan trong nhat
summary = {
    "raw_rows": int(pd.read_csv(DATA_PATH).shape[0]),
    "clean_rows": int(len(df)),
    "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
    "best_model": results_df.iloc[0].to_dict(),
}

with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# In bang tong ket cuoi cung
print("\n===== TONG KET =====")
print(results_df.to_string(index=False))
print("\nDa luu file ket qua vao thu muc results/")
print("Da luu file du lieu da xu ly vao:", PROCESSED_PATH)
