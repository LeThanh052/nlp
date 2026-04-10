import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Nếu muốn tách từ tiếng Việt tự động thì dùng underthesea
from underthesea import word_tokenize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# =========================================================
# 1. ĐỌC DỮ LIỆU
# =========================================================
# Giả sử file có các cột:
# - text: nội dung bài báo
# - label: nhãn (1 = Kinh tế/Kinh doanh, 0 = Không phải)
#
# Bạn sửa lại tên file và tên cột cho đúng dataset của bạn.

ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    ROOT / "vietnamese_news_2000_new.csv",
    ROOT / "vietnamese-news.csv",
    Path.cwd() / "vietnamese_news_2000_new.csv",
    Path.cwd() / "vietnamese-news.csv",
]

DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)
if DATA_PATH is None:
    searched = "\n".join(f"- {path}" for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Khong tim thay file du lieu. Da kiem tra cac duong dan sau:\n"
        f"{searched}"
    )

df = pd.read_csv(DATA_PATH)
print("Dang doc du lieu tu:", DATA_PATH)

# Đổi tên cột nếu cần
# Ví dụ nếu dataset có cột "content" và "category"
# df = df.rename(columns={"content": "text", "category": "label"})

print("Kích thước dữ liệu:", df.shape)
print(df.head())

# Bỏ dòng bị thiếu
df = df.dropna(subset=["text", "label"]).copy()

# Ép label về dạng số nếu label đang là text
# Ví dụ:
# Kinh tế/Kinh doanh -> 1
# Còn lại -> 0
if df["label"].dtype == "object":
    df["label"] = df["label"].apply(
        lambda x: 1 if str(x).strip().lower() in ["kinh tế", "kinh doanh", "kinh_te", "business", "economy"] else 0
    )

print("\nPhân bố nhãn:")
print(df["label"].value_counts())


# =========================================================
# 2. HÀM TIỀN XỬ LÝ
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)     # bỏ link
    text = re.sub(r"\d+", " ", text)                # bỏ số
    text = re.sub(r"[^\w\s]", " ", text)            # bỏ dấu câu
    text = re.sub(r"\s+", " ", text).strip()        # bỏ khoảng trắng thừa
    return text


def tokenize_vietnamese(text):
    """
    Tách từ tiếng Việt, có gộp từ ghép bằng dấu _
    """
    text = clean_text(text)
    tokens = word_tokenize(text, format="text").split()
    return tokens


# Tiền xử lý
df["tokens"] = df["text"].apply(tokenize_vietnamese)

# Chuẩn bị text đã tokenize cho TF-IDF
df["text_tokenized"] = df["tokens"].apply(lambda x: " ".join(x))


# =========================================================
# 3. TÁCH TRAIN / TEST
# =========================================================
X_train_text, X_test_text, y_train, y_test, X_train_tokens, X_test_tokens = train_test_split(
    df["text_tokenized"],
    df["label"],
    df["tokens"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nSố mẫu train:", len(X_train_text))
print("Số mẫu test :", len(X_test_text))


# =========================================================
# 4. BASELINE TF-IDF + LOGISTIC REGRESSION
# =========================================================
tfidf_lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000))
])

tfidf_lr.fit(X_train_text, y_train)
pred_tfidf_lr = tfidf_lr.predict(X_test_text)

print("\n================ TF-IDF + Logistic Regression ================")
print("Accuracy:", accuracy_score(y_test, pred_tfidf_lr))
print("F1-score:", f1_score(y_test, pred_tfidf_lr))
print(classification_report(y_test, pred_tfidf_lr))


# =========================================================
# 5. BASELINE TF-IDF + SVM
# =========================================================
tfidf_svm = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("clf", SVC(kernel="linear"))
])

tfidf_svm.fit(X_train_text, y_train)
pred_tfidf_svm = tfidf_svm.predict(X_test_text)

print("\n================ TF-IDF + SVM ================")
print("Accuracy:", accuracy_score(y_test, pred_tfidf_svm))
print("F1-score:", f1_score(y_test, pred_tfidf_svm))
print(classification_report(y_test, pred_tfidf_svm))


# =========================================================
# 6. TRAIN WORD2VEC
# =========================================================
# Train trên toàn bộ corpus token
sentences = df["tokens"].tolist()

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,   # số chiều vector
    window=5,          # ngữ cảnh
    min_count=2,       # bỏ từ quá hiếm
    workers=4,
    sg=1,              # 1 = skip-gram, 0 = CBOW
    epochs=10
)

print("\nĐã train Word2Vec xong.")
print("Vector size:", w2v_model.vector_size)


# =========================================================
# 7. HÀM BIỂU DIỄN MỖI BÀI BÁO = TRUNG BÌNH VECTOR TỪ
# =========================================================
def document_vector(tokens, model):
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    # Nếu không có token nào nằm trong vocab, trả vector 0
    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    # Lấy trung bình các vector từ
    return np.mean(vectors, axis=0)


X_train_w2v = np.array([document_vector(tokens, w2v_model) for tokens in X_train_tokens])
X_test_w2v  = np.array([document_vector(tokens, w2v_model) for tokens in X_test_tokens])

print("\nShape vector train:", X_train_w2v.shape)
print("Shape vector test :", X_test_w2v.shape)


# =========================================================
# 8. WORD2VEC + LOGISTIC REGRESSION
# =========================================================
w2v_lr = LogisticRegression(max_iter=1000)
w2v_lr.fit(X_train_w2v, y_train)
pred_w2v_lr = w2v_lr.predict(X_test_w2v)

print("\n================ Word2Vec + Logistic Regression ================")
print("Accuracy:", accuracy_score(y_test, pred_w2v_lr))
print("F1-score:", f1_score(y_test, pred_w2v_lr))
print(classification_report(y_test, pred_w2v_lr))


# =========================================================
# 9. WORD2VEC + SVM
# =========================================================
w2v_svm = SVC(kernel="linear")
w2v_svm.fit(X_train_w2v, y_train)
pred_w2v_svm = w2v_svm.predict(X_test_w2v)

print("\n================ Word2Vec + SVM ================")
print("Accuracy:", accuracy_score(y_test, pred_w2v_svm))
print("F1-score:", f1_score(y_test, pred_w2v_svm))
print(classification_report(y_test, pred_w2v_svm))


# =========================================================
# 10. SO SÁNH KẾT QUẢ
# =========================================================
results = pd.DataFrame({
    "Model": [
        "TF-IDF + Logistic Regression",
        "TF-IDF + SVM",
        "Word2Vec + Logistic Regression",
        "Word2Vec + SVM"
    ],
    "Accuracy": [
        accuracy_score(y_test, pred_tfidf_lr),
        accuracy_score(y_test, pred_tfidf_svm),
        accuracy_score(y_test, pred_w2v_lr),
        accuracy_score(y_test, pred_w2v_svm)
    ],
    "F1-score": [
        f1_score(y_test, pred_tfidf_lr),
        f1_score(y_test, pred_tfidf_svm),
        f1_score(y_test, pred_w2v_lr),
        f1_score(y_test, pred_w2v_svm)
    ]
})

print("\n================ BẢNG SO SÁNH KẾT QUẢ ================")
print(results.sort_values(by="F1-score", ascending=False))
