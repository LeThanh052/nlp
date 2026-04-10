import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


RANDOM_STATE = 42


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_cnn_model(vocab_size, max_len, embedding_dim=32):
    return Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )


def main():
    set_seed()

    # 1. Du lieu toy
    texts = [
        "co phieu ngan hang tang manh",
        "thi truong chung khoan giam diem",
        "doanh thu quy nay tang cao",
        "loi nhuan doanh nghiep vuot ky vong",
        "doi tuyen bong da thang tran",
        "ca si ra mat album moi",
        "du lich he rat nhon nhip",
        "bo phim moi dat doanh thu phong ve cao",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    # 2. Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("=== PHAN A - CNN TOY DEMO ===")
    print("Train texts:", x_train)
    print("Test texts :", x_test)
    print("y_train    :", y_train)
    print("y_test     :", y_test)

    # 3. Tokenize + sequence + padding
    max_words = 1000
    max_len = 10

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding="post", truncating="post")
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding="post", truncating="post")

    print("\nWord index:", tokenizer.word_index)
    print("\nTrain sequences:", x_train_seq)
    print("Test sequences :", x_test_seq)
    print("\nTrain padding:")
    print(x_train_pad)
    print("Test padding:")
    print(x_test_pad)

    # 4. CNN model
    model = build_cnn_model(max_words, max_len)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 5. Train
    model.fit(
        x_train_pad,
        np.array(y_train),
        epochs=10,
        batch_size=2,
        verbose=1,
    )

    # 6. Evaluate
    loss, acc = model.evaluate(x_test_pad, np.array(y_test), verbose=0)
    print("\nTest accuracy:", round(float(acc), 4))

    # 7. Predict new texts
    new_texts = [
        "gia vang hom nay tang manh",
        "tran chung ket bong da rat hap dan",
    ]
    new_seq = tokenizer.texts_to_sequences(new_texts)
    new_pad = pad_sequences(new_seq, maxlen=max_len, padding="post", truncating="post")
    preds = model.predict(new_pad, verbose=0).reshape(-1)

    print("\nNew sequences:", new_seq)
    print("New padding:")
    print(new_pad)
    print("\nPredictions:")
    for text, prob in zip(new_texts, preds):
        print(f"{text} -> {float(prob):.4f}")


if __name__ == "__main__":
    main()
