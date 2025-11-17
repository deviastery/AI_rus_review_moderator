import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .model import REVIEW_LENGTH

def load_dataset(csv_path):
    texts = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines[1:], start=2):  # пропускаем первую строку с классами
        
        line = line.strip()
        if not line or line.startswith('#'):  # пропускаем пустые строки и комментарии
            continue

        first_space = line.find(' ')
        if first_space == -1:
            print(f"Строка {i}: нет метки или отзыва -> пропущена: '{line}'")
            continue

        labels_part = line[:first_space]
        text_part = line[first_space + 1:]

        current_labels = set(label.strip() for label in labels_part.split(',') if label.strip())

        texts.append(text_part)
        labels.append(current_labels)
    return texts, labels

def preprocess_text(text):
    """Очистка отзыва"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_vectorize(text, tokenizer):
    """Токенизация и векторизация отзыва перед подачей в модель"""
    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=REVIEW_LENGTH, padding='post', truncating='post')
    return padded