import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from .model import build_model, REVIEW_LENGTH, VOCABULARY_SIZE
from .preprocessor import load_dataset

# Пути относительно текущего файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATASET = os.path.join(BASE_DIR, 'dataset_reviews.csv')
ADDED_DATASET = os.path.join(BASE_DIR, 'added_dataset_reviews.csv')
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")

def train_model(
    epochs = 20,
    batch_size = 512,
    added_weight = 3.0
): 
    print("Загрузка датасета...")
    texts_main, labels_main = load_dataset(MAIN_DATASET)
    texts_added, labels_added = load_dataset(ADDED_DATASET) if os.path.isfile(ADDED_DATASET) else ([], [])

    if not texts_added:
        print("В added_dataset_reviews.csv нет отзывов — обучение только на основном датасете.")
        texts = texts_main
        labels = labels_main

        print(f"Всего отзывов: {len(texts)}")
    else:
        # Дублирование отзывов
        dup_factor = int(added_weight)
        if dup_factor > 1:
            texts_added = texts_added * dup_factor
            labels_added = labels_added * dup_factor

        texts = texts_main + texts_added
        labels = labels_main + labels_added
        print(f"Всего отзывов: {len(texts)} (добавленных: {int(len(texts_added)/dup_factor)})")

    # Преобразование меток в multi-label формат
    y_list = []
    for label_set in labels:
        y_row = [
            1 if "normal" in label_set else 0,
            1 if "insult" in label_set else 0,
            1 if "threat" in label_set else 0,
            1 if "obscenity" in label_set else 0,
            1 if "meaningless" in label_set else 0
        ]
        y_list.append(y_row)

    y = np.array(y_list, dtype=np.float32)  # shape: (N, 5)
    # Токенизация текста
    tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, oov_token="<OOV>")
    # Сначала обучаем на added_dataset
    if texts_added:
        tokenizer.fit_on_texts(texts_added)
    # Затем — на всём датасете
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Дополнение отзыва до размера REVIEW_LENGTH
    X = pad_sequences(sequences, maxlen=REVIEW_LENGTH, padding='post', truncating='post')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=np.argmax(y, axis=1)
    )

    model = build_model()
    model.summary()
        
    # Папка для логов
    log_dir = "moderator/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    # Папка для модели и токенизатора
    os.makedirs("moderator/data", exist_ok=True)

    print("Начинаем обучение...")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=0, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.5, patience=2, min_lr=1e-6, verbose=0, mode='max'),
        TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')
    ]

    # Балансировка классов
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = dict(enumerate(class_weights))


    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,         
        epochs=epochs,    
        validation_data=(X_val, y_val),
        verbose=1,  
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Сохранение токенизатора
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    print("Модель и токенизатор сохранены в папку 'data/'")
    return history