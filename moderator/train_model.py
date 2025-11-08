import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import numpy as np
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from .model import build_model, MAX_SEQ_LEN, MAX_NUM_WORDS
from .preprocessor import load_dataset

# Путь к dataset_reviews.csv относительно текущего файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'dataset_reviews.csv')

print("Загрузка датасета...")
texts, labels = load_dataset(dataset_path)
print(f"Загружено {len(texts)} отзывов")

y_list = []

# Преобразование меток в multi-label формат
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
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Дополнение массива до размера MAX_SEQ_LEN
X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2
)

VOCAB_SIZE = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
model = build_model(vocab_size=VOCAB_SIZE)
model.summary()

# Папка для логов
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

print("Начинаем обучение...")
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,               
        restore_best_weights=True 
    ),
    ModelCheckpoint("best.keras", save_best_only=True, monitor='val_loss'),
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
    batch_size=256,         
    epochs=50,    
    validation_data=(X_val, y_val),
    verbose=1, 
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Сохранение модели и токенизатора
os.makedirs("data", exist_ok=True)
model.save("data/model.keras")

tokenizer_json = tokenizer.to_json()

with open("data/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("Модель и токенизатор сохранены в папку 'data/'")