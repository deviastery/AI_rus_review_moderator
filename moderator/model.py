from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, GlobalAveragePooling1D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Категории контента модератора
CATEGORIES = [
    "normal",
    "insult",
    "threat",
    "obscenity",
    "meaningless"
]
# Максимальное количество слов в отзыве
MAX_SEQ_LEN = 64

def build_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, 64), 
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(128, activation="selu"),
        Dropout(0.5),
        Dense(64, activation="selu"),
        Dense(len(CATEGORIES), activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

def load_pretrained_model(model_path="data/model.keras", tokenizer_path="data/tokenizer.json"):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)
    model = load_model(model_path)
    return model, tokenizer