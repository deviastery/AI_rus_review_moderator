from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, LSTM, BatchNormalization
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
# Размерность векторного представления каждого слова
EMBEDDING_VECTOR_LENGTH = 128
# Размерность отзыва
REVIEW_LENGTH = 500
# Размерность словаря
VOCABULARY_SIZE = 20_000

def build_model():
    model = Sequential([
        Embedding(VOCABULARY_SIZE, EMBEDDING_VECTOR_LENGTH, input_length=REVIEW_LENGTH), 
        LSTM(100, dropout=0.8),
        BatchNormalization(),
        Dropout(0.8),
        Dense(len(CATEGORIES), activation='sigmoid')
    ])
    model.compile(
        optimizer='RMSprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_pretrained_model(model_path="data/model.keras", tokenizer_path="data/tokenizer.json"):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)
    model = load_model(model_path)
    return model, tokenizer