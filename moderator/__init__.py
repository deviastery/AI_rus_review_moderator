import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from .model import load_pretrained_model, CATEGORIES
from .preprocessor import tokenize_and_vectorize
from .utils import probabilities_to_levels
from .train_model import train_model

# Пути относительно текущего файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ADDED_DATASET = os.path.join(BASE_DIR, 'added_dataset_reviews.csv')

# Глобальный модератор
_default_moderator = None

def _get_moderator():
    """Возвращает глобальный экземпляр ReviewModerator (создаёт при первом вызове)"""
    global _default_moderator
    if _default_moderator is None:
        model_path = os.path.join(DATA_DIR, 'model.keras')
        tokenizer_path = os.path.join(DATA_DIR, 'tokenizer.json')
        _default_moderator = ReviewModerator(model_path, tokenizer_path)
    return _default_moderator

class ReviewModerator:
    def __init__(self, model_path=None, tokenizer_path=None):
        if model_path is None:
            model_path = os.path.join(DATA_DIR, 'model.keras')
        if tokenizer_path is None:
            tokenizer_path = os.path.join(DATA_DIR, 'tokenizer.json')
        
        self.model, self.tokenizer = load_pretrained_model(model_path, tokenizer_path)
        self.categories = CATEGORIES

    def predict_proba(self, text):
        """Возвращает вероятности по категориям"""
        X = tokenize_and_vectorize(text, self.tokenizer)
        probs = self.model.predict(X, verbose=0)[0]
        return {cat: float(p) for cat, p in zip(self.categories, probs) if cat != 'normal'}

    def predict_levels(self, text):
        """Возвращает уровни 0–5 по категориям"""
        probs = self.predict_proba(text)
        levels = {cat: probabilities_to_levels([p])[0] for cat, p in probs.items() if cat != 'normal'}
        return levels

def moderate(text: str, return_probs=False):
    """Анализирует текст и возвращает уровни или вероятности"""
    try:
        moderator = _get_moderator()
        return moderator.predict_proba(text) if return_probs else moderator.predict_levels(text)
    
    except Exception as e:
        print("Произошла ошибка при модерации отзыва")
        print(f"Ошибка: {type(e).__name__}: {e}")
        return False
    
def add_review(text: str, labels: list):
    """
    Добавляет отзыв в added_dataset_reviews.csv в формате: 'метки текст'
    Возвращает True при успехе, False — при ошибке.
    """
    try:
        # Валидация входных данных
        if not isinstance(text, str):
            raise ValueError("Текст должен быть строкой")
        if not isinstance(labels, (list, tuple)):
            raise ValueError("Метки должны быть списком или кортежем")
        labels = [lbl for lbl in labels if lbl in CATEGORIES]
        if not labels:
            raise ValueError(f"Нет валидных меток. Допустимые: {CATEGORIES}")
        
        # Формируем строку
        labels_str = ",".join(labels)
        line = f"{labels_str} {text}\n"

        # Записываем в файл
        file_exists = os.path.isfile(ADDED_DATASET)
        with open(ADDED_DATASET, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("normal,insult,threat,obscenity,meaningless\n\n")
                f.write("# Это файл, в который вы можете написать свои отзывы с одним или несколькими классами\n")
                f.write("# Писать свои отзывы можно после строки с классами в следующем формате:\n")
                f.write("# threat,meaningless Я бы избил всех, кто тут работает\n")
                f.write("# meaningless Окей, норм\n\n")
            f.write(line)

        print(f"Отзыв добавлен в added_dataset_reviews.csv")
        return True

    except Exception as e:
        print("Произошла ошибка при добавлении отзыва")
        print(f"Ошибка: {type(e).__name__}: {e}")
        return False

def retrain_model(
    epochs = 5,
    batch_size = 256,
    added_weight = 3.0
):
    """
    Полное переобучение модели с приоритетом на added_dataset_reviews.csv.

    Args:
        epochs: количество эпох
        batch_size: размер батча
        added_weight: во сколько раз чаще брать отзывы из added_dataset (по умолчанию 3)
    """
    
    try:
        result = train_model(epochs, batch_size, added_weight)
        _default_moderator = None
        return result is not None

    except Exception as e:
        print("Произошла ошибка при обучении модели")
        print(f"Ошибка: {type(e).__name__}: {e}")
        return False