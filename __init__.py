import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from .model import load_pretrained_model, CATEGORIES
from .preprocessor import tokenize_and_vectorize
from .utils import probabilities_to_levels

# Путь к data/ относительно текущего файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

class ReviewModerator:
    def __init__(self, model_path=None, tokenizer_path=None):
        if model_path is None:
            model_path = os.path.join(DATA_DIR, 'model.keras')
        if tokenizer_path is None:
            tokenizer_path = os.path.join(DATA_DIR, 'tokenizer.json')
        
        self.model, self.tokenizer = load_pretrained_model(model_path, tokenizer_path)
        self.categories = CATEGORIES

    def predict_proba(self, text: str) -> dict:
        """Возвращает вероятности по категориям"""
        X = tokenize_and_vectorize(text, self.tokenizer)
        probs = self.model.predict(X, verbose=0)[0]  # shape: (5,)
        return {cat: float(p) for cat, p in zip(self.categories, probs)}

    def predict_levels(self, text: str) -> dict:
        """Возвращает уровни 0–5 по категориям"""
        probs = self.predict_proba(text)
        levels = {cat: probabilities_to_levels([p])[0] for cat, p in probs.items() if cat != 'normal'}
        return levels

_default_moderator = None

def moderate(text: str, return_probs=False):
    global _default_moderator
    if _default_moderator is None:
        _default_moderator = ReviewModerator()
    if return_probs:
        return _default_moderator.predict_proba(text)
    else:
        return _default_moderator.predict_levels(text)