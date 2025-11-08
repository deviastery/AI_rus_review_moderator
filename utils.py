import matplotlib.pyplot as plt

def probabilities_to_levels(probs):
    return [probability_to_level(p) for p in probs]

def probability_to_level(prob):
    """Преобразует вероятность [0,1] в уровень 0–5"""
    thresholds = [0.10, 0.25, 0.40, 0.55, 0.75, 1.01]
    for level, thresh in enumerate(thresholds):
        if prob < thresh:
            return level
    return 5