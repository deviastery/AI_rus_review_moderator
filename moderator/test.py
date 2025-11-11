from . import moderate, add_review, retrain_model

# Модерирование отзыва
review = "Окей"
result = moderate(review) # Возвращает уровни от 0 до 5
print(result) # Output: {'insult': 0, 'threat': 0, 'obscenity': 0, 'meaningless': 5}
review = "Окей"
result = moderate(review, True) # Возвращает вероятности
print(result) # Output: {'insult': 0.027452949434518814, 'threat': 0.03570876270532608, 'obscenity': 0.14793378114700317, 'meaningless': 0.7804442644119263}

# Добавление отзыва в датасет
# Нужно указать один или несколько классов (normal, insult, threat, obscenity, meaningless)
# Также можно писать отзывы вручную в файл added_dataset_reviews.csv
review = "Убью"
classes = ["threat", "meaningless"]
success = add_review(review, classes)
if not success:
    print("Произошла ошибка при добавлении отзыва")

# После добавления отзывов необходимо запустить переобучение модели
# Обучение модели
success = retrain_model(
    epochs=50,              # количество эпох (по умолчанию 50)
    batch_size=256,         # количество батчей (по умолчанию 256)
    added_weight=3.0,       # добавленный вес для added-отзывов (по умолчанию х3)
    max_vocab_size=30000    # размер словаря (по умолчанию 30 000)
)
if not success:
    print("Произошла ошибка при обучении модели")