from . import moderate, add_review, retrain_model

# Модерирование отзыва
review = "Окей"
result = moderate(review) # Возвращает уровни от 0 до 5
print(result) # Output: {'insult': 0, 'threat': 0, 'obscenity': 0, 'meaningless': 5}
if not result:
    print("Произошла ошибка при модерации отзыва")

review = "Окей"
result = moderate(review, True) # Возвращает вероятности
print(result) # Output: {'insult': 0.027452949434518814, 'threat': 0.03570876270532608, 'obscenity': 0.14793378114700317, 'meaningless': 0.7804442644119263}
if not result:
    print("Произошла ошибка при модерации отзыва")

# Добавление отзыва в датасет
# Нужно указать один или несколько классов (normal, insult, threat, obscenity, meaningless)
# Также можно писать отзывы вручную в файл added_dataset_reviews.csv
review = "Побью"
classes = ["threat", "meaningless"]
success = add_review(review, classes)
if not success:
    print("Произошла ошибка при добавлении отзыва")

# После добавления отзывов необходимо запустить переобучение модели (это займет некоторое время)
# Обучение модели
success = retrain_model(
    epochs=20,              # количество эпох (по умолчанию 20)
    batch_size=512,         # количество батчей (по умолчанию 512)
    added_weight=3.0       # добавленный вес для added-отзывов (по умолчанию х3)
)
if not success:
    print("Произошла ошибка при обучении модели")