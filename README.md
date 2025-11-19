# AI_rus_review_moderator
**ИИ-модератор для автоматической проверки отзывов на русском языке.**
## About
**AI_rus_review_moderator** — это ИИ-модератор для русскоязычных отзывов, который оценивает степень токсичности по 4 категориям:

- insult (оскорбления и унижения)
- threat (угрозы насилием или вредом)
- obscenity (нецензурная лексика)
- meaningless (бессодержательные/односложные отзывы)

Для каждой категории модель возвращает уровень от 0 до 5, где:
<br/>
- 0-1 (нейтрально / допустимо)
- 2-3 (потенциально проблемно)
- 4-5 (явное нарушение)

Модель обучена на расширенной версии датасета [toxic-russian-comments](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments), 
дополненного пользовательскими отзывами. 

## Installing
> Перед установкой AI_rus_review_moderator, убедитесь, что у вас соответствующие системные требования:
>
> Python = 3.10

<br/>
Для установки проекта нужно скачать репозиторий, затем создать виртуальное окружение:
<br/>

```bash
# 1. Создать окружение
python -m venv venv

# 2. Активировать
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt
```

## Features
Примеры использования также есть в moderator.test
<br/>
**Модерация отзыва**
<br/>
Пример использования:
<br/>

```bash
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
```

**Добавление отзыва в датасет**
<br/>
> При первом вызове add_review() создаётся файл added_dataset_reviews.csv в той же папке, где запущен скрипт.
> Он содержит заголовок и инструкции в виде комментариев (#), которые автоматически игнорируются при обучении.
<br/>
Пример использования:
<br/>

```bash
# Добавление отзыва в датасет
# Нужно указать один или несколько классов (normal, insult, threat, obscenity, meaningless)
# Также можно писать отзывы вручную в файл added_dataset_reviews.csv
review = "Побью"
classes = ["threat", "meaningless"]
success = add_review(review, classes)
if not success:
    print("Произошла ошибка при добавлении отзыва")
```

**Обучение модели после добавление нового отзыва в датасет**
<br/>
Пример использования:
<br/>

```bash
# Обучение модели
success = retrain_model(
    epochs=20,              # количество эпох (по умолчанию 20)
    batch_size=512,         # количество батчей (по умолчанию 512)
    added_weight=3.0        # добавленный вес для added-отзывов (по умолчанию х3)
)
if not success:
    print("Произошла ошибка при обучении модели")
```

## Notes
Посмотреть на графики обучения можно на tensorboard:
<br/>

```bash
python -m tensorboard.main --logdir=logs/fit
```
