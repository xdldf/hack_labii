# NER Hackathon Project

Проект для обучения и развертывания модели Named Entity Recognition (NER) на русском языке с использованием трансформеров.

## Структура проекта

```
├── train.py              # Скрипт для обучения модели
├── predict_server.py     # FastAPI сервер для предсказаний
├── dataset_clean.py      # Скрипт для очистки данных
├── requirements.txt       # Python зависимости
├── Dockerfile           # Docker конфигурация
├── datasets/            # Папка с датасетами
└── models/              # Папка с обученными моделями
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Обучение модели

### Подготовка данных

1. Поместите CSV файл с данными в папку `datasets/`
2. Файл должен содержать колонки:
   - `sample` - текст для анализа
   - `annotation` - аннотации в формате `[(start, end, label), ...]`

### Запуск обучения

```bash
python train.py --train_path ./datasets/train_clean.csv --model_name DeepPavlov/rubert-base-cased --output_dir ./models/deeppavlov
```

#### Параметры обучения:

- `--train_path` - путь к обучающему датасету (по умолчанию: `./datasets/train_clean.csv`)
- `--model_name` - название модели (по умолчанию: `DeepPavlov/rubert-base-cased`)
- `--output_dir` - папка для сохранения модели (по умолчанию: `./models/deeppavlov`)
- `--epochs` - количество эпох (по умолчанию: 3)
- `--batch_size` - размер батча (по умолчанию: 8)
- `--lr` - скорость обучения (по умолчанию: 5e-5)
- `--max_length` - максимальная длина последовательности (по умолчанию: 128)

## Docker

### Сборка образа

```bash
# Для Docker Compose
docker build -t ner_api:latest .

# Для обычного Docker
docker build -t ner-hackathon .
```

### Запуск контейнера

#### Вариант 1: Docker Compose (рекомендуется)

```bash
# Запуск сервиса
docker-compose up -d

# Остановка сервиса
docker-compose down

# Просмотр логов
docker-compose logs -f ner
```

#### Вариант 2: Docker команды

```bash
# Запуск с локальной моделью
docker run -p 8000:8000 -v $(pwd)/models:/app/models ner-hackathon

# Запуск с переменной окружения для пути к модели
docker run -p 8000:8000 -e MODEL_DIR=/app/models/deeppavlov -v $(pwd)/models:/app/models ner-hackathon
```

### Параметры запуска:

**Docker Compose:**
- `ports: "8000:8000"` - проброс порта
- `volumes: ./models/deeppavlov:/app/models/deeppavlov` - монтирование папки с моделями
- `environment: MODEL_DIR=/app/models/deeppavlov` - путь к модели в контейнере
- `container_name: ner_api` - имя контейнера
- `image: ner_api:latest` - используемый образ

**Docker команды:**
- `-p 8000:8000` - проброс порта
- `-v $(pwd)/models:/app/models` - монтирование папки с моделями
- `-e MODEL_DIR=/app/models/deeppavlov` - путь к модели в контейнере

## Использование API

После запуска сервера доступны следующие эндпоинты:

### Проверка здоровья

```bash
curl http://localhost:8000/health
```

Ответ:
```json
{"status": "ok"}
```

### Предсказание NER

```bash
curl -X POST "http://176.119.174.102:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": "алёнка шоколад 200 г"}'
```

Ответ:
```json

[{"start_index":0,"end_index":6,"entity":"B-BRAND"},{"start_index":7,"end_index":14,"entity":"B-TYPE"},{"start_index":15,"end_index":18,"entity":"B-VOLUME"},{"start_index":19,"end_index":20,"entity":"I-VOLUME"}]

```

#### Параметры запроса:

- `input` (обязательный) - текст для анализа
- `model_dir` (опциональный) - путь к модели (если не указан, используется значение по умолчанию)
- `max_length` (опциональный) - максимальная длина последовательности (по умолчанию: 128)

## Локальный запуск сервера

```bash
python predict_server.py
```


## Требования к системе

- Python 3.11+
- CUDA (опционально, для ускорения на GPU)
- Docker (для контейнеризации)


