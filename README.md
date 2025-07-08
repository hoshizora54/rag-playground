# RAG Система поиска и ответов

**Comprehensive Retrieval-Augmented Generation System** - продвинутая система для интеллектуального поиска по документам и генерации ответов на базе OpenSearch, Yandex GPT и современных методов обработки естественного языка.

## Обзор системы

Это полнофункциональная RAG система корпоративного уровня, которая объединяет:
- **Интеллектуальный поиск** с гибридным подходом (семантический + ключевой)
- **Генерацию ответов** на основе Yandex GPT
- **Продвинутые методы улучшения качества** (HyDE, ColBERT reranking)
- **Систему оценки качества** с автоматическим и ручным оцениванием
- **Веб-интерфейс** для пользователей и программный API

## Ключевые возможности

### 🔍 Интеллектуальный поиск
- **Гибридный поиск**: Комбинация семантического и ключевого поиска
- **HyDE (Hypothetical Document Embeddings)**: Улучшение качества поиска через генерацию гипотетических документов
- **ColBERT реранжирование**: Улучшение релевантности результатов
- **Многоязычная поддержка**: Работа с документами на разных языках
- **Фильтрация и сортировка**: Гибкие настройки поиска

### 📄 Обработка документов
- **PDF-парсинг**: Извлечение текста и изображений из PDF
- **Интеллектуальное разбиение**: Семантическое и обычное chunking
- **Векторизация**: Автоматическое создание embeddings
- **Мета-данные**: Сохранение структуры и контекста документов
- **Поддержка изображений**: Извлечение и индексация изображений

### 🧪 Система оценки качества
- **Пакетное тестирование**: Загрузка CSV файлов с тестовыми данными
- **Автоматическая оценка**: LLM-as-judge для оценки качества ответов
- **Ручная разметка**: Интерфейс для экспертной оценки
- **Аналитика**: Детальная статистика и метрики производительности
- **Сравнительный анализ**: Сравнение автоматических и ручных оценок

### 🎯 Пользовательские интерфейсы
- **Streamlit UI**: Интуитивно понятный веб-интерфейс
- **FastAPI**: RESTful API для интеграции
- **Интерактивные компоненты**: Настройка параметров в реальном времени
- **Визуализация результатов**: Графики и таблицы для анализа

## Технологический стек

### Основные компоненты
- **Python 3.8+**: Основной язык разработки
- **FastAPI 0.104.1**: Асинхронный веб-фреймворк для API
- **Streamlit 1.28.1**: Веб-интерфейс для пользователей
- **OpenSearch 2.11.0**: Поисковая система и векторная база данных
- **PostgreSQL 15+**: Реляционная база данных для аналитики

### AI и ML компоненты
- **Yandex GPT**: Генерация ответов и embeddings
- **ColBERT**: Реранжирование результатов поиска
- **HyDE**: Улучшение качества поиска
- **LangChain**: Фреймворк для работы с LLM

### Инфраструктура
- **Docker**: Контейнеризация приложения
- **MinIO**: S3-совместимое хранилище для изображений
- **Nginx**: Обратный прокси-сервер
- **Kibana**: Мониторинг и аналитика OpenSearch

## Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   OpenSearch    │
│                 │◄──►│                 │◄──►│                 │
│   - Поиск       │    │   - Endpoints   │    │   - Индексация  │
│   - Загрузка    │    │   - Валидация   │    │   - Поиск       │
│   - Тестирование│    │   - Логирование │    │   - Аналитика   │
│   - Разметка    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Yandex GPT    │              │
         │              │                 │              │
         │              │   - Embeddings  │              │
         │              │   - Генерация   │              │
         │              │   - Оценка      │              │
         │              └─────────────────┘              │
         │                                                │
         ▼                                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Services      │    │   MinIO         │
│                 │    │                 │    │                 │
│   - Результаты  │    │   - RAG Logic   │    │   - Изображения │
│   - Рейтинги    │    │   - Evaluation  │    │   - Файлы       │
│   - Статистика  │    │   - Indexing    │    │   - Backups     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Установка и развертывание

### Требования
- Python 3.8 или выше
- Docker и Docker Compose (для контейнеризации)
- OpenSearch кластер
- PostgreSQL база данных
- API ключ Yandex GPT

### Быстрый старт с Docker

1. **Клонируйте репозиторий**:
```bash
git clone <repository-url>
cd rag-system
```

2. **Настройте переменные окружения**:
```bash
cp .env.example .env
# Отредактируйте .env файл с вашими настройками
```

3. **Запустите с Docker Compose**:
```bash
docker-compose up -d
```

4. **Доступ к сервисам**:
- Streamlit UI: http://localhost:8501
- FastAPI документация: http://localhost:8000/docs
- OpenSearch: http://localhost:9200
- Kibana: http://localhost:5601

### Установка для разработки

1. **Создайте виртуальное окружение**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

2. **Установите зависимости**:
```bash
pip install -r requirements.txt
```

3. **Настройте конфигурацию**:
```env
# .env файл
YANDEX_GPT_API_KEY=your_api_key_here
YANDEX_GPT_FOLDER_ID=your_folder_id

OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_system
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

4. **Запустите сервисы**:
```bash
# API сервер
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Streamlit UI (в отдельном терминале)
streamlit run app_ui.py --server.port 8501
```

## Подробное руководство по использованию

### 1. Загрузка и индексация документов

#### Через Streamlit UI:
1. Выберите режим "Загрузка документов"
2. Загрузите PDF файл
3. Укажите название индекса
4. Выберите метод разбиения:
   - **Обычное разбиение**: Фиксированный размер чанков
   - **Семантическое разбиение**: На основе смысловых границ
5. Настройте параметры (размер чанка, overlap)
6. Нажмите "Загрузить документ"

#### Через API:
```bash
curl -X POST "http://localhost:8000/indexing/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "index_name=my_index" \
  -F "chunking_method=semantic"
```

### 2. Поиск и генерация ответов

#### Базовый поиск:
```python
import requests

response = requests.post("http://localhost:8000/search/query", json={
    "query": "Что такое машинное обучение?",
    "index_name": "my_index",
    "k": 5,
    "use_hyde": True,
    "use_colbert": True
})

result = response.json()
print(result["answer"])
```

#### Расширенные параметры:
- **k**: Количество результатов поиска (по умолчанию: 5)
- **use_hyde**: Использовать HyDE для улучшения запроса
- **use_colbert**: Применить ColBERT реранжирование
- **search_type**: "hybrid", "semantic", "keyword"
- **threshold**: Минимальный порог релевантности

### 3. Система оценки качества

#### Пакетное тестирование:
Создайте CSV файл с тестовыми данными:
```csv
question,expected_answer
Что такое искусственный интеллект?,ИИ - это область компьютерных наук...
Как работает нейронная сеть?,Нейронная сеть состоит из узлов...
```

Загрузите через интерфейс или API:
```python
with open('test_data.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/evaluation/batch",
        files={'file': f},
        data={'index_name': 'my_index'}
    )
```

#### Ручная разметка:
1. Перейдите в режим "Разметка ответов"
2. Просмотрите неоцененные тесты
3. Сравните ожидаемый и фактический ответы
4. Поставьте оценку: ПРАВИЛЬНО/НЕПРАВИЛЬНО
5. Добавьте комментарий (опционально)
6. Сохраните оценку

### 4. Аналитика и мониторинг

#### Статистика через API:
```python
# Общая статистика
stats = requests.get("http://localhost:8000/evaluation/statistics").json()

# Статистика по рейтингам
ratings = requests.get("http://localhost:8000/evaluation/statistics/ratings").json()

# Детали теста
test_details = requests.get(f"http://localhost:8000/evaluation/test/{test_id}").json()
```

## API Reference

### Поиск
- `POST /search/query` - Основной поиск
- `POST /search/semantic` - Семантический поиск
- `POST /search/keyword` - Поиск по ключевым словам

### Индексация
- `POST /indexing/upload` - Загрузка документа
- `GET /indexing/indices` - Список индексов
- `DELETE /indexing/index/{name}` - Удаление индекса

### Оценка качества
- `POST /evaluation/batch` - Пакетное тестирование
- `GET /evaluation/unrated` - Неоцененные тесты
- `POST /evaluation/rate` - Сохранение оценки
- `GET /evaluation/statistics` - Статистика
- `GET /evaluation/test/{id}` - Детали теста

### Служебные
- `GET /health` - Проверка здоровья системы
- `GET /info` - Информация о системе

## Схемы баз данных

### PostgreSQL - Таблица rag_test_results
```sql
CREATE TABLE rag_test_results (
    id SERIAL PRIMARY KEY,
    test_session_id UUID NOT NULL,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    actual_answer TEXT,
    search_results JSONB,
    response_time FLOAT,
    accuracy_score FLOAT,
    gpt_evaluation TEXT,
    sources TEXT[],
    
    -- Параметры поиска
    index_name VARCHAR(255),
    search_params JSONB,
    
    -- Ручная разметка
    user_rating BOOLEAN,
    user_comment TEXT,
    rated_by VARCHAR(255),
    rated_at TIMESTAMP,
    
    -- Метаданные
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### OpenSearch - Схема документов
```json
{
  "mappings": {
    "properties": {
      "content": {"type": "text"},
      "content_vector": {"type": "knn_vector", "dimension": 1024},
      "title": {"type": "text"},
      "page_number": {"type": "integer"},
      "chunk_index": {"type": "integer"},
      "source_file": {"type": "keyword"},
      "created_at": {"type": "date"},
      "images": {"type": "keyword"}
    }
  }
}
```

## Продвинутые возможности

### HyDE (Hypothetical Document Embeddings)
Система использует HyDE для улучшения качества поиска:
1. Генерирует гипотетический ответ на запрос
2. Создает embedding для этого ответа
3. Ищет документы, похожие на гипотетический ответ
4. Комбинирует результаты с оригинальным поиском

### ColBERT Reranking
Реранжирование результатов для повышения релевантности:
1. Получает топ-N результатов от OpenSearch
2. Применяет ColBERT модель для точного скоринга
3. Переупорядочивает результаты по релевантности
4. Возвращает топ-K наиболее релевантных документов

### Семантическое разбиение
Интеллектуальное разбиение документов:
1. Анализирует семантическую структуру текста
2. Определяет естественные границы между темами
3. Создает чанки переменного размера
4. Сохраняет контекст и связность информации

## Производительность и масштабирование

### Оптимизация поиска
- Кэширование векторов запросов
- Параллельная обработка чанков
- Batch-обработка для больших объемов
- Асинхронные операции

### Мониторинг
- Логирование всех операций
- Метрики производительности
- Отслеживание ошибок
- Аналитика использования

### Масштабирование
- Горизонтальное масштабирование OpenSearch
- Репликация PostgreSQL
- Load balancing для API
- Кэширование на уровне приложения

## Безопасность

### Аутентификация
- API ключи для внешнего доступа
- Токены сессий для веб-интерфейса
- Ограничение доступа по IP

### Данные
- Шифрование данных в покое
- HTTPS для всех соединений
- Валидация входных данных
- Санитизация запросов

## Устранение неисправностей

### Типичные проблемы

#### 1. Ошибки подключения к OpenSearch
```bash
# Проверьте статус OpenSearch
curl -u admin:admin http://localhost:9200/_cluster/health

# Проверьте логи
docker logs opensearch-node1
```

#### 2. Проблемы с Yandex GPT
```bash
# Проверьте API ключ
curl -H "Authorization: Bearer $YANDEX_GPT_API_KEY" \
  https://llm.api.cloud.yandex.net/foundationModels/v1/completion
```

#### 3. Ошибки базы данных
```sql
-- Проверьте подключение
SELECT version();

-- Проверьте таблицы
\dt
```

### Логирование
Логи сохраняются в:
- `/var/log/rag-system/app.log` - основные логи приложения
- `/var/log/rag-system/api.log` - логи API запросов
- `/var/log/rag-system/search.log` - логи поиска

### Мониторинг здоровья
```bash
# Проверка статуса всех сервисов
curl http://localhost:8000/health

# Детальная проверка компонентов
curl http://localhost:8000/health/detailed
```

## Разработка

### Структура проекта
```
├── app/
│   ├── api/                 # FastAPI endpoints
│   │   ├── evaluation.py    # Endpoints для оценки качества
│   │   ├── indexing.py      # Endpoints для индексации
│   │   └── search.py        # Endpoints для поиска
│   ├── core/               # Конфигурация и утилиты
│   │   ├── config.py       # Настройки приложения
│   │   └── database.py     # Подключение к БД
│   ├── logics/             # Бизнес-логика
│   │   ├── colbert.py      # ColBERT реранжирование
│   │   ├── hyde.py         # HyDE логика
│   │   └── semantic_chunking.py  # Семантическое разбиение
│   ├── schemas/            # Pydantic модели
│   │   ├── evaluation.py   # Модели для оценки
│   │   ├── indexing.py     # Модели для индексации
│   │   └── search.py       # Модели для поиска
│   ├── services/           # Сервисы приложения
│   │   ├── evaluation_service.py  # Сервис оценки
│   │   ├── indexing_service.py    # Сервис индексации
│   │   └── search_service.py      # Сервис поиска
│   └── main.py            # Точка входа API
├── docker/                # Docker конфигурация
│   ├── Dockerfile         # Основной Dockerfile
│   └── docker-compose.yml # Композиция сервисов
├── docs/                  # Документация
├── tests/                 # Тесты
├── app_ui.py             # Streamlit интерфейс
└── requirements.txt      # Python зависимости
```

### Запуск тестов
```bash
# Все тесты
pytest

# Конкретная категория
pytest tests/test_search.py
pytest tests/test_evaluation.py

# С покрытием
pytest --cov=app tests/
```

### Линтинг и форматирование
```bash
# Проверка стиля
flake8 app/
black app/

# Проверка типов
mypy app/
```

## Дополнительная информация

### Поддерживаемые форматы
- **Документы**: PDF (с текстом и изображениями)
- **Тестовые данные**: CSV, JSON
- **Экспорт результатов**: CSV, JSON, Excel

### Ограничения
- Максимальный размер PDF: 100MB
- Максимальное количество страниц: 1000
- Максимальная длина запроса: 1000 символов
- Максимальное количество результатов: 100

### Дорожная карта
- [ ] Поддержка DOCX и TXT файлов
- [ ] Многопользовательская система
- [ ] Продвинутая аналитика
- [ ] Интеграция с внешними системами
- [ ] Мобильное приложение

### Поддержка
- Документация: [docs/](docs/)
- Вопросы: Создайте issue в репозитории
- Техподдержка: support@rag-system.com

---

**Автор**: RAG System Team  
**Версия**: 2.0.0  
**Лицензия**: MIT

*Система постоянно развивается и улучшается. Следите за обновлениями в репозитории.*

