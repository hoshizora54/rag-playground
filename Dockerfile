# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем requirements файл
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY . .

# Создаем директории для логов
RUN mkdir -p /app/logs

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Открываем порты для FastAPI и Streamlit
EXPOSE 8000 8503

# Создаем startup скрипт
RUN echo '#!/bin/bash\n\
echo "Starting RAG PDF System..."\n\
echo "Starting FastAPI server..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 &\n\
echo "Starting Streamlit UI..."\n\
streamlit run app_ui_api.py --server.port 8503 --server.address 0.0.0.0 &\n\
echo "All services started!"\n\
wait' > /app/start.sh

RUN chmod +x /app/start.sh

# Запускаем оба сервиса
CMD ["/app/start.sh"] 