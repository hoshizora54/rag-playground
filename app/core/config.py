"""Конфигурация приложения."""

import os


class Config:
    """Конфигурация приложения."""
    
    # Minio настройки
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', '')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', '')
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"
    
    # OpenSearch настройки
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
    OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")
    
    # Yandex GPT настройки
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
    YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
    YANDEX_EMBEDDING_MODEL = os.getenv("YANDEX_EMBEDDING_MODEL", "text-search-doc")
    YANDEX_LLM_MODEL = os.getenv("YANDEX_LLM_MODEL", "yandexgpt-lite")
    
    # API URLs
    YANDEX_EMBEDDING_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
    YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    YANDEX_LLM_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"  
    
    # Общие настройки
    BASE_FOLDER = "documents"
    VECTOR_DIMENSION = 256  # Размерность векторов Yandex GPT
    
    # Настройки чанков
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    
    POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

config = Config()
