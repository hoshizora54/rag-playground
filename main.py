"""Главный файл FastAPI приложения для RAG сервиса."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.api.answers import router as answers_router
from app.api.evaluation import router as evaluation_router
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    logger.info("Запуск RAG PDF сервиса")
    logger.info("Сервис готов к работе")
    yield
    # Shutdown
    logger.info("Завершение работы RAG PDF сервиса")


# Создание FastAPI приложения
app = FastAPI(
    title="RAG PDF Service",
    description="Сервис для загрузки PDF документов и поиска по ним с использованием RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Подключение роутеров
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(answers_router)
app.include_router(evaluation_router)


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "RAG PDF Service",
        "description": "Сервис для загрузки PDF и поиска по ним",
        "endpoints": {
            "documents": "/documents/",
            "search": "/search/",
            "answers": "/answers/",
            "evaluation": "/evaluation/",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    return {
        "status": "healthy",
        "service": "rag-pdf-service",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)