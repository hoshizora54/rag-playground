"""API роутер для генерации ответов."""

from fastapi import APIRouter, HTTPException
from app.core.logger import logger
from app.services.answer_service import answer_service
from app.schemas.search import UserQuery
from typing import Dict, Any

router = APIRouter(prefix="/answers", tags=["answers"])


@router.post("/generate")
async def generate_answer(user_query: UserQuery) -> Dict[str, Any]:
    """
    Генерирует ответ на запрос пользователя.
    
    Args:
        user_query: Запрос пользователя с параметрами поиска
        
    Returns:
        Словарь с ответом, источниками и изображениями
    """
    try:
        result = await answer_service.generate_answer(user_query)
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Проверка работоспособности сервиса ответов."""
    return {"status": "healthy", "service": "answer_service"} 