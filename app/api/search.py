"""API роутер для поиска документов."""

from fastapi import APIRouter, HTTPException
from typing import List

from app.services.search_service import handle_query, handle_index_check
from app.schemas.search import UserQuery, SearchResult, CheckIndex, IndexCheckResponse
from app.core.logger import logger

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/query", response_model=SearchResult)
async def search_documents(query: UserQuery):
    """
    Выполняет поиск по документам.
    
    Args:
        query: Параметры поискового запроса
        
    Returns:
        Результаты поиска
    """
    logger.info(f"Поиск в индексе {query.index_name}: {query.query_text}")
    
    result = await handle_query(query)
    return result


@router.post("/check", response_model=IndexCheckResponse)
async def check_file_in_index(request: CheckIndex):
    """
    Проверяет наличие файла в индексе.
    
    Args:
        request: Параметры для проверки файла
        
    Returns:
        Статус наличия файла в индексе
    """
    logger.info(f"Проверка файла {request.source} в индексе {request.index_name}")
    
    result = await handle_index_check(request)
    return result


@router.get("/health")
async def health_check():
    """
    Проверка работоспособности сервиса поиска.
    
    Returns:
        Статус сервиса
    """
    return {"status": "healthy", "service": "search"} 