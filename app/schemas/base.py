"""Базовые схемы данных для RAG сервиса."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class BaseResponse(BaseModel):
    """Базовый класс для всех ответов API."""
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Схема для ошибок."""
    success: bool = False
    error_code: Optional[str] = None


class ResponseModel(BaseResponse):
    """Универсальная модель ответа API."""
    data: Optional[Dict[str, Any]] = None 