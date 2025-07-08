"""Модуль логирования для RAG сервиса."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает и возвращает логгер.
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        format_string: Формат сообщений
    
    Returns:
        Настроенный логгер
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger


# Глобальный логгер для приложения
logger = setup_logger("rag_service") 