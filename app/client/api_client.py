"""HTTP клиент для взаимодействия UI с FastAPI бэкендом."""

import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import logging
from io import BytesIO

logger = logging.getLogger(__name__)


class APIClient:
    """HTTP клиент для работы с FastAPI бэкендом."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Инициализация API клиента.
        
        Args:
            base_url: Базовый URL API бэкенда
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = 60  # таймаут запросов в секундах
        
    async def health_check(self) -> Dict[str, Any]:
        """Проверка работоспособности API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья API: {e}")
            return {"status": "error", "message": str(e)}
    
    async def generate_answer(self, user_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует ответ на запрос пользователя.
        
        Args:
            user_query: Параметры запроса пользователя
            
        Returns:
            Результат генерации ответа
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/answers/generate",
                    json=user_query,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка генерации ответа: HTTP {response.status} - {error_text}")
                        return {
                            "error": f"HTTP {response.status}",
                            "message": error_text
                        }
        except Exception as e:
            logger.error(f"Ошибка при запросе генерации ответа: {e}")
            return {"error": "connection_error", "message": str(e)}
    
    async def search_documents(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет поиск документов.
        
        Args:
            query: Параметры поискового запроса
            
        Returns:
            Результаты поиска
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search/query",
                    json=query,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка поиска: HTTP {response.status} - {error_text}")
                        return {
                            "error": f"HTTP {response.status}",
                            "message": error_text
                        }
        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return {"error": "connection_error", "message": str(e)}
    
    async def upload_document(self, file_data: bytes, filename: str, 
                            index_name: str, use_semantic_chunking: bool = False,
                            semantic_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Загружает документ на сервер.
        
        Args:
            file_data: Содержимое файла
            filename: Имя файла
            index_name: Имя индекса
            use_semantic_chunking: Использовать семантическое разбиение
            semantic_threshold: Порог семантического сходства
            
        Returns:
            Результат загрузки
        """
        try:
            # Подготавливаем данные для отправки
            form_data = aiohttp.FormData()
            form_data.add_field('file', BytesIO(file_data), filename=filename, content_type='application/pdf')
            form_data.add_field('index_name', index_name)
            form_data.add_field('use_semantic_chunking', str(use_semantic_chunking).lower())
            form_data.add_field('semantic_threshold', str(semantic_threshold))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/documents/upload",
                    data=form_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка загрузки документа: HTTP {response.status} - {error_text}")
                        return {
                            "error": f"HTTP {response.status}",
                            "message": error_text
                        }
        except Exception as e:
            logger.error(f"Ошибка при загрузке документа: {e}")
            return {"error": "connection_error", "message": str(e)}
    
    async def get_indices(self) -> List[str]:
        """
        Получает список доступных индексов.
        
        Returns:
            Список имен индексов
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/documents/indices",
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        indices = result.get('indices', [])
                        # Фильтруем системные индексы
                        return [idx for idx in indices if not idx.startswith('.')]
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка получения индексов: HTTP {response.status} - {error_text}")
                        return []
        except Exception as e:
            logger.error(f"Ошибка при получении индексов: {e}")
            return []
    
    async def evaluate_rag_batch(self, file_data: bytes, filename: str, 
                               index_name: str, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запускает пакетную оценку RAG системы.
        
        Args:
            file_data: Содержимое CSV файла
            filename: Имя файла
            index_name: Имя индекса для тестирования
            search_params: Параметры поиска
            
        Returns:
            Результаты оценки
        """
        try:
            # Подготавливаем данные для отправки
            form_data = aiohttp.FormData()
            form_data.add_field('file', BytesIO(file_data), filename=filename, content_type='text/csv')
            form_data.add_field('index_name', index_name)
            
            # Добавляем параметры поиска
            for key, value in search_params.items():
                form_data.add_field(key, str(value))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/evaluation/batch",
                    data=form_data,
                    timeout=300  # Увеличенный таймаут для пакетной обработки
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка пакетной оценки: HTTP {response.status} - {error_text}")
                        return {
                            "error": f"HTTP {response.status}",
                            "message": error_text
                        }
        except Exception as e:
            logger.error(f"Ошибка при пакетной оценке: {e}")
            return {"error": "connection_error", "message": str(e)}
    
    async def evaluate_rag_single(self, question: str, expected_answer: str,
                                index_name: str, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценивает качество ответа на один вопрос.
        
        Args:
            question: Вопрос
            expected_answer: Ожидаемый ответ
            index_name: Имя индекса для поиска
            search_params: Параметры поиска
            
        Returns:
            Результат оценки
        """
        try:
            # Подготавливаем данные для отправки
            form_data = aiohttp.FormData()
            form_data.add_field('question', question)
            form_data.add_field('expected_answer', expected_answer)
            form_data.add_field('index_name', index_name)
            
            # Добавляем параметры поиска
            for key, value in search_params.items():
                form_data.add_field(key, str(value))
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/evaluation/single",
                    data=form_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка одиночной оценки: HTTP {response.status} - {error_text}")
                        return {
                            "error": f"HTTP {response.status}",
                            "message": error_text
                        }
        except Exception as e:
            logger.error(f"Ошибка при одиночной оценке: {e}")
            return {"error": "connection_error", "message": str(e)}


# Глобальный экземпляр API клиента
api_client = APIClient() 