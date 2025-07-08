"""Дополнительный сервис для работы с OpenSearch."""

from typing import List, Dict, Any
from app.logics.opensearch import opensearch_worker
from app.core.logger import logger


class OpenSearchService:
    """Дополнительный сервис для работы с OpenSearch."""
    
    def __init__(self):
        self.worker = opensearch_worker
    
    def get_indices_list(self) -> List[str]:
        """
        Получает список всех индексов.
        
        Returns:
            Список имен индексов
        """
        try:
            return self.worker.list_indices()
        except Exception as e:
            logger.error(f"Ошибка при получении списка индексов: {e}")
            return []
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Получает информацию о кластере OpenSearch.
        
        Returns:
            Информация о кластере
        """
        try:
            return self.worker.get_opensearch_info()
        except Exception as e:
            logger.error(f"Ошибка при получении информации о кластере: {e}")
            return {}
    
    def create_new_index(self, index_name: str) -> bool:
        """
        Создает новый индекс.
        
        Args:
            index_name: Имя нового индекса
            
        Returns:
            True если индекс создан успешно
        """
        try:
            return self.worker.create_index(index_name)
        except Exception as e:
            logger.error(f"Ошибка при создании индекса {index_name}: {e}")
            return False


# Глобальный экземпляр сервиса
opensearch_service = OpenSearchService()
