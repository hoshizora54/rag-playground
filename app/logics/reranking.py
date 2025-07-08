"""
Быстрое ColBERT-подобное реранжирование на основе document-level embeddings.

Использует Yandex API для получения embeddings целых документов и запроса,
затем вычисляет косинусное сходство как ColBERT score.
"""

import requests
import numpy as np
from typing import List, Dict, Any, Optional
from app.core.config import config
from app.core.logger import logger


class FastColBERTReranker:
    """Быстрый реранкер на основе document-level embeddings."""
    
    def __init__(self):
        """Инициализация быстрого ColBERT реранкера."""
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
    
    async def rerank_results(self, 
                            query: str, 
                            results: List[Dict[str, Any]], 
                            top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Быстро переранжирует результаты поиска используя document-level ColBERT score.
        
        Args:
            query: Поисковый запрос
            results: Результаты поиска
            top_k: Количество топ результатов
            
        Returns:
            Переранжированные результаты
        """
        try:
            if not results:
                return []
            
            logger.info(f"Начинаем быстрое ColBERT реранжирование {len(results)} результатов")
            
            # Получаем embedding для запроса один раз
            query_embedding = await self._get_embedding(query)
            
            if not query_embedding:
                logger.warning("Не удалось получить embedding для запроса, возвращаем исходные результаты")
                return results[:top_k]
            
            reranked = await self._fast_colbert_rerank(query_embedding, results)
            
            # Ограничиваем количество результатов
            final_results = reranked[:top_k]
            
            logger.info(f"Быстрое ColBERT реранжирование завершено. Возвращаем {len(final_results)} результатов")
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка при быстром ColBERT реранжировании: {e}")
            return results[:top_k]
    
    async def _fast_colbert_rerank(self, 
                                   query_embedding: List[float], 
                                   results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Быстрое реранжирование с использованием document-level embeddings.
        
        Args:
            query_embedding: Embedding запроса
            results: Результаты поиска
            
        Returns:
            Переранжированные результаты
        """
        try:
            logger.info("Применяем быстрое ColBERT реранжирование")
            
            scored_results = []
            
            for i, result in enumerate(results):
                text = result.get("text", "")
                
                if not text.strip():
                    # Если текст пустой, ставим минимальный скор
                    colbert_score = 0.0
                else:
                    # Получаем embedding для всего документа без ограничений
                    doc_embedding = await self._get_embedding(text)
                    
                    if doc_embedding:
                        # Вычисляем косинусное сходство как ColBERT score
                        colbert_score = self._cosine_similarity(query_embedding, doc_embedding)
                    else:
                        colbert_score = 0.0
                
                # Комбинируем с исходным скором
                original_score = result.get("_score", 0.0)
                combined_score = 0.8 * colbert_score + 0.2 * self._normalize_score(original_score)
                
                result_copy = result.copy()
                result_copy["_rerank_score"] = combined_score
                result_copy["_colbert_score"] = colbert_score
                
                scored_results.append(result_copy)
                
                logger.debug(f"Результат {i+1}: ColBERT={colbert_score:.3f}, "
                           f"комбинированный={combined_score:.3f}")
            
            # Сортируем по комбинированному скору
            reranked = sorted(scored_results, key=lambda x: x["_rerank_score"], reverse=True)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Ошибка в быстром ColBERT реранжировании: {e}")
            return results
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Получает embedding для текста через Yandex API.
        
        Args:
            text: Текст для векторизации (без ограничений по длине)
            
        Returns:
            Вектор embedding
        """
        try:
            # Очищаем текст от лишних пробелов, но не ограничиваем длину
            cleaned_text = " ".join(text.split())
            
            if not cleaned_text:
                return None
            
            data = {
                "modelUri": f"emb://{config.YANDEX_FOLDER_ID}/{config.YANDEX_EMBEDDING_MODEL}",
                "text": cleaned_text
            }
            
            response = requests.post(
                config.YANDEX_EMBEDDING_URL,
                headers=self.headers,
                json=data,
                timeout=30  # Увеличиваем timeout для длинных текстов
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                return embedding if embedding else None
            else:
                logger.warning(f"Yandex API вернул статус {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.debug(f"Ошибка при получении embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Вычисляет косинусное сходство между векторами.
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
            
        Returns:
            Косинусное сходство от 0 до 1
        """
        try:
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # Нормализуем в диапазон [0, 1]
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Ошибка при расчете косинусного сходства: {e}")
            return 0.0
    
    def _normalize_score(self, score: float) -> float:
        """
        Нормализует скор в диапазон [0, 1].
        
        Args:
            score: Исходный скор
            
        Returns:
            Нормализованный скор
        """
        try:
            if score <= 0:
                return 0.0
            elif score >= 10:
                return 1.0
            else:
                return score / 10.0
                
        except Exception as e:
            logger.error(f"Ошибка при нормализации скора: {e}")
            return 0.5


# Глобальный экземпляр быстрого ColBERT реранкера
colbert_reranker = FastColBERTReranker() 