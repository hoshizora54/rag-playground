"""
HyDE (Hypothetical Document Embeddings) - технология для улучшения поиска.

Алгоритм:
1. Генерирует гипотетический ответ на вопрос пользователя
2. Создает embedding для гипотетического ответа
3. Ищет документы похожие на гипотетический ответ
4. Комбинирует с обычным поиском для улучшения результатов
"""

import requests
from typing import List, Dict, Any, Optional
from app.core.config import config
from app.core.logger import logger


class HyDEProcessor:
    """Класс для реализации HyDE технологии."""
    
    def __init__(self):
        """Инициализация HyDE процессора."""
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
    
    async def generate_hypothetical_documents(self, query: str, num_hypotheses: int = 1) -> List[str]:
        """
        Генерирует гипотетические документы для запроса.
        
        Args:
            query: Запрос пользователя
            num_hypotheses: Количество гипотетических документов
            
        Returns:
            Список гипотетических документов
        """
        try:
            hypotheses = []
            
            for i in range(num_hypotheses):
                # Различные промпты для генерации разнообразных гипотез
                prompts = [
                    f"Напиши подробный академический ответ на вопрос: {query}",
                    f"Объясни простыми словами: {query}",
                    f"Приведи практический пример для: {query}",
                    f"Дай техническое определение: {query}"
                ]
                
                prompt = prompts[i % len(prompts)]
                hypothesis = await self._generate_hypothesis(prompt)
                
                if hypothesis and hypothesis not in hypotheses:
                    hypotheses.append(hypothesis)
                    logger.info(f"Сгенерирована гипотеза {i+1}: {hypothesis[:100]}...")
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Ошибка при генерации гипотетических документов: {e}")
            return []
    
    async def _generate_hypothesis(self, prompt: str) -> Optional[str]:
        """
        Генерирует одну гипотезу через Yandex GPT.
        
        Args:
            prompt: Промпт для генерации
            
        Returns:
            Сгенерированная гипотеза
        """
        try:
            data = {
                "modelUri": f"gpt://{config.YANDEX_FOLDER_ID}/{config.YANDEX_LLM_MODEL}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.7,  # Более высокая температура для разнообразия
                    "maxTokens": 500
                },
                "messages": [
                    {
                        "role": "system",
                        "text": "Ты эксперт, который создает подробные, информативные ответы. Отвечай конкретно и структурированно."
                    },
                    {
                        "role": "user",
                        "text": prompt
                    }
                ]
            }
            
            response = requests.post(
                config.YANDEX_LLM_URL,
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                alternatives = result.get("result", {}).get("alternatives", [])
                
                if alternatives:
                    hypothesis = alternatives[0].get("message", {}).get("text", "")
                    return hypothesis.strip() if hypothesis else None
                else:
                    logger.warning("Yandex GPT вернул пустой ответ для гипотезы")
                    return None
            else:
                logger.error(f"Ошибка Yandex GPT API при генерации гипотезы: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при генерации гипотезы: {e}")
            return None
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создает embeddings для текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов embeddings
        """
        embeddings = []
        
        for text in texts:
            try:
                data = {
                    "modelUri": f"emb://{config.YANDEX_FOLDER_ID}/{config.YANDEX_EMBEDDING_MODEL}",
                    "text": text
                }
                
                response = requests.post(
                    config.YANDEX_EMBEDDING_URL,
                    headers=self.headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    
                    if embedding:
                        embeddings.append(embedding)
                        logger.debug(f"Создан embedding для текста длиной {len(text)} символов")
                    else:
                        logger.warning("Получен пустой embedding")
                        embeddings.append([0.0] * config.VECTOR_DIMENSION)
                else:
                    logger.error(f"Ошибка создания embedding: {response.status_code}")
                    embeddings.append([0.0] * config.VECTOR_DIMENSION)
                    
            except Exception as e:
                logger.error(f"Ошибка при создании embedding: {e}")
                embeddings.append([0.0] * config.VECTOR_DIMENSION)
        
        return embeddings
    
    def combine_embeddings(self, original_embedding: List[float], 
                          hypothesis_embeddings: List[List[float]], 
                          weights: Optional[List[float]] = None) -> List[float]:
        """
        Комбинирует оригинальный embedding с гипотетическими.
        
        Args:
            original_embedding: Оригинальный вектор запроса
            hypothesis_embeddings: Векторы гипотетических документов
            weights: Веса для комбинирования
            
        Returns:
            Комбинированный вектор
        """
        try:
            if not hypothesis_embeddings:
                return original_embedding
            
            # Устанавливаем веса по умолчанию
            if weights is None:
                total_embeddings = 1 + len(hypothesis_embeddings)
                original_weight = 0.5  # 50% для оригинального запроса
                hypothesis_weight = 0.5 / len(hypothesis_embeddings)  # 50% делим между гипотезами
                weights = [original_weight] + [hypothesis_weight] * len(hypothesis_embeddings)
            
            # Нормализуем веса
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Комбинируем векторы
            combined = [0.0] * len(original_embedding)
            all_embeddings = [original_embedding] + hypothesis_embeddings
            
            for embedding, weight in zip(all_embeddings, weights):
                for i, value in enumerate(embedding):
                    combined[i] += value * weight
            
            logger.info(f"Скомбинирован embedding из {len(all_embeddings)} векторов")
            return combined
            
        except Exception as e:
            logger.error(f"Ошибка при комбинировании embeddings: {e}")
            return original_embedding


# Глобальный экземпляр процессора HyDE
hyde_processor = HyDEProcessor() 