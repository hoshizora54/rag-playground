"""
Анализатор запросов с автоматическим определением языка и переводом.

Функции:
1. Определение языка запроса пользователя
2. Определение языка документов в индексе
3. Автоматический перевод запроса для улучшения поиска
4. Обеспечение ответов всегда на русском языке
"""

import re
import requests
from typing import Dict, Any, Optional, List
from app.core.config import config
from app.core.logger import logger


class QueryAnalyzer:
    """Анализатор запросов с поддержкой многоязычности."""
    
    def __init__(self):
        """Инициализация анализатора."""
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
    
    async def analyze_and_translate_query(self, query: str, document_samples: List[str] = None) -> Dict[str, Any]:
        """
        Анализирует запрос и переводит при необходимости.
        
        Args:
            query: Запрос пользователя
            document_samples: Образцы текста из документов
            
        Returns:
            Результат анализа с переведенным запросом
        """
        try:
            logger.info(f"Анализ запроса: {query}")
            
            # 1. Определяем язык запроса
            query_language = self._detect_language(query)
            
            # 2. Определяем язык документов
            document_language = "unknown"
            if document_samples:
                document_language = self._detect_document_language(document_samples)
            
            # 3. Переводим запрос если нужно
            translated_query = query
            needs_translation = False
            
            if self._should_translate(query_language, document_language):
                if query_language == "ru" and document_language == "en":
                    # Переводим с русского на английский
                    translated = await self._translate_text(query, "ru", "en")
                    if translated:
                        translated_query = translated
                        needs_translation = True
                        logger.info(f"Запрос переведен: {query} -> {translated_query}")
                
                elif query_language == "en" and document_language == "ru":
                    # Переводим с английского на русский
                    translated = await self._translate_text(query, "en", "ru")
                    if translated:
                        translated_query = translated
                        needs_translation = True
                        logger.info(f"Запрос переведен: {query} -> {translated_query}")
            
            result = {
                "original_query": query,
                "translated_query": translated_query,
                "query_language": query_language,
                "document_language": document_language,
                "needs_translation": needs_translation,
                "translation_success": needs_translation and translated_query != query
            }
            
            logger.info(f"Результат анализа: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при анализе запроса: {e}")
            return {
                "original_query": query,
                "translated_query": query,
                "query_language": "unknown",
                "document_language": "unknown",
                "needs_translation": False,
                "translation_success": False
            }
    
    def _detect_language(self, text: str) -> str:
        """
        Определяет язык текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Код языка: 'ru', 'en', 'unknown'
        """
        # Простая эвристика на основе символов
        russian_chars = len(re.findall(r'[а-яё]', text.lower()))
        english_chars = len(re.findall(r'[a-z]', text.lower()))
        
        total_chars = russian_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        russian_ratio = russian_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if russian_ratio > 0.6:
            return "ru"
        elif english_ratio > 0.6:
            return "en"
        else:
            # Смешанный текст - определяем по доминирующему языку
            return "ru" if russian_chars > english_chars else "en"
    
    def _detect_document_language(self, document_samples: List[str]) -> str:
        """
        Определяет основной язык документов.
        
        Args:
            document_samples: Образцы текста из документов
            
        Returns:
            Код языка документов
        """
        if not document_samples:
            return "unknown"
        
        # Анализируем первые несколько образцов
        combined_text = " ".join(document_samples[:3])
        if len(combined_text) < 50:  # Если текста мало, берем больше образцов
            combined_text = " ".join(document_samples[:5])
        
        return self._detect_language(combined_text)
    
    def _should_translate(self, query_lang: str, doc_lang: str) -> bool:
        """
        Определяет, нужно ли переводить запрос.
        
        Args:
            query_lang: Язык запроса
            doc_lang: Язык документов
            
        Returns:
            True если нужен перевод
        """
        # Переводим только если языки разные и оба определены
        return (query_lang != doc_lang and 
                query_lang in ["ru", "en"] and 
                doc_lang in ["ru", "en"])
    
    async def _translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Переводит текст с помощью Yandex GPT.
        
        Args:
            text: Текст для перевода
            source_lang: Исходный язык (ru/en)
            target_lang: Целевой язык (ru/en)
            
        Returns:
            Переведенный текст или None
        """
        try:
            lang_names = {"ru": "русский", "en": "английский"}
            source_name = lang_names.get(source_lang, source_lang)
            target_name = lang_names.get(target_lang, target_lang)
            
            system_prompt = f"Переведи текст с {source_name} языка на {target_name} язык. Отвечай только переводом, без дополнительных комментариев."
            
            data = {
                "modelUri": f"gpt://{config.YANDEX_FOLDER_ID}/{config.YANDEX_LLM_MODEL}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.1,  # Низкая температура для точного перевода
                    "maxTokens": 200
                },
                "messages": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": text}
                ]
            }
            
            response = requests.post(
                config.YANDEX_LLM_URL,
                headers=self.headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                alternatives = result.get("result", {}).get("alternatives", [])
                
                if alternatives:
                    translation = alternatives[0].get("message", {}).get("text", "")
                    return translation.strip() if translation else None
            
            logger.warning(f"Ошибка перевода: HTTP {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при переводе: {e}")
            return None
    
    async def ensure_russian_response(self, response_text: str, detected_language: str = None) -> str:
        """
        Обеспечивает, что ответ всегда на русском языке.
        
        Args:
            response_text: Текст ответа
            detected_language: Определенный язык ответа
            
        Returns:
            Ответ на русском языке
        """
        try:
            # Если язык не определен, определяем его
            if not detected_language:
                detected_language = self._detect_language(response_text)
            
            # Если ответ не на русском, переводим
            if detected_language == "en":
                logger.info("Ответ на английском, переводим на русский")
                translated = await self._translate_text(response_text, "en", "ru")
                if translated:
                    return translated
            
            return response_text
            
        except Exception as e:
            logger.error(f"Ошибка при обеспечении русского ответа: {e}")
            return response_text


# Глобальный экземпляр анализатора
query_analyzer = QueryAnalyzer() 