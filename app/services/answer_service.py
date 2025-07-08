"""Сервис для генерации ответов на запросы пользователя."""

import requests
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import config
from app.core.logger import logger
from app.logics.opensearch import opensearch_worker
from app.logics.minio import minio_worker
from app.logics.query_analyzer import query_analyzer
from app.logics.postgres_storage import postgres_storage
from app.schemas.search import UserQuery


class AnswerService:
    """Сервис для генерации ответов на основе найденных документов."""
    
    def __init__(self):
        """Инициализация сервиса ответов."""
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }

    async def generate_answer(self, user_query: UserQuery) -> Dict[str, Any]:
        """
        Генерирует ответ на запрос пользователя на основе найденных документов.
        
        Args:
            user_query: Запрос пользователя
            
        Returns:
            Словарь с ответом и дополнительной информацией
        """
        import time
        start_time = time.time()
        search_time_ms = None
        generation_time_ms = None
        error_message = None
        success = True
        
        try:
            # Инициализируем таблицы если нужно
            try:
                postgres_storage.create_user_queries_table()
            except Exception as e:
                logger.warning(f"Ошибка создания таблицы user_queries: {e}")
            
            logger.info(f"Обрабатываем запрос: {user_query.query_text}")
            
            # Анализируем запрос и переводим при необходимости
            analysis = await query_analyzer.analyze_and_translate_query(user_query.query_text)
            logger.info(f"Анализ запроса: {analysis}")
            
            search_start = time.time()
            
            # Используем переведенный запрос для поиска
            search_query = analysis.get('translated_query', user_query.query_text)
            query_embedding = user_query.query_embed
            
            # HyDE: Генерируем гипотетические документы для улучшения поиска
            if user_query.use_hyde:
                logger.info("Применяем HyDE технологию")
                
                from app.logics.hyde import hyde_processor
                
                # Генерируем гипотетические документы для переведенного запроса
                hypotheses = await hyde_processor.generate_hypothetical_documents(
                    search_query, 
                    user_query.hyde_num_hypotheses
                )
                
                if hypotheses:
                    # Создаем embeddings для гипотез
                    hypothesis_embeddings = hyde_processor.create_embeddings(hypotheses)
                    
                    if hypothesis_embeddings and query_embedding:
                        # Комбинируем оригинальный embedding с гипотетическими
                        enhanced_embedding = hyde_processor.combine_embeddings(
                            query_embedding, 
                            hypothesis_embeddings
                        )
                        query_embedding = enhanced_embedding
                        logger.info(f"HyDE: Скомбинирован embedding из {len(hypothesis_embeddings) + 1} векторов")
            
            # Увеличиваем размер поиска если планируется реранжирование
            search_size = user_query.size * 2 if user_query.reranking else user_query.size
            
            # Выполняем поиск в OpenSearch
            search_results = opensearch_worker.execute_search(
                query_id=0,
                query_text=search_query,  # Используем переведенный запрос
                index=user_query.index_name,
                query_embed=query_embedding,  # Может быть улучшен через HyDE
                k=user_query.k,
                size=search_size,  # Увеличенный размер для реранжирования
                sematic=user_query.sematic,
                keyword=user_query.keyword,
                fields=user_query.fields,
                reranking=False,  # Отключаем встроенное реранжирование, используем продвинутое
                trashold=user_query.trashold,
                trashold_bertscore=user_query.trashold_bertscore
            )
            
            # Применяем ColBERT реранжирование если включено
            if user_query.reranking and search_results:
                logger.info("Применяем ColBERT реранжирование")
                
                from app.logics.reranking import colbert_reranker
                
                # Для реранжирования используем оригинальный запрос (для лучшего понимания контекста)
                reranked_results = await colbert_reranker.rerank_results(
                    query=user_query.query_text,  # Оригинальный запрос пользователя
                    results=search_results,
                    top_k=user_query.size
                )
                
                search_results = reranked_results
                logger.info(f"Реранжирование завершено. Финальных результатов: {len(search_results)}")
            else:
                # Если реранжирование отключено, просто ограничиваем количество
                search_results = search_results[:user_query.size]
            
            search_end = time.time()
            search_time_ms = int((search_end - search_start) * 1000)
            
            logger.info(f"Найдено {len(search_results)} документов за {search_time_ms} мс")
            
            if not search_results:
                response_text = "По вашему запросу не найдено релевантных документов. Попробуйте изменить формулировку или проверить правильность написания."
                
                # Логгируем запрос без результатов
                try:
                    postgres_storage.save_user_query(
                        query_text=user_query.query_text,
                        index_name=user_query.index_name,
                        search_params=user_query.dict(),
                        search_results=[],
                        response_text=response_text,
                        search_time_ms=search_time_ms,
                        total_time_ms=int((time.time() - start_time) * 1000),
                        technologies_used={
                            "hyde": user_query.use_hyde,
                            "reranking": user_query.reranking,
                            "semantic_weight": user_query.sematic,
                            "keyword_weight": user_query.keyword,
                            "hyde_hypotheses": user_query.hyde_num_hypotheses if user_query.use_hyde else 0,
                            "rerank_model": "colbert" if user_query.reranking else None
                        },
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Ошибка логгирования запроса: {e}")
                
                return {
                    "answer": response_text,
                    "sources": [],
                    "context": "",
                    "images": [],
                    "search_time_ms": search_time_ms,
                    "generation_time_ms": 0,
                    "total_time_ms": int((time.time() - start_time) * 1000)
                }
            
            generation_start = time.time()
            
            # Извлекаем контекст из результатов поиска для генерации ответа (ограничиваем для производительности LLM)
            context_parts = []
            for i, result in enumerate(search_results[:5], 1):  # Ограничиваем контекст топ-5 результатами для LLM
                text = result.get("text", "")
                if text:
                    context_parts.append(f"[Источник {i}]: {text}")
            
            # Формируем источники для отображения (используем запрошенное пользователем количество)
            sources = []
            display_count = min(len(search_results), user_query.size)  # Используем запрошенный размер
            
            for i, result in enumerate(search_results[:display_count], 1):
                text = result.get("text", "")
                file_name = result.get("file_name", "Неизвестный документ")
                page = result.get("page_number", "Неизвестная страница")
                score = result.get("score", 0)
                
                sources.append({
                    "index": i,
                    "file_name": file_name,
                    "page_number": page,
                    "score": round(score, 3),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                })
            
            context = "\n\n".join(context_parts)
            
            # Генерируем ответ через Yandex GPT
            prompt = f"""
            Контекст:
            {context}
            
            Вопрос: {user_query.query_text}
            
            Инструкции:
            - Дайте точный и информативный ответ на основе предоставленного контекста
            - Если информации недостаточно, честно скажите об этом
            - Не выдумывайте факты, которых нет в контексте
            - Отвечайте на русском языке
            - Будьте краткими, но исчерпывающими
            """
            
            answer = self._generate_answer_with_yandex_gpt(prompt)
            
            generation_end = time.time()
            generation_time_ms = int((generation_end - generation_start) * 1000)
            
            # Собираем изображения из найденных документов
            images = self._collect_images(search_results)
            
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Логгируем успешный запрос
            try:
                postgres_storage.save_user_query(
                    query_text=user_query.query_text,
                    index_name=user_query.index_name,
                    search_params=user_query.dict(),
                    search_results=search_results[:10],  # Сохраняем только топ-10 результатов
                    response_text=answer,
                    response_images=[{"file_name": img.get("file_name"), "image_name": img.get("image_name"), "image_url": img.get("image_url")} for img in images],
                    search_time_ms=search_time_ms,
                    generation_time_ms=generation_time_ms,
                    total_time_ms=total_time_ms,
                    technologies_used={
                        "hyde": user_query.use_hyde,
                        "reranking": user_query.reranking,
                        "semantic_weight": user_query.sematic,
                        "keyword_weight": user_query.keyword,
                        "query_analysis": analysis,
                        "hyde_hypotheses": user_query.hyde_num_hypotheses if user_query.use_hyde else 0,
                        "rerank_model": "colbert" if user_query.reranking else None
                    },
                    success=True
                )
            except Exception as e:
                logger.warning(f"Ошибка логгирования запроса: {e}")
            
            logger.info(f"Ответ сгенерирован за {generation_time_ms} мс. Всего: {total_time_ms} мс")
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "images": images,
                "search_time_ms": search_time_ms,
                "generation_time_ms": generation_time_ms,
                "total_time_ms": total_time_ms,
                "search_info": {
                    "technologies": {
                        "hyde": user_query.use_hyde,
                        "reranking": user_query.reranking,
                        "hyde_hypotheses": user_query.hyde_num_hypotheses if user_query.use_hyde else 0,
                        "rerank_model": "colbert" if user_query.reranking else None
                    },
                    "translation": {
                        "query_language": analysis.get('query_language', 'unknown'),
                        "document_language": analysis.get('document_language', 'unknown'),
                        "translated": analysis.get('needs_translation', False),
                        "translated_query": analysis.get('translated_query', user_query.query_text) if analysis.get('needs_translation', False) else None
                    }
                }
            }
            
        except Exception as e:
            error_message = str(e)
            success = False
            total_time_ms = int((time.time() - start_time) * 1000)
            
            logger.error(f"Ошибка при генерации ответа: {e}")
            
            # Логгируем ошибочный запрос
            try:
                postgres_storage.save_user_query(
                    query_text=user_query.query_text,
                    index_name=user_query.index_name,
                    search_params=user_query.dict(),
                    search_time_ms=search_time_ms,
                    generation_time_ms=generation_time_ms,
                    total_time_ms=total_time_ms,
                    technologies_used={
                        "hyde": user_query.use_hyde,
                        "reranking": user_query.reranking,
                        "semantic_weight": user_query.sematic,
                        "keyword_weight": user_query.keyword,
                        "hyde_hypotheses": user_query.hyde_num_hypotheses if user_query.use_hyde else 0,
                        "rerank_model": "colbert" if user_query.reranking else None
                    },
                    success=False,
                    error_message=error_message
                )
            except Exception as log_error:
                logger.warning(f"Ошибка логгирования ошибочного запроса: {log_error}")
            
            raise
    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Формирует контекст для генерации ответа."""
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            text = result.get("text", "")
            page = result.get("page_number", "неизвестно")
            file_name = result.get("file_name", "неизвестно")
            
            context_part = f"""
Документ {idx}:
Источник: {file_name}, страница {page}
Содержание: {text}
"""
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Генерирует ответ через Yandex GPT API."""
        try:
            system_prompt = """Ты эксперт-аналитик документов. Твоя работа - создавать подробные, структурированные ответы на основе приложенных документов."""

            user_message = f"""
ВОПРОС: {query}

ДОКУМЕНТЫ ДЛЯ АНАЛИЗА:
{context}

ИНСТРУКЦИЯ: Создай подробный, структурированный ответ на вопрос, используя ТОЛЬКО информацию из документов выше. 

ТРЕБОВАНИЯ К ОТВЕТУ:
1. Ответ должен быть структурированным и содержать информацию из документов
2. Ответ должен быть написан на русском языке
3. Ответ должен быть без перечислений, чисто ответ по источникам

Дай полный и качественный ответ на основе предоставленных документов!
"""

            data = {
                "modelUri": f"gpt://{config.YANDEX_FOLDER_ID}/{config.YANDEX_LLM_MODEL}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.1,  # Очень низкая температура для точности
                    "maxTokens": 3000    # Больше токенов для полных ответов
                },
                "messages": [
                    {
                        "role": "system",
                        "text": system_prompt
                    },
                    {
                        "role": "user", 
                        "text": user_message
                    }
                ]
            }
            
            logger.info(f"Отправляем запрос к YandexGPT. Длина пользовательского сообщения: {len(user_message)} символов")
            logger.debug(f"Полный запрос к YandexGPT: {data}")
            
            response = requests.post(
                config.YANDEX_LLM_URL,
                headers=self.headers,
                json=data,
                timeout=60
            )
            
            logger.info(f"Ответ от YandexGPT: статус {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Полный ответ от YandexGPT: {result}")
                
                alternatives = result.get("result", {}).get("alternatives", [])
                
                if alternatives:
                    answer = alternatives[0].get("message", {}).get("text", "")
                    logger.info(f"Получен ответ от YandexGPT (длина: {len(answer)} символов): {answer[:200]}...")
                    
                    # Обеспечиваем, что ответ всегда на русском языке
                    answer = await query_analyzer.ensure_russian_response(answer)
                    
                    return answer if answer else "Не удалось сгенерировать ответ."
                else:
                    logger.warning("Yandex GPT вернул пустой ответ")
                    logger.warning(f"Структура ответа: {result}")
                    return "Не удалось сгенерировать ответ."
            else:
                logger.error(f"Ошибка Yandex GPT API: {response.status_code} - {response.text}")
                return f"Ошибка при генерации ответа: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа через Yandex GPT: {e}")
            return f"Произошла ошибка при генерации ответа: {str(e)}"
    
    def _collect_images(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Собирает все изображения из найденных документов и создает presigned URL-ы для них."""
        # Собираем информацию об изображениях по документам
        result_images = []
        unique_images = set()
        
        for result in search_results:
            file_name = result.get("file_name")
            images = result.get("images", [])
            
            if file_name and images and isinstance(images, list):
                for image_name in images:
                    # Создаем уникальный ключ для избежания дублирования
                    unique_key = f"{file_name}_{image_name}"
                    
                    if unique_key not in unique_images:
                        unique_images.add(unique_key)
                        
                        # Генерируем presigned URL для изображения
                        image_url = minio_worker.get_image_url(file_name, image_name, expires=3600)
                        
                        if image_url:
                            result_images.append({
                                'file_name': file_name,
                                'image_name': image_name,
                                'image_url': image_url,
                                'page_number': result.get("page_number", "неизвестно")
                            })
        
        logger.info(f"Найдено {len(result_images)} уникальных изображений из MinIO")
        
        return result_images
    
    def _format_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Форматирует источники для отображения."""
        sources = []
        
        for idx, result in enumerate(search_results, 1):
            source = {
                "index": idx,
                "file_name": result.get("file_name", "неизвестно"),
                "page_number": result.get("page_number", "неизвестно"),
                "paragraph_index": result.get("paragraph_index", "неизвестно"),
                "score": round(result.get("score", 0), 3),
                "text_preview": result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get("text", ""),
                "chunk_id": result.get("chunk_id", ""),
                "images_count": len(result.get("images", []))
            }
            sources.append(source)
        
        return sources

    def _create_direct_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Создает структурированный ответ на основе найденных документов без использования YandexGPT.
        
        Args:
            query: Вопрос пользователя
            search_results: Результаты поиска
            
        Returns:
            Полный структурированный ответ
        """
        try:
            # Анализируем запрос для создания подходящего ответа
            query_lower = query.lower()
            
            answer_parts = []
            
            # 1. Введение и определение
            if "что такое" in query_lower or "определение" in query_lower:
                answer_parts.append("## Определение")
                top_result = search_results[0] if search_results else None
                if top_result:
                    text = top_result.get("text", "")
                    key_sentences = self._extract_key_sentences(text, query)
                    if key_sentences:
                        answer_parts.append(f"На основе анализа документов: {key_sentences}")
                    
            # 2. Основное содержание
            answer_parts.append("## Детальная информация")
            
            for idx, result in enumerate(search_results[:3], 1):
                text = result.get("text", "")
                page = result.get("page_number", "неизвестно")
                file_name = result.get("file_name", "неизвестно")
                
                # Извлекаем ключевые предложения
                key_info = self._extract_relevant_content(text, query)
                
                section = f"""
### Источник {idx}: {file_name} (страница {page})

{key_info}
"""
                answer_parts.append(section.strip())
            
            # 3. Дополнительные детали
            if len(search_results) > 3:
                additional_info = []
                for result in search_results[3:6]:  # Еще 3 документа
                    text = result.get("text", "")
                    page = result.get("page_number", "неизвестно")
                    file_name = result.get("file_name", "неизвестно")
                    
                    key_phrase = self._extract_key_phrase(text, query)
                    if key_phrase:
                        additional_info.append(f"• {key_phrase} ({file_name}, стр. {page})")
                
                if additional_info:
                    answer_parts.append("## Дополнительная информация")
                    answer_parts.extend(additional_info)
            
            # 4. Заключение
            total_docs = len(search_results)
            images_count = sum(1 for result in search_results if result.get("images", []))
            
            conclusion = f"""
## Заключение

Найдено {total_docs} релевантных фрагментов документов по запросу "{query}". """
            
            if images_count > 0:
                conclusion += f"К материалам прикреплено {images_count} изображений для дополнительного изучения."
            
            answer_parts.append(conclusion.strip())
            
            return "\n\n".join(answer_parts)
            
        except Exception as e:
            logger.error(f"Ошибка при создании прямого ответа: {e}")
            return self._create_simple_fallback(query, search_results)
    
    def _extract_key_sentences(self, text: str, query: str) -> str:
        """Извлекает ключевые предложения из текста относительно запроса."""
        # Разбиваем на предложения
        sentences = text.split('. ')
        query_words = set(query.lower().split())
        
        # Ищем предложения с наибольшим пересечением с запросом
        best_sentences = []
        for sentence in sentences[:5]:  # Первые 5 предложений
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                best_sentences.append((sentence, overlap))
        
        # Сортируем по релевантности и берем лучшие
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        return '. '.join([s[0] for s in best_sentences[:2]]) + '.'
    
    def _extract_relevant_content(self, text: str, query: str) -> str:
        """Извлекает наиболее релевантный контент из текста."""
        # Если текст короткий, возвращаем целиком
        if len(text) <= 400:
            return text
            
        # Ищем наиболее релевантные части
        query_words = set(query.lower().split())
        sentences = text.split('. ')
        
        relevant_parts = []
        for i, sentence in enumerate(sentences):
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            if overlap > 0:
                # Берем контекст вокруг релевантного предложения
                start = max(0, i-1)
                end = min(len(sentences), i+2)
                context = '. '.join(sentences[start:end])
                relevant_parts.append(context)
                
                if len('. '.join(relevant_parts)) > 300:
                    break
        
        result = '. '.join(relevant_parts)
        return result if result else text[:400] + "..."
    
    def _extract_key_phrase(self, text: str, query: str) -> str:
        """Извлекает ключевую фразу из текста."""
        sentences = text.split('. ')
        query_words = set(query.lower().split())
        
        for sentence in sentences[:3]:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) > 0:
                # Возвращаем первую релевантную фразу
                return sentence[:150] + ("..." if len(sentence) > 150 else "")
        
        return text[:100] + "..." if len(text) > 100 else text
    
    def _create_simple_fallback(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Создает простой fallback ответ."""
        return f"Найдено {len(search_results)} документов по запросу '{query}'. Просмотрите источники и изображения для получения подробной информации."

    def _generate_answer_with_yandex_gpt(self, prompt: str) -> str:
        """
        Генерирует ответ через Yandex GPT API.
        
        Args:
            prompt: Промпт для генерации ответа
            
        Returns:
            Сгенерированный ответ
        """
        try:
            payload = {
                "modelUri": f"gpt://{config.YANDEX_FOLDER_ID}/{config.YANDEX_LLM_MODEL}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.1,
                    "maxTokens": 2000
                },
                "messages": [
                    {
                        "role": "user",
                        "text": prompt
                    }
                ]
            }
            
            response = requests.post(
                config.YANDEX_COMPLETION_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "")
                
                if not answer:
                    logger.warning("Пустой ответ от Yandex GPT")
                    return "Извините, не удалось сгенерировать ответ на основе найденной информации."
                
                return answer
            else:
                logger.error(f"Ошибка API Yandex GPT: {response.status_code}, {response.text}")
                return "Извините, произошла ошибка при генерации ответа."
                
        except Exception as e:
            logger.error(f"Ошибка при обращении к Yandex GPT API: {e}")
            return "Извините, произошла ошибка при генерации ответа."


class RAGEvaluationService:
    """Сервис для оценки качества RAG системы."""
    
    def __init__(self):
        """Инициализация сервиса оценки."""
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
    
    async def evaluate_rag_performance(self, test_data: List[Dict[str, str]], 
                                     index_name: str, 
                                     search_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Оценивает производительность RAG системы на тестовых данных.
        
        Args:
            test_data: Список словарей с полями 'question' и 'expected_answer'
            index_name: Индекс для поиска
            search_params: Параметры поиска (опционально)
            
        Returns:
            Результаты тестирования с метриками и анализом
        """
        try:
            logger.info(f"Начинаем оценку RAG на {len(test_data)} вопросах")
            
            # Инициализируем PostgreSQL таблицу
            try:
                postgres_storage.create_table()
                logger.info("PostgreSQL таблица инициализирована")
            except Exception as e:
                logger.warning(f"Ошибка инициализации PostgreSQL: {e}")
            
            # Генерируем уникальный ID сессии тестирования
            test_session_id = postgres_storage.generate_session_id()
            logger.info(f"Создана сессия тестирования: {test_session_id}")
            
            # Параметры поиска по умолчанию
            default_params = {
                "sematic": 0.7,
                "keyword": 0.3,
                "size": 5,
                "use_hyde": False,
                "reranking": False
            }
            
            if search_params:
                default_params.update(search_params)
            
            results = []
            correct_answers = 0
            
            for i, test_case in enumerate(test_data, 1):
                import time
                test_start_time = time.time()
                
                logger.info(f"Обрабатываем тест {i}/{len(test_data)}")
                
                question = test_case.get('question', '').strip()
                expected_answer = test_case.get('expected_answer', '').strip()
                
                if not question or not expected_answer:
                    logger.warning(f"Пропускаем тест {i}: пустой вопрос или ответ")
                    continue
                
                # Получаем ответ от RAG системы
                search_start_time = time.time()
                error_message = None
                rag_answer = ""
                search_results = []
                sources_count = 0
                
                try:
                    user_query = UserQuery(
                        query_text=question,
                        index_name=index_name,
                        **default_params
                    )
                    
                    generation_start_time = time.time()
                    rag_response = await answer_service.generate_answer(user_query)
                    generation_time_ms = int((time.time() - generation_start_time) * 1000)
                    
                    rag_answer = rag_response.get("answer", "")
                    search_results = rag_response.get("search_results", [])
                    sources_count = len(rag_response.get("sources", []))
                    
                except Exception as e:
                    logger.error(f"Ошибка при получении ответа для вопроса {i}: {e}")
                    rag_answer = f"Ошибка: {str(e)}"
                    error_message = str(e)
                    generation_time_ms = int((time.time() - generation_start_time) * 1000) if 'generation_start_time' in locals() else None
                
                search_time_ms = int((time.time() - search_start_time) * 1000)
                
                # Сравниваем ответы через LLM
                comparison_result = await self._compare_answers(
                    question, expected_answer, rag_answer
                )
                
                score = comparison_result.get("score", 0)
                explanation = comparison_result.get("explanation", "")
                
                if score == 1:
                    correct_answers += 1
                
                total_time_ms = int((time.time() - test_start_time) * 1000)
                
                # Формируем метрики
                metrics = {
                    "accuracy": score,
                    "score": score,
                    "explanation": explanation,
                    "sources_count": sources_count
                }
                
                # Формируем информацию о технологиях
                technologies = {
                    "hyde": default_params.get("use_hyde", False),
                    "reranking": default_params.get("reranking", False),
                    "semantic_weight": default_params.get("sematic", 0.7),
                    "keyword_weight": default_params.get("keyword", 0.3)
                }
                
                # Сохраняем результат в PostgreSQL
                try:
                    postgres_storage.save_test_result(
                        test_session_id=test_session_id,
                        question=question,
                        expected_answer=expected_answer,
                        actual_answer=rag_answer,
                        search_results=search_results,
                        search_params=default_params,
                        metrics=metrics,
                        index_name=index_name,
                        technologies=technologies,
                        search_time_ms=search_time_ms,
                        generation_time_ms=generation_time_ms,
                        total_time_ms=total_time_ms,
                        error_message=error_message
                    )
                    logger.debug(f"Результат теста {i} сохранен в PostgreSQL")
                except Exception as e:
                    logger.warning(f"Ошибка сохранения в PostgreSQL: {e}")
                
                test_result = {
                    "test_id": i,
                    "question": question,
                    "expected_answer": expected_answer,
                    "rag_answer": rag_answer,
                    "score": score,
                    "explanation": explanation,
                    "sources_count": sources_count,
                    "search_time_ms": search_time_ms,
                    "generation_time_ms": generation_time_ms,
                    "total_time_ms": total_time_ms
                }
                results.append(test_result)
                
                logger.info(f"Тест {i}: оценка {score}, источников: {sources_count}, время: {total_time_ms}мс")
            
            # Вычисляем метрики
            total_tests = len(results)
            accuracy = correct_answers / total_tests if total_tests > 0 else 0
            
            # Получаем общий анализ от модели
            overall_analysis = await self._generate_overall_analysis(results, accuracy)
            
            evaluation_results = {
                "total_tests": total_tests,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "accuracy_percentage": round(accuracy * 100, 2),
                "search_parameters": default_params,
                "test_results": results,
                "overall_analysis": overall_analysis,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_session_id": test_session_id,
                "postgresql_saved": True
            }
            
            logger.info(f"Оценка завершена. Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%). Сессия: {test_session_id}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Ошибка при оценке RAG: {e}")
            raise
    
    async def _compare_answers(self, question: str, expected: str, actual: str) -> Dict[str, Any]:
        """
        Сравнивает ожидаемый и фактический ответы через LLM.
        
        Args:
            question: Вопрос
            expected: Ожидаемый ответ
            actual: Фактический ответ от RAG
            
        Returns:
            Результат сравнения с оценкой и объяснением
        """
        try:
            system_prompt = """Ты эксперт по оценке качества ответов на вопросы. 
Твоя задача - сравнить два ответа на один вопрос и определить, совпадают ли они по смыслу.

КРИТЕРИИ ОЦЕНКИ:
- Оценка 1: ответы совпадают по основному смыслу, передают одну и ту же информацию
- Оценка 0: ответы не совпадают, передают разную информацию или один из них неточный

Ответ должен быть в формате:
ОЦЕНКА: [0 или 1]
ОБЪЯСНЕНИЕ: [краткое объяснение почему такая оценка]"""

            user_message = f"""
ВОПРОС: {question}

ОЖИДАЕМЫЙ ОТВЕТ: {expected}

ФАКТИЧЕСКИЙ ОТВЕТ: {actual}

Оцени, совпадают ли ответы по смыслу."""

            data = {
                "modelUri": f"gpt://{config.YANDEX_FOLDER_ID}/{config.YANDEX_LLM_MODEL}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.1,
                    "maxTokens": 500
                },
                "messages": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": user_message}
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
                    comparison_text = alternatives[0].get("message", {}).get("text", "")
                    
                    # Парсим ответ
                    score = 0
                    explanation = "Не удалось распарсить ответ"
                    
                    lines = comparison_text.split('\n')
                    for line in lines:
                        if line.startswith('ОЦЕНКА:'):
                            try:
                                score = int(line.split(':')[1].strip())
                            except:
                                pass
                        elif line.startswith('ОБЪЯСНЕНИЕ:'):
                            explanation = line.split(':', 1)[1].strip()
                    
                    return {
                        "score": score,
                        "explanation": explanation,
                        "raw_response": comparison_text
                    }
            
            logger.warning(f"Ошибка при сравнении ответов: {response.status_code}")
            return {"score": 0, "explanation": "Ошибка при сравнении через LLM"}
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении ответов: {e}")
            return {"score": 0, "explanation": f"Ошибка: {str(e)}"}
    
    async def _generate_overall_analysis(self, results: List[Dict[str, Any]], 
                                       accuracy: float) -> str:
        """
        Генерирует общий анализ результатов тестирования.
        
        Args:
            results: Результаты всех тестов
            accuracy: Общая точность
            
        Returns:
            Анализ результатов
        """
        try:
            # Подготавливаем статистику для анализа
            correct_count = sum(1 for r in results if r.get("score", 0) == 1)
            incorrect_count = len(results) - correct_count
            
            # Примеры неудачных ответов
            failed_examples = [r for r in results if r.get("score", 0) == 0][:3]
            successful_examples = [r for r in results if r.get("score", 0) == 1][:3]
            
            system_prompt = """Ты эксперт по анализу качества RAG систем. 
Проанализируй результаты тестирования и дай развернутую оценку работы системы.

В анализе укажи:
1. Общую оценку качества (отлично/хорошо/удовлетворительно/плохо)
2. Основные проблемы, которые ты видишь
3. Рекомендации по улучшению
4. Анализ типичных ошибок"""

            examples_text = ""
            if failed_examples:
                examples_text += "\nПРИМЕРЫ НЕУДАЧНЫХ ОТВЕТОВ:\n"
                for i, example in enumerate(failed_examples, 1):
                    examples_text += f"""
Пример {i}:
Вопрос: {example['question']}
Ожидалось: {example['expected_answer'][:200]}...
Получено: {example['rag_answer'][:200]}...
Причина: {example['explanation']}
"""
            
            if successful_examples:
                examples_text += "\nПРИМЕРЫ УСПЕШНЫХ ОТВЕТОВ:\n"
                for i, example in enumerate(successful_examples, 1):
                    examples_text += f"""
Пример {i}:
Вопрос: {example['question']}
Ожидалось: {example['expected_answer'][:200]}...
Получено: {example['rag_answer'][:200]}...
"""

            user_message = f"""
РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ RAG СИСТЕМЫ:
- Всего тестов: {len(results)}
- Правильных ответов: {correct_count}
- Неправильных ответов: {incorrect_count}
- Точность: {accuracy:.3f} ({accuracy*100:.1f}%)

{examples_text}

Проанализируй эти результаты и дай рекомендации по улучшению."""

            data = {
                "modelUri": f"gpt://{config.YANDEX_FOLDER_ID}/{config.YANDEX_LLM_MODEL}",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.3,
                    "maxTokens": 2000
                },
                "messages": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": user_message}
                ]
            }
            
            response = requests.post(
                config.YANDEX_LLM_URL,
                headers=self.headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                alternatives = result.get("result", {}).get("alternatives", [])
                
                if alternatives:
                    return alternatives[0].get("message", {}).get("text", "")
            
            return f"Точность системы составляет {accuracy*100:.1f}%. Анализ недоступен."
            
        except Exception as e:
            logger.error(f"Ошибка при генерации анализа: {e}")
            return f"Точность системы составляет {accuracy*100:.1f}%. Ошибка при генерации анализа: {str(e)}"


# Глобальные экземпляры сервисов
answer_service = AnswerService()
rag_evaluation_service = RAGEvaluationService() 