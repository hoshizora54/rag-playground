"""Сервис для поиска документов с продвинутыми технологиями."""

import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException

from app.core.logger import logger
from app.logics.opensearch import opensearch_worker
from app.logics.hyde import hyde_processor
from app.logics.reranking import colbert_reranker
from app.logics.query_analyzer import query_analyzer
from app.schemas.search import UserQuery, SearchResult, CheckIndex, IndexCheckResponse


async def _get_document_samples(index_name: str) -> List[str]:
    """
    Получает образцы текста из документов для определения языка.
    
    Args:
        index_name: Имя индекса
        
    Returns:
        Список образцов текста
    """
    try:
        # Выполняем простой поиск для получения образцов
        sample_results = opensearch_worker.execute_search(
            query_id=0,
            query_text="*",  # Получаем любые документы
            index=index_name,
            query_embed=None,
            k=5,
            size=5,
            sematic=0.0,
            keyword=1.0,
            fields=["text"],
            reranking=False,
            trashold=0.0,
            trashold_bertscore=0.0
        )
        
        if isinstance(sample_results, list):
            samples = [result.get("text", "") for result in sample_results[:5]]
        else:
            results = sample_results.get("results", [])
            samples = [result.get("text", "") for result in results[:5]]
        
        # Фильтруем пустые образцы
        samples = [sample for sample in samples if sample.strip()]
        
        logger.info(f"Получено {len(samples)} образцов документов для определения языка")
        return samples
        
    except Exception as e:
        logger.warning(f"Ошибка при получении образцов документов: {e}")
        return []


async def handle_query(user_query: UserQuery) -> SearchResult:
    """
    Обрабатывает поисковый запрос пользователя с продвинутыми RAG технологиями.
    
    Args:
        user_query: Запрос пользователя
        
    Returns:
        Результаты поиска
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Обрабатываем запрос: {user_query.query_text}")
        
        # Шаг 1: Получаем образцы документов для определения языка
        document_samples = await _get_document_samples(user_query.index_name)
        
        # Шаг 2: Анализируем запрос и переводим при необходимости
        analysis_result = await query_analyzer.analyze_and_translate_query(
            user_query.query_text, 
            document_samples
        )
        
        # Используем переведенный запрос для поиска
        search_query = analysis_result['translated_query']
        
        logger.info(f"Язык запроса: {analysis_result['query_language']}, "
                   f"Язык документов: {analysis_result['document_language']}")
        if analysis_result['needs_translation']:
            logger.info(f"Запрос переведен для поиска: {user_query.query_text} -> {search_query}")
        
        # Шаг 3: Подготавливаем embeddings для поиска
        query_embedding = user_query.query_embed
        logger.info(f"Query embedding в search_service: {type(query_embedding)}, длина: {len(query_embedding) if query_embedding else 'None'}")
        
        # HyDE: Генерируем гипотетические документы для улучшения поиска
        if user_query.use_hyde:
            logger.info("Применяем HyDE технологию")
            
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
        
        # Параметры для основного поиска
        kwargs = {
            "query_embed": query_embedding,
            "k": user_query.k,
            "size": user_query.size * 2 if user_query.reranking else user_query.size,  # Больше результатов для реранжирования
            "sematic": user_query.sematic,
            "keyword": user_query.keyword,
            "fields": user_query.fields,
            "reranking": False,  # Отключаем встроенное реранжирование, используем продвинутое
            "trashold": user_query.trashold,
            "trashold_bertscore": user_query.trashold_bertscore
        }

        # Шаг 4: Выполняем основной поиск с переведенным запросом
        logger.info(f"Вызываем OpenSearch с параметрами: query_embed={type(kwargs.get('query_embed'))}, размер={len(kwargs.get('query_embed', [])) if kwargs.get('query_embed') else 'None'}")
        raw_results = opensearch_worker.execute_search(
            query_id=kwargs.get("query_id", 0),
            query_text=search_query,  # Используем переведенный запрос
            index=user_query.index_name, 
            **kwargs
        )
        
        # Приводим результаты к единому формату
        if isinstance(raw_results, list):
            search_results = raw_results
        else:
            search_results = raw_results.get("results", [])
        
        logger.info(f"Основной поиск вернул {len(search_results)} результатов")
        
        # Шаг 5: Применяем продвинутое реранжирование
        final_results = search_results
        
        if user_query.reranking and search_results:
            logger.info("Применяем ColBERT реранжирование")
            
            # Для реранжирования используем оригинальный запрос (для лучшего понимания контекста)
            reranked_results = await colbert_reranker.rerank_results(
                query=user_query.query_text,  # Оригинальный запрос пользователя
                results=search_results,
                top_k=user_query.size
            )
            
            final_results = reranked_results
            logger.info(f"Реранжирование завершено. Финальных результатов: {len(final_results)}")
        else:
            # Если реранжирование отключено, просто ограничиваем количество
            final_results = search_results[:user_query.size]
        
        # Добавляем метаинформацию о примененных технологиях и переводе
        for i, result in enumerate(final_results):
            result["_applied_technologies"] = {
                "hyde": user_query.use_hyde,
                "reranking": user_query.reranking,
                "rerank_model": "colbert" if user_query.reranking else None,
                "position": i + 1
            }
            result["_translation_info"] = {
                "original_query": analysis_result['original_query'],
                "translated_query": analysis_result['translated_query'],
                "query_language": analysis_result['query_language'],
                "document_language": analysis_result['document_language'],
                "translation_used": analysis_result['needs_translation'],
                "translation_success": analysis_result['translation_success']
            }
        
        # Вычисляем время поиска
        search_time = time.time() - start_time
        
        logger.info(f"Поиск завершен за {search_time:.2f} сек. Возвращаем {len(final_results)} результатов")
        return SearchResult(
            query=user_query.query_text,
            results=final_results,
            search_time=search_time
        )
        
    except Exception as e:
        search_time = time.time() - start_time
        logger.exception(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения результатов: {str(e)}")


async def handle_index_check(request: CheckIndex) -> IndexCheckResponse:
    """
    Проверяет наличие файла в индексе.
    
    Args:
        request: Запрос на проверку
        
    Returns:
        Статус наличия файла в индексе
    """
    try:
        status = opensearch_worker.check_file_in_index(
            file_name=request.source,
            index_name=request.index_name
        )
        return IndexCheckResponse(status=status)
        
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 