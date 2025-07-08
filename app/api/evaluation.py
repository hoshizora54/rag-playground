"""API для оценки качества RAG системы."""

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from typing import List, Optional
import pandas as pd
import io
from app.services.answer_service import rag_evaluation_service
from app.schemas.base import BaseResponse
from app.core.logger import logger
from app.logics.postgres_storage import postgres_storage

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

@router.post("/batch", response_model=BaseResponse)
async def evaluate_batch(
    index_name: str = Form(...),
    semantic_weight: float = Form(0.7),
    keyword_weight: float = Form(0.3),
    k: int = Form(5),
    size: int = Form(10),
    use_hyde: bool = Form(False),
    hyde_num_hypotheses: int = Form(1),
    reranking: bool = Form(False),
    file: UploadFile = File(...)
):
    """
    Запускает пакетную оценку качества RAG системы.
    
    Args:
        index_name: Название индекса
        semantic_weight: Вес семантического поиска
        keyword_weight: Вес ключевого поиска  
        k: Количество ближайших векторов
        size: Размер результатов
        use_hyde: Использовать HyDE
        hyde_num_hypotheses: Количество гипотез для HyDE
        reranking: Использовать ColBERT реранжирование
        file: CSV файл с тестовыми данными
        
    Returns:
        Результаты оценки
    """
    try:
        # Читаем CSV файл
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Проверяем наличие нужных колонок
        required_columns = ['question', 'expected_answer']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV файл должен содержать колонки: {required_columns}"
            )
        
        # Конвертируем в список словарей
        test_data = df.to_dict('records')
        
        # Параметры поиска
        search_params = {
            'semantic_weight': semantic_weight,
            'keyword_weight': keyword_weight,
            'k': k,
            'size': size,
            'use_hyde': use_hyde,
            'hyde_num_hypotheses': hyde_num_hypotheses,
            'reranking': reranking
        }
        
        # Запускаем оценку
        result = await rag_evaluation_service.evaluate_rag_performance(
            test_data=test_data,
            index_name=index_name,
            search_params=search_params
        )
        
        return BaseResponse(
            success=True,
            message="Оценка выполнена успешно",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Ошибка при пакетной оценке: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/single", response_model=BaseResponse)
async def evaluate_single(
    question: str = Form(...),
    expected_answer: str = Form(...),
    index_name: str = Form(...),
    semantic_weight: float = Form(0.7),
    keyword_weight: float = Form(0.3),
    k: int = Form(5), 
    size: int = Form(10),
    use_hyde: bool = Form(False),
    hyde_num_hypotheses: int = Form(1),
    reranking: bool = Form(False)
):
    """
    Оценивает качество ответа на один вопрос.
    
    Args:
        question: Вопрос
        expected_answer: Ожидаемый ответ
        index_name: Название индекса
        semantic_weight: Вес семантического поиска
        keyword_weight: Вес ключевого поиска
        k: Количество ближайших векторов
        size: Размер результатов
        use_hyde: Использовать HyDE
        hyde_num_hypotheses: Количество гипотез для HyDE
        reranking: Использовать ColBERT реранжирование
        
    Returns:
        Результат оценки для одного вопроса
    """
    try:
        test_data = [{
            'question': question,
            'expected_answer': expected_answer
        }]
        
        search_params = {
            'semantic_weight': semantic_weight,
            'keyword_weight': keyword_weight,
            'k': k,
            'size': size,
            'use_hyde': use_hyde,
            'hyde_num_hypotheses': hyde_num_hypotheses,
            'reranking': reranking
        }
        
        result = await rag_evaluation_service.evaluate_rag_performance(
            test_data=test_data,
            index_name=index_name,
            search_params=search_params
        )
        
        return BaseResponse(
            success=True,
            message="Оценка выполнена успешно",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Ошибка при оценке одного вопроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=BaseResponse)
async def get_test_statistics(
    index_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Получает статистику по тестированию RAG из PostgreSQL.
    
    Args:
        index_name: Фильтр по имени индекса (опционально)
        start_date: Начальная дата в формате YYYY-MM-DD (опционально)
        end_date: Конечная дата в формате YYYY-MM-DD (опционально)
        
    Returns:
        Статистика тестирования
    """
    try:
        from datetime import datetime
        
        # Конвертируем строки дат в datetime объекты
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        stats = postgres_storage.get_test_statistics(
            index_name=index_name,
            start_date=start_dt,
            end_date=end_dt
        )
        
        return BaseResponse(
            success=True,
            message="Статистика получена успешно",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=BaseResponse)
async def get_recent_sessions(limit: int = 10):
    """
    Получает список последних сессий тестирования.
    
    Args:
        limit: Количество сессий для возврата (по умолчанию 10)
        
    Returns:
        Список последних сессий тестирования
    """
    try:
        sessions = postgres_storage.get_recent_sessions(limit=limit)
        
        return BaseResponse(
            success=True,
            message=f"Получено {len(sessions)} последних сессий",
            data=sessions
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении сессий: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}", response_model=BaseResponse)
async def get_session_results(session_id: str):
    """
    Получает все результаты для конкретной сессии тестирования.
    
    Args:
        session_id: ID сессии тестирования
        
    Returns:
        Результаты тестирования для указанной сессии
    """
    try:
        results = postgres_storage.get_test_session_results(session_id)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Сессия с ID {session_id} не найдена"
            )
        
        return BaseResponse(
            success=True,
            message=f"Получено {len(results)} результатов для сессии {session_id}",
            data=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении результатов сессии: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=BaseResponse)
async def check_postgres_health():
    """
    Проверяет состояние подключения к PostgreSQL.
    
    Returns:
        Статус подключения к PostgreSQL
    """
    try:
        is_connected = postgres_storage.test_connection()
        
        if is_connected:
            return BaseResponse(
                success=True,
                message="PostgreSQL подключение работает",
                data={"status": "connected", "postgres_host": postgres_storage.connection_params['host']}
            )
        else:
            return BaseResponse(
                success=False,
                message="Не удается подключиться к PostgreSQL",
                data={"status": "disconnected"}
            )
            
    except Exception as e:
        logger.error(f"Ошибка при проверке PostgreSQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== API endpoints для ручной разметки ответов =====

@router.get("/unrated", response_model=BaseResponse)
async def get_unrated_tests(limit: int = 50):
    """
    Получает список неоцененных тестов для ручной разметки.
    
    Args:
        limit: Максимальное количество тестов для возврата (по умолчанию 50)
        
    Returns:
        Список неоцененных тестов
    """
    try:
        unrated_tests = postgres_storage.get_unrated_tests(limit=limit)
        
        return BaseResponse(
            success=True,
            message=f"Получено {len(unrated_tests)} неоцененных тестов",
            data=unrated_tests
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении неоцененных тестов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rate", response_model=BaseResponse)
async def rate_test_result(
    test_id: int = Form(...),
    is_correct: bool = Form(...),
    comment: Optional[str] = Form(None),
    rated_by: str = Form("api_user")
):
    """
    Сохраняет пользовательскую оценку результата теста.
    
    Args:
        test_id: ID теста в базе данных
        is_correct: True если ответ правильный, False если неправильный
        comment: Комментарий к оценке (необязательно)
        rated_by: Кто поставил оценку (по умолчанию "api_user")
        
    Returns:
        Результат сохранения оценки
    """
    try:
        success = postgres_storage.save_user_rating(
            test_id=test_id,
            is_correct=is_correct,
            comment=comment,
            rated_by=rated_by
        )
        
        if success:
            rating_text = "ПРАВИЛЬНО" if is_correct else "НЕПРАВИЛЬНО"
            return BaseResponse(
                success=True,
                message=f"Оценка сохранена: {rating_text}",
                data={
                    "test_id": test_id,
                    "rating": is_correct,
                    "comment": comment,
                    "rated_by": rated_by
                }
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Тест с ID {test_id} не найден"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при сохранении оценки: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/ratings", response_model=BaseResponse)
async def get_rating_statistics(
    index_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Получает статистику по пользовательским оценкам.
    
    Args:
        index_name: Фильтр по имени индекса (опционально)
        start_date: Начальная дата в формате YYYY-MM-DD (опционально)
        end_date: Конечная дата в формате YYYY-MM-DD (опционально)
        
    Returns:
        Статистика пользовательских оценок
    """
    try:
        from datetime import datetime
        
        # Конвертируем строки дат в datetime объекты
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        stats = postgres_storage.get_test_statistics_with_ratings(
            index_name=index_name,
            start_date=start_dt,
            end_date=end_dt
        )
        
        return BaseResponse(
            success=True,
            message="Статистика оценок получена успешно",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении статистики оценок: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/{test_id}", response_model=BaseResponse)
async def get_test_details(test_id: int):
    """
    Получает детали конкретного теста для разметки.
    
    Args:
        test_id: ID теста в базе данных
        
    Returns:
        Детальная информация о тесте
    """
    try:
        # Получаем детали теста (используем метод для получения сессии, но с фильтром по ID)
        test_details = postgres_storage.get_connection()
        
        with test_details as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        id, test_session_id, test_timestamp, question, expected_answer, actual_answer,
                        search_results, search_params, metrics, index_name, technologies,
                        search_time_ms, generation_time_ms, total_time_ms, error_message, 
                        user_rating, user_comment, rated_by, rated_at, created_at
                    FROM rag_test_results 
                    WHERE id = %s;
                """, (test_id,))
                
                result = cursor.fetchone()
                
                if not result:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Тест с ID {test_id} не найден"
                    )
                
                # Преобразуем в словарь
                columns = [desc[0] for desc in cursor.description]
                test_dict = dict(zip(columns, result))
        
        return BaseResponse(
            success=True,
            message=f"Детали теста {test_id} получены успешно",
            data=test_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении деталей теста: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== API endpoints для аналитики пользовательских запросов =====

@router.get("/user-queries/statistics", response_model=BaseResponse)
async def get_user_queries_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Получает статистику по пользовательским запросам.
    
    Args:
        start_date: Начальная дата в формате YYYY-MM-DD (опционально)
        end_date: Конечная дата в формате YYYY-MM-DD (опционально)
        user_id: ID пользователя для фильтрации (опционально)
        
    Returns:
        Статистика пользовательских запросов
    """
    try:
        from datetime import datetime
        
        # Конвертируем строки дат в datetime объекты
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        stats = postgres_storage.get_user_queries_statistics(
            start_date=start_dt,
            end_date=end_dt,
            user_id=user_id
        )
        
        return BaseResponse(
            success=True,
            message="Статистика запросов получена успешно",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении статистики запросов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-queries/popular", response_model=BaseResponse)
async def get_popular_queries(limit: int = 10):
    """
    Получает список популярных запросов.
    
    Args:
        limit: Максимальное количество запросов (по умолчанию 10)
        
    Returns:
        Список популярных запросов
    """
    try:
        popular_queries = postgres_storage.get_popular_queries(limit=limit)
        
        return BaseResponse(
            success=True,
            message=f"Получено {len(popular_queries)} популярных запросов",
            data=popular_queries
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении популярных запросов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-queries/recent", response_model=BaseResponse)
async def get_recent_queries(
    limit: int = 50,
    user_id: Optional[str] = None
):
    """
    Получает последние запросы.
    
    Args:
        limit: Максимальное количество запросов (по умолчанию 50)
        user_id: ID пользователя для фильтрации (опционально)
        
    Returns:
        Список последних запросов
    """
    try:
        recent_queries = postgres_storage.get_recent_queries(
            limit=limit,
            user_id=user_id
        )
        
        return BaseResponse(
            success=True,
            message=f"Получено {len(recent_queries)} последних запросов",
            data=recent_queries
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении последних запросов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user-queries/create-table", response_model=BaseResponse)
async def create_user_queries_table():
    """
    Создает таблицу для пользовательских запросов.
    
    Returns:
        Результат создания таблицы
    """
    try:
        postgres_storage.create_user_queries_table()
        
        return BaseResponse(
            success=True,
            message="Таблица user_queries создана/обновлена успешно",
            data={"table": "user_queries", "status": "created"}
        )
        
    except Exception as e:
        logger.error(f"Ошибка при создании таблицы user_queries: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 