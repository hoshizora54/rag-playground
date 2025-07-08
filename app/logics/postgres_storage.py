"""
Модуль для работы с PostgreSQL для сохранения результатов тестирования RAG.
"""

import psycopg2
import psycopg2.extras
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import uuid
from app.core.config import config
from app.core.logger import logger


class PostgreSQLStorage:
    """Класс для работы с PostgreSQL для сохранения результатов тестирования RAG."""
    
    def __init__(self):
        """Инициализация подключения к PostgreSQL."""
        self.connection_params = {
            'host': config.POSTGRES_HOST,
            'port': config.POSTGRES_PORT,
            'database': config.POSTGRES_DB,
            'user': config.POSTGRES_USER,
            'password': config.POSTGRES_PASSWORD
        }
        logger.info(f"Подключение к PostgreSQL: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
    
    def get_connection(self):
        """Получает подключение к PostgreSQL."""
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            logger.error(f"Ошибка подключения к PostgreSQL: {e}")
            raise
    
    def migrate_table_for_ratings(self):
        """Добавляет поля для ручной разметки в существующую таблицу."""
        migration_sql = """
        -- Добавляем новые поля для ручной разметки, если они еще не существуют
        DO $$ 
        BEGIN
            -- Проверяем и добавляем поле user_rating
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'rag_test_results' AND column_name = 'user_rating'
            ) THEN
                ALTER TABLE rag_test_results 
                ADD COLUMN user_rating BOOLEAN DEFAULT NULL;
            END IF;
            
            -- Проверяем и добавляем поле user_comment
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'rag_test_results' AND column_name = 'user_comment'
            ) THEN
                ALTER TABLE rag_test_results 
                ADD COLUMN user_comment TEXT DEFAULT NULL;
            END IF;
            
            -- Проверяем и добавляем поле rated_by
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'rag_test_results' AND column_name = 'rated_by'
            ) THEN
                ALTER TABLE rag_test_results 
                ADD COLUMN rated_by VARCHAR(255) DEFAULT NULL;
            END IF;
            
            -- Проверяем и добавляем поле rated_at
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'rag_test_results' AND column_name = 'rated_at'
            ) THEN
                ALTER TABLE rag_test_results 
                ADD COLUMN rated_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;
            END IF;
        END $$;
        
        -- Создаем индексы для новых полей, если они еще не существуют
        CREATE INDEX IF NOT EXISTS idx_rag_test_user_rating ON rag_test_results(user_rating);
        CREATE INDEX IF NOT EXISTS idx_rag_test_rated_by ON rag_test_results(rated_by);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(migration_sql)
                    conn.commit()
                logger.info("Миграция таблицы для поддержки ручной разметки выполнена успешно")
        except Exception as e:
            logger.error(f"Ошибка при миграции таблицы: {e}")
            raise

    def create_table(self):
        """Создает таблицу для сохранения результатов тестирования RAG."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS rag_test_results (
            id SERIAL PRIMARY KEY,
            test_session_id VARCHAR(255) NOT NULL,
            test_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL,
            expected_answer TEXT NOT NULL,
            actual_answer TEXT,
            search_results JSONB,
            search_params JSONB,
            metrics JSONB,
            index_name VARCHAR(255),
            technologies JSONB,
            search_time_ms INTEGER,
            generation_time_ms INTEGER,
            total_time_ms INTEGER,
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Создаем индексы для основных полей
        CREATE INDEX IF NOT EXISTS idx_rag_test_session_id ON rag_test_results(test_session_id);
        CREATE INDEX IF NOT EXISTS idx_rag_test_timestamp ON rag_test_results(test_timestamp);
        CREATE INDEX IF NOT EXISTS idx_rag_test_index_name ON rag_test_results(index_name);
        CREATE INDEX IF NOT EXISTS idx_rag_test_created_at ON rag_test_results(created_at);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    conn.commit()
                logger.info("Таблица rag_test_results создана/обновлена успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании таблицы: {e}")
            raise
    
    def create_user_queries_table(self):
        """Создает таблицу для сохранения пользовательских запросов."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS user_queries (
            id SERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            user_id VARCHAR(255) DEFAULT 'anonymous',
            session_id VARCHAR(255),
            index_name VARCHAR(255),
            search_params JSONB,
            search_results JSONB,
            response_text TEXT,
            response_images JSONB,
            search_time_ms INTEGER,
            generation_time_ms INTEGER,
            total_time_ms INTEGER,
            technologies_used JSONB,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            ip_address INET,
            user_agent TEXT
        );
        
        -- Создаем индексы
        CREATE INDEX IF NOT EXISTS idx_user_queries_timestamp ON user_queries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_user_queries_user_id ON user_queries(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_queries_session_id ON user_queries(session_id);
        CREATE INDEX IF NOT EXISTS idx_user_queries_index_name ON user_queries(index_name);
        CREATE INDEX IF NOT EXISTS idx_user_queries_success ON user_queries(success);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    conn.commit()
                logger.info("Таблица user_queries создана/обновлена успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании таблицы user_queries: {e}")
            raise
    
    def save_test_result(self, 
                        test_session_id: str,
                        question: str,
                        expected_answer: str,
                        actual_answer: str,
                        search_results: List[Dict[str, Any]],
                        search_params: Dict[str, Any],
                        metrics: Dict[str, Any],
                        index_name: str,
                        technologies: Dict[str, Any],
                        search_time_ms: Optional[int] = None,
                        generation_time_ms: Optional[int] = None,
                        total_time_ms: Optional[int] = None,
                        error_message: Optional[str] = None) -> int:
        """
        Сохраняет результат одного теста в PostgreSQL.
        
        Args:
            test_session_id: ID сессии тестирования
            question: Вопрос
            expected_answer: Ожидаемый ответ
            actual_answer: Фактический ответ
            search_results: Результаты поиска
            search_params: Параметры поиска
            metrics: Метрики качества
            index_name: Имя индекса
            technologies: Использованные технологии
            search_time_ms: Время поиска в мс
            generation_time_ms: Время генерации ответа в мс
            total_time_ms: Общее время в мс
            error_message: Сообщение об ошибке если есть
            
        Returns:
            ID созданной записи
        """
        insert_sql = """
        INSERT INTO rag_test_results (
            test_session_id, question, expected_answer, actual_answer,
            search_results, search_params, metrics, index_name, technologies,
            search_time_ms, generation_time_ms, total_time_ms, error_message
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (
                        test_session_id,
                        question,
                        expected_answer,
                        actual_answer,
                        json.dumps(search_results, ensure_ascii=False),
                        json.dumps(search_params, ensure_ascii=False),
                        json.dumps(metrics, ensure_ascii=False),
                        index_name,
                        json.dumps(technologies, ensure_ascii=False),
                        search_time_ms,
                        generation_time_ms,
                        total_time_ms,
                        error_message
                    ))
                    record_id = cursor.fetchone()[0]
                    conn.commit()
                logger.debug(f"Результат теста сохранен с ID: {record_id}")
                return record_id
        except Exception as e:
            logger.error(f"Ошибка при сохранении результата теста: {e}")
            raise
    
    def save_user_rating(self, 
                        test_id: int, 
                        is_correct: bool, 
                        comment: Optional[str] = None, 
                        rated_by: str = "user") -> bool:
        """
        Сохраняет пользовательскую оценку результата теста.
        
        Args:
            test_id: ID записи в таблице
            is_correct: True если ответ правильный, False если неправильный
            comment: Комментарий пользователя (необязательно)
            rated_by: Кто поставил оценку (по умолчанию "user")
            
        Returns:
            True если сохранение прошло успешно
        """
        update_sql = """
        UPDATE rag_test_results 
        SET 
            user_rating = %s,
            user_comment = %s,
            rated_by = %s,
            rated_at = CURRENT_TIMESTAMP
        WHERE id = %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(update_sql, (is_correct, comment, rated_by, test_id))
                    rows_affected = cursor.rowcount
                    conn.commit()
                
                if rows_affected > 0:
                    logger.debug(f"Пользовательская оценка сохранена для теста ID: {test_id}")
                    return True
                else:
                    logger.warning(f"Тест с ID {test_id} не найден")
                    return False
                    
        except Exception as e:
            logger.error(f"Ошибка при сохранении пользовательской оценки: {e}")
            raise
    
    def get_test_session_results(self, test_session_id: str) -> List[Dict[str, Any]]:
        """Получает все результаты для определенной сессии тестирования."""
        select_sql = """
        SELECT 
            id, test_session_id, test_timestamp, question, expected_answer, actual_answer,
            search_results, search_params, metrics, index_name, technologies,
            search_time_ms, generation_time_ms, total_time_ms, error_message, 
            user_rating, user_comment, rated_by, rated_at, created_at
        FROM rag_test_results 
        WHERE test_session_id = %s 
        ORDER BY created_at ASC;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(select_sql, (test_session_id,))
                    results = cursor.fetchall()
                    
                    # Конвертируем в обычные словари
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Ошибка при получении результатов сессии: {e}")
            raise
    
    def get_unrated_tests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Получает неоцененные тесты для ручной разметки.
        
        Args:
            limit: Максимальное количество записей
            
        Returns:
            Список неоцененных тестов
        """
        select_sql = """
        SELECT 
            id, test_session_id, test_timestamp, question, expected_answer, actual_answer,
            search_results, search_params, metrics, index_name, technologies,
            search_time_ms, generation_time_ms, total_time_ms, error_message, created_at
        FROM rag_test_results 
        WHERE user_rating IS NULL 
        ORDER BY created_at DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(select_sql, (limit,))
                    results = cursor.fetchall()
                    
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Ошибка при получении неоцененных тестов: {e}")
            raise
    
    def get_test_statistics_with_ratings(self, 
                                        index_name: Optional[str] = None,
                                        start_date: Optional[datetime] = None,
                                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Получает статистику по тестированию с учетом пользовательских оценок."""
        conditions = []
        params = []
        
        if index_name:
            conditions.append("index_name = %s")
            params.append(index_name)
        
        if start_date:
            conditions.append("test_timestamp >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("test_timestamp <= %s")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        stats_sql = f"""
        SELECT 
            COUNT(*) as total_tests,
            COUNT(DISTINCT test_session_id) as total_sessions,
            COUNT(DISTINCT index_name) as total_indices,
            AVG((metrics->>'accuracy')::float) as avg_accuracy,
            AVG((metrics->>'score')::float) as avg_score,
            AVG(search_time_ms) as avg_search_time_ms,
            AVG(generation_time_ms) as avg_generation_time_ms,
            AVG(total_time_ms) as avg_total_time_ms,
            COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as errors_count,
            -- Статистика по пользовательским оценкам
            COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END) as rated_tests,
            COUNT(CASE WHEN user_rating = true THEN 1 END) as user_correct_count,
            COUNT(CASE WHEN user_rating = false THEN 1 END) as user_incorrect_count,
            COUNT(CASE WHEN user_rating IS NULL THEN 1 END) as unrated_count,
            CASE 
                WHEN COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END) > 0 
                THEN COUNT(CASE WHEN user_rating = true THEN 1 END)::float / COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END)::float
                ELSE NULL 
            END as user_accuracy
        FROM rag_test_results 
        {where_clause};
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(stats_sql, params)
                    stats = dict(cursor.fetchone())
                    return stats
        except Exception as e:
            logger.error(f"Ошибка при получении статистики с оценками: {e}")
            raise

    def get_test_statistics(self, 
                           index_name: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Получает статистику по тестированию."""
        return self.get_test_statistics_with_ratings(index_name, start_date, end_date)

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получает последние сессии тестирования."""
        recent_sql = """
        SELECT 
            test_session_id,
            MIN(test_timestamp) as start_time,
            MAX(test_timestamp) as end_time,
            COUNT(*) as tests_count,
            index_name,
            AVG((metrics->>'accuracy')::float) as avg_accuracy,
            COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as errors_count
        FROM rag_test_results 
        GROUP BY test_session_id, index_name
        ORDER BY MIN(test_timestamp) DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(recent_sql, (limit,))
                    sessions = cursor.fetchall()
                    return [dict(row) for row in sessions]
        except Exception as e:
            logger.error(f"Ошибка при получении последних сессий: {e}")
            raise
    
    def generate_session_id(self) -> str:
        """Генерирует уникальный ID для сессии тестирования."""
        return str(uuid.uuid4())
    
    def test_connection(self) -> bool:
        """Тестирует подключение к PostgreSQL."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Ошибка тестирования подключения: {e}")
            return False

    def save_user_query(self,
                        query_text: str,
                        user_id: str = "anonymous",
                        session_id: Optional[str] = None,
                        index_name: Optional[str] = None,
                        search_params: Optional[Dict[str, Any]] = None,
                        search_results: Optional[List[Dict[str, Any]]] = None,
                        response_text: Optional[str] = None,
                        response_images: Optional[List[Dict[str, Any]]] = None,
                        search_time_ms: Optional[int] = None,
                        generation_time_ms: Optional[int] = None,
                        total_time_ms: Optional[int] = None,
                        technologies_used: Optional[Dict[str, Any]] = None,
                        success: bool = True,
                        error_message: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None) -> int:
        """
        Сохраняет пользовательский запрос в базу данных.
        
        Args:
            query_text: Текст запроса
            user_id: ID пользователя (по умолчанию 'anonymous')
            session_id: ID сессии
            index_name: Имя индекса
            search_params: Параметры поиска
            search_results: Результаты поиска
            response_text: Текст ответа
            response_images: Изображения в ответе
            search_time_ms: Время поиска в мс
            generation_time_ms: Время генерации ответа в мс
            total_time_ms: Общее время в мс
            technologies_used: Использованные технологии
            success: Успешность запроса
            error_message: Сообщение об ошибке если есть
            ip_address: IP адрес пользователя
            user_agent: User Agent
            
        Returns:
            ID созданной записи
        """
        insert_sql = """
        INSERT INTO user_queries (
            query_text, user_id, session_id, index_name, search_params,
            search_results, response_text, response_images, search_time_ms,
            generation_time_ms, total_time_ms, technologies_used, success,
            error_message, ip_address, user_agent
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, (
                        query_text,
                        user_id,
                        session_id,
                        index_name,
                        json.dumps(search_params, ensure_ascii=False) if search_params else None,
                        json.dumps(search_results, ensure_ascii=False) if search_results else None,
                        response_text,
                        json.dumps(response_images, ensure_ascii=False) if response_images else None,
                        search_time_ms,
                        generation_time_ms,
                        total_time_ms,
                        json.dumps(technologies_used, ensure_ascii=False) if technologies_used else None,
                        success,
                        error_message,
                        ip_address,
                        user_agent
                    ))
                    record_id = cursor.fetchone()[0]
                    conn.commit()
                logger.debug(f"Пользовательский запрос сохранен с ID: {record_id}")
                return record_id
        except Exception as e:
            logger.error(f"Ошибка при сохранении пользовательского запроса: {e}")
            raise
    
    def get_user_queries_statistics(self, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Получает статистику по пользовательским запросам.
        
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            user_id: ID пользователя для фильтрации
            
        Returns:
            Статистика запросов
        """
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= %s")
            params.append(end_date)
        
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        stats_sql = f"""
        SELECT 
            COUNT(*) as total_queries,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT session_id) as unique_sessions,
            COUNT(DISTINCT index_name) as indices_used,
            COUNT(CASE WHEN success = true THEN 1 END) as successful_queries,
            COUNT(CASE WHEN success = false THEN 1 END) as failed_queries,
            AVG(search_time_ms) as avg_search_time_ms,
            AVG(generation_time_ms) as avg_generation_time_ms,
            AVG(total_time_ms) as avg_total_time_ms,
            MIN(timestamp) as first_query,
            MAX(timestamp) as last_query
        FROM user_queries 
        {where_clause};
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(stats_sql, params)
                    stats = dict(cursor.fetchone())
                    return stats
        except Exception as e:
            logger.error(f"Ошибка при получении статистики запросов: {e}")
            raise
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получает список популярных запросов.
        
        Args:
            limit: Максимальное количество запросов
            
        Returns:
            Список популярных запросов
        """
        popular_sql = """
        SELECT 
            query_text,
            COUNT(*) as query_count,
            COUNT(DISTINCT user_id) as unique_users,
            AVG(total_time_ms) as avg_time_ms,
            MAX(timestamp) as last_used
        FROM user_queries 
        WHERE success = true
        GROUP BY query_text
        ORDER BY query_count DESC, last_used DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(popular_sql, (limit,))
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Ошибка при получении популярных запросов: {e}")
            raise
    
    def get_recent_queries(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получает последние запросы.
        
        Args:
            limit: Максимальное количество запросов
            user_id: ID пользователя для фильтрации
            
        Returns:
            Список последних запросов
        """
        where_clause = "WHERE user_id = %s" if user_id else ""
        params = [user_id] if user_id else []
        params.append(limit)
        
        recent_sql = f"""
        SELECT 
            id, query_text, user_id, session_id, index_name, 
            response_text, search_time_ms, generation_time_ms, total_time_ms,
            technologies_used, success, error_message, timestamp
        FROM user_queries 
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(recent_sql, params)
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Ошибка при получении последних запросов: {e}")
            raise


# Глобальный экземпляр
postgres_storage = PostgreSQLStorage() 