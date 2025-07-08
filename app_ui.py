"""Streamlit интерфейс для RAG системы."""

import streamlit as st
import pandas as pd
import asyncio
from typing import List, Dict, Any
import io
import time
import json
import logging

# Импорты модулей приложения
from app.services.indexing_service import indexing_service
from app.services.answer_service import answer_service, rag_evaluation_service
from app.logics.opensearch import opensearch_worker
from app.logics.postgres_storage import PostgreSQLStorage

logger = logging.getLogger(__name__)


def run_async_fn(fn, *args):
    """Выполняет асинхронную функцию в синхронном контексте Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(fn(*args))
    finally:
        loop.close()


async def upload_document_async(file, index_name: str, use_semantic_chunking: bool = False, 
                                semantic_threshold: float = 0.8) -> str:
    """Асинхронная загрузка документа через indexing_service."""
    
    # Создаем wrapper для совместимости с FastAPI
    class StreamlitFile:
        def __init__(self, file_obj):
            self.file_obj = file_obj
            self.filename = file_obj.name
        
        async def read(self):
            return self.file_obj.read()
    
    try:
        wrapped_file = StreamlitFile(file)
        result = await indexing_service.upload_and_index_pdf(
            wrapped_file, 
            index_name,
            use_semantic_chunking=use_semantic_chunking,
            semantic_threshold=semantic_threshold
        )
        
        if result.get("success"):
            message = result.get("message", "Документ успешно загружен")
            chunks_count = result.get("data", {}).get("chunks_count", 0)
            return f"{message}. Создано {chunks_count} чанков."
        else:
            return f"Ошибка: {result.get('message', 'Неизвестная ошибка')}"
            
    except Exception as e:
        return f"Ошибка при загрузке: {str(e)}"


def get_indices() -> List[str]:
    """Получает список доступных индексов."""
    try:
        indices = opensearch_worker.list_indices()
        return [idx for idx in indices if not idx.startswith('.')]
    except Exception as e:
        st.error(f"Ошибка получения индексов: {e}")
        return []


async def search_and_answer_async(
    query_text: str,
    index_name: str,
    semantic_weight: float,
    keyword_weight: float,
    size: int,
    use_hyde: bool = False,
    hyde_num_hypotheses: int = 1,
    reranking: bool = False
) -> Dict[str, Any]:
    """Асинхронный поиск и генерация ответа."""
    
    from app.schemas.search import UserQuery
    
    user_query = UserQuery(
        query_text=query_text,
        index_name=index_name,
        sematic=semantic_weight,
        keyword=keyword_weight,
        size=size,
        use_hyde=use_hyde,
        hyde_num_hypotheses=hyde_num_hypotheses,
        reranking=reranking
    )
    
    # Получаем результат через answer_service
    result = await answer_service.generate_answer(user_query)
    return result


async def evaluate_rag_async(
    test_data: List[Dict[str, str]], 
    index_name: str,
    search_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Асинхронная оценка RAG."""
    
    result = await rag_evaluation_service.evaluate_rag_performance(
        test_data=test_data,
        index_name=index_name,
        search_params=search_params
    )
    return result


def create_streamlit_interface():
    """Создает интерфейс Streamlit."""
    
    st.set_page_config(
        page_title="RAG Система",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("RAG Система поиска и ответов")
    st.markdown("**Retrieval-Augmented Generation** система на базе OpenSearch и Yandex GPT")
    
    # Боковая панель для настроек
    with st.sidebar:
        st.header("Настройки")
        
        # Выбор режима работы
        mode = st.selectbox(
            "Режим работы:",
            ["Поиск и ответы", "Загрузка документов", "Тестирование RAG", "Разметка ответов", "Аналитика запросов"],
            help="Выберите нужный режим работы системы"
        )
        
        if mode == "Поиск и ответы":
            st.subheader("Параметры поиска")
            
            # Выбор индекса
            indices = get_indices()
            if indices:
                selected_index = st.selectbox("Выберите индекс:", indices)
            else:
                st.warning("Не найдено ни одного индекса")
                selected_index = None
            
            # Веса для поиска
            st.write("**Настройка весов поиска:**")
            semantic_weight = st.slider("Семантический поиск", 0.0, 1.0, 0.7, 0.1)
            keyword_weight = st.slider("Ключевой поиск", 0.0, 1.0, 0.3, 0.1)
            
            # Нормализация весов
            total = semantic_weight + keyword_weight
            if total > 0:
                semantic_weight = semantic_weight / total
                keyword_weight = keyword_weight / total
            else:
                semantic_weight, keyword_weight = 0.7, 0.3
            
            st.write(f"Нормализованные веса: Семантический: {semantic_weight:.2f}, Ключевой: {keyword_weight:.2f}")
            
            # Количество результатов
            size = st.slider("Количество результатов", 1, 20, 5)
            
            # Дополнительные технологии
            st.subheader("Дополнительные технологии")
            
            use_hyde = st.checkbox(
                "HyDE (Hypothetical Document Embeddings)",
                help="Улучшает качество поиска генерируя гипотетические документы"
            )
            
            hyde_num_hypotheses = 1
            if use_hyde:
                hyde_num_hypotheses = st.slider("Количество гипотез HyDE", 1, 5, 1)
            
            reranking = st.checkbox(
                "ColBERT Reranking",
                help="Переранжирование результатов поиска с помощью ColBERT модели"
            )
            
            # Настройки отображения
            st.subheader("Отображение")
            show_sources = st.checkbox("Показать источники", value=True)
            show_context = st.checkbox("Показать контекст", value=False)
    
    # Основной интерфейс
    if mode == "Поиск и ответы":
        st.header("Поиск и генерация ответов")
        
        # Поле ввода запроса
        user_query = st.text_area(
            "Введите ваш вопрос:",
            placeholder="Например: Что такое машинное обучение?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("Найти ответ", type="primary")
        with col2:
            if not selected_index:
                st.warning("Выберите индекс для поиска")
        
        # Обработка запроса
        if search_button and user_query.strip() and selected_index:
            with st.spinner("Ищу ответ..."):
                start_time = time.time()
                
                try:
                    # Выполняем поиск и генерацию ответа
                    result = run_async_fn(
                        search_and_answer_async,
                        user_query,
                        selected_index,
                        semantic_weight,
                        keyword_weight,
                        size,
                        use_hyde,
                        hyde_num_hypotheses,
                        reranking
                    )
                    
                    search_time = time.time() - start_time
                    
                    # Отображаем информацию о поиске
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Время поиска", f"{search_time:.2f} сек")
                    with col2:
                        sources_count = len(result.get("sources", []))
                        st.metric("Источников найдено", sources_count)
                    with col3:
                        images_count = len(result.get("images", []))
                        st.metric("Изображений", images_count)
                    
                    # Дополнительная информация
                    with st.expander("Информация о поиске", expanded=False):
                        search_info = result.get("search_info", {})
                        tech_info = search_info.get("technologies", {})
                        translation_info = search_info.get("translation", {})
                        
                        if translation_info:
                            if translation_info.get("translated"):
                                translated_query = translation_info.get("translated_query", "")
                                if translated_query:
                                    st.info("**Переведенный запрос:**")
                                    if isinstance(translated_query, list):
                                        for i, tq in enumerate(translated_query, 1):
                                            st.code(f"{i}. {tq}")
                                    else:
                                        st.code(translated_query)
                                else:
                                    query_lang = translation_info.get("query_language", "неизвестно")
                                    doc_lang = translation_info.get("document_language", "неизвестно")
                                    if query_lang != "unknown" and doc_lang != "unknown":
                                        st.success(f"**Языки совпадают**: {query_lang} и {doc_lang}")
                                
                                # Технологии RAG
                                tech_status = []
                                if tech_info.get("hyde"):
                                    tech_status.append("HyDE: Активен")
                                if tech_info.get("reranking"):
                                    tech_status.append("ColBERT Reranking: Активен")
                                
                                if tech_status:
                                    for status in tech_status:
                                        st.success(status)
                                else:
                                    st.info("Базовый поиск")
                        
                        # Отображаем ответ
                        st.subheader("Ответ ассистента")
                        answer = result.get("answer", "Не удалось получить ответ")
                        st.markdown(answer)
                        
                        # Отображаем источники
                        if show_sources:
                            sources = result.get("sources", [])
                            if sources:
                                st.subheader("Источники")
                                sources_df = pd.DataFrame(sources)
                                st.dataframe(sources_df, use_container_width=True)
                        
                        # Отображаем контекст
                        if show_context:
                            context = result.get("context_used", "")
                            if context:
                                st.subheader("Использованный контекст")
                                with st.expander("Показать контекст"):
                                    st.text(context)
                        
                        # Отображаем изображения
                        images = result.get("images", [])
                        if images:
                            st.subheader(f"Найденные изображения ({len(images)})")
                            
                            # Отображаем изображения в галерее
                            cols = st.columns(min(3, len(images)))
                            for i, img_info in enumerate(images):
                                with cols[i % len(cols)]:
                                    try:
                                        image_data = img_info.get('image_data')
                                        image_name = img_info.get('image_name', f'Изображение {i+1}')
                                        file_name = img_info.get('file_name', 'Неизвестный документ')
                                        
                                        if image_data:
                                            st.image(
                                                image_data, 
                                                caption=f"{image_name}\nИз: {file_name}",
                                                use_column_width=True
                                            )
                                        else:
                                            st.error(f"Не удалось загрузить изображение: {image_name}")
                                    except Exception as e:
                                        st.error(f"Ошибка загрузки изображения: {e}")
                
                except Exception as e:
                    st.error(f"Ошибка при поиске: {e}")
    
    elif mode == "Загрузка документов":
        st.header("Загрузка PDF документов")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Загрузка файла")
            
            # Загрузка файла
            uploaded_file = st.file_uploader(
                "Выберите PDF файл",
                type=['pdf'],
                help="Поддерживаются только PDF файлы"
            )
            
            # Название индекса
            index_name = st.text_input(
                "Название индекса",
                placeholder="Например: medical_docs, legal_docs",
                help="Введите уникальное название для индекса"
            )
            
            # Настройки разбиения на чанки
            st.subheader("Настройки разбиения на чанки")
            
            chunking_method = st.radio(
                "Метод разбиения:",
                options=["Обычное (по абзацам)", "Семантическое"],
                help="Обычное - быстрое разбиение по абзацам. Семантическое - умное группирование по смыслу."
            )
            
            use_semantic_chunking = chunking_method == "Семантическое"
            
            semantic_threshold = 0.8
            if use_semantic_chunking:
                semantic_threshold = st.slider(
                    "Порог семантического сходства:",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    help="Выше = больше объединения предложений, ниже = больше отдельных чанков"
                )
                
                st.info(f"""
                **Семантическое разбиение:**
                - Анализирует смысл каждого предложения
                - Группирует похожие предложения в один чанк
                - Порог {semantic_threshold:.2f}: {'высокий (меньше чанков)' if semantic_threshold > 0.7 else 'низкий (больше чанков)'}
                """)
            else:
                st.info("""
                **Обычное разбиение:**
                - Быстрое разделение по абзацам
                - Каждый абзац = отдельный чанк
                - Рекомендуется для больших документов
                """)
            
            # Кнопка загрузки
            if st.button("Загрузить документ", type="primary"):
                if not uploaded_file:
                    st.error("Выберите файл для загрузки")
                elif not index_name:
                    st.error("Введите название индекса")
                else:
                    with st.spinner("Загружаю и индексирую документ..."):
                        # Загружаем документ
                        result = run_async_fn(
                            upload_document_async, 
                            uploaded_file, 
                            index_name,
                            use_semantic_chunking,
                            semantic_threshold
                        )
                        
                        if "успешно загружен" in result:
                            st.success(result)
                        else:
                            st.error(result)
        
        with col2:
            st.subheader("Доступные индексы")
            indices = get_indices()
            if indices:
                for idx in indices:
                    st.text(f"{idx}")
            else:
                st.info("Индексы не найдены")
    
    elif mode == "Тестирование RAG":
        st.header("Тестирование качества RAG системы")
        
        st.markdown("""
        **Онлайн тестирование RAG**
        
        Загрузите CSV файл с вопросами и эталонными ответами для оценки качества работы системы.
        Система автоматически сравнит ответы и предоставит подробный анализ.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Загрузка тестовых данных")
            
            # Загрузка CSV файла
            test_file = st.file_uploader(
                "CSV файл с тестовыми данными",
                type=['csv'],
                help="CSV файл должен содержать колонки: 'question' (или 'вопрос') и 'expected_answer' (или 'ответ')"
            )
            
            if test_file:
                try:
                    # Сначала проверяем, что файл не пустой
                    if test_file.size == 0:
                        st.error("Загруженный файл пустой. Пожалуйста, загрузите корректный CSV файл.")
                    else:
                        # Сбрасываем курсор в начало файла
                        test_file.seek(0)
                        
                        # Предварительный просмотр данных
                        df = pd.read_csv(test_file, encoding='utf-8')
                        
                        # Проверяем, что датафрейм не пустой
                        if df.empty:
                            st.error("CSV файл не содержит данных. Пожалуйста, проверьте содержимое файла.")
                        else:
                            st.write("**Предварительный просмотр данных:**")
                            st.dataframe(df.head(), use_container_width=True)
                            st.write(f"Всего строк: {len(df)}")
                            
                            # Проверка колонок
                            columns = list(df.columns)
                            st.write(f"**Найденные колонки:** {columns}")
                            
                            # Поиск подходящих колонок
                            question_cols = [col for col in columns if any(word in col.lower() for word in ['вопрос', 'question', 'запрос'])]
                            answer_cols = [col for col in columns if any(word in col.lower() for word in ['ответ', 'answer', 'expected'])]
                            
                            if question_cols and answer_cols:
                                st.success(f"Найдены подходящие колонки: {question_cols[0]} и {answer_cols[0]}")
                            else:
                                st.error("Не найдены необходимые колонки. Убедитесь, что файл содержит колонки с вопросами и ответами.")
                        
                except pd.errors.EmptyDataError:
                    st.error("CSV файл пустой или содержит только заголовки. Пожалуйста, добавьте данные в файл.")
                except pd.errors.ParserError as e:
                    st.error(f"Ошибка парсинга CSV файла: {e}. Проверьте формат файла.")
                except UnicodeDecodeError:
                    st.error("Ошибка кодировки файла. Попробуйте сохранить файл в UTF-8.")
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {e}")
        
        with col2:
            st.subheader("Настройки тестирования")
            
            # Выбор индекса для тестирования
            test_indices = get_indices()
            if test_indices:
                test_index = st.selectbox("Индекс для тестирования:", test_indices)
            else:
                st.warning("Не найдено ни одного индекса")
                test_index = None
            
            # Настройки поиска для тестирования
            st.write("**Параметры поиска:**")
            test_semantic = st.slider("Семантический вес", 0.0, 1.0, 0.7, 0.1, key="test_semantic")
            test_keyword = st.slider("Ключевой вес", 0.0, 1.0, 0.3, 0.1, key="test_keyword")
            test_size = st.slider("Размер результатов", 1, 20, 5, key="test_size")
            
            # Дополнительные технологии для тестирования
            test_hyde = st.checkbox("Использовать HyDE", key="test_hyde")
            test_reranking = st.checkbox("Использовать ColBERT", key="test_reranking")
            
            # Кнопка запуска тестирования
            if st.button("Запустить тестирование", type="primary", disabled=not (test_file and test_index)):
                # Нормализация весов
                total = test_semantic + test_keyword
                if total > 0:
                    test_semantic = test_semantic / total
                    test_keyword = test_keyword / total
                else:
                    test_semantic, test_keyword = 0.7, 0.3
                
                # Подготавливаем тестовые данные
                try:
                    # Сбрасываем курсор в начало файла перед повторным чтением
                    test_file.seek(0)
                    df = pd.read_csv(test_file, encoding='utf-8')
                    
                    # Проверяем, что датафрейм не пустой
                    if df.empty:
                        st.error("CSV файл не содержит данных для тестирования")
                    else:
                        # Определяем колонки
                        question_cols = [col for col in df.columns if any(word in col.lower() for word in ['вопрос', 'question', 'запрос'])]
                        answer_cols = [col for col in df.columns if any(word in col.lower() for word in ['ответ', 'answer', 'expected'])]
                        
                        if not question_cols or not answer_cols:
                            st.error("Не найдены подходящие колонки в файле")
                        else:
                            # Формируем тестовые данные
                            test_data = []
                            for index, row in df.iterrows():
                                question = str(row[question_cols[0]]).strip()
                                expected_answer = str(row[answer_cols[0]]).strip()
                                
                                # Пропускаем строки с пустыми данными
                                if question and question != 'nan' and expected_answer and expected_answer != 'nan':
                                    test_data.append({
                                        'question': question,
                                        'expected_answer': expected_answer
                                    })
                            
                            if not test_data:
                                st.error("В файле нет валидных пар вопрос-ответ. Проверьте, что ячейки заполнены.")
                            else:
                                st.info(f"Подготовлено {len(test_data)} тестовых пар для обработки")
                                
                                # Параметры поиска
                                search_params = {
                                    'semantic_weight': test_semantic,
                                    'keyword_weight': test_keyword,
                                    'size': test_size,
                                    'use_hyde': test_hyde,
                                    'reranking': test_reranking
                                }
                                
                                # Запускаем тестирование
                                with st.spinner(f"Тестирую {len(test_data)} вопросов..."):
                                    eval_result = run_async_fn(
                                        evaluate_rag_async,
                                        test_data,
                                        test_index,
                                        search_params
                                    )
                                
                                # Отображаем результаты
                                st.success("Тестирование завершено!")
                                
                                # Основные метрики
                                st.subheader("Результаты тестирования")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Всего тестов", eval_result.get("total_tests", 0))
                                with col2:
                                    st.metric("Правильных ответов", eval_result.get("correct_answers", 0))
                                with col3:
                                    accuracy = eval_result.get("accuracy", 0)
                                    st.metric("Точность", f"{accuracy:.3f}")
                                with col4:
                                    accuracy_pct = eval_result.get("accuracy_percentage", 0)
                                    st.metric("Точность %", f"{accuracy_pct:.1f}%")
                                
                                # Прогресс-бар для точности
                                st.progress(accuracy)
                                
                                # Детальные результаты
                                if "test_results" in eval_result:
                                    st.subheader("Детальные результаты")
                                    
                                    # Создаем датафрейм для отображения
                                    results_data = []
                                    for test in eval_result["test_results"]:
                                        # Определяем статус ответа
                                        is_correct = test.get('score', 0) == 1
                                        status_icon = "ВЕРНО" if is_correct else "НЕВЕРНО"
                                        
                                        results_data.append({
                                            'Тест №': test.get('test_id', 0),
                                            'Статус': status_icon,
                                            'Оценка': 'ВЕРНО' if test['score'] == 1 else 'НЕВЕРНО',
                                            'Вопрос': test.get('question', '')[:100] + '...' if len(test.get('question', '')) > 100 else test.get('question', ''),
                                            'Время (мс)': test.get('total_time_ms', 0),
                                            'Источники': test.get('sources_count', 0)
                                        })
                                    
                                    results_df = pd.DataFrame(results_data)
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Детальный просмотр отдельных тестов
                                    st.subheader("Детальный анализ")
                                    
                                    # Выбор теста для детального просмотра
                                    test_options = [f"Тест {test['test_id']}: {test['question'][:50]}..." 
                                                  for test in eval_result["test_results"]]
                                    
                                    selected_test_idx = st.selectbox(
                                        "Выберите тест для детального просмотра:",
                                        range(len(test_options)),
                                        format_func=lambda x: test_options[x]
                                    )
                                    
                                    if selected_test_idx is not None:
                                        selected_test = eval_result["test_results"][selected_test_idx]
                                        
                                        # Отображение деталей теста
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            st.write("**Вопрос:**")
                                            st.write(selected_test.get('question', ''))
                                            
                                            st.write("**Ожидаемый ответ:**")
                                            st.write(selected_test.get('expected_answer', ''))
                                        
                                        with col2:
                                            st.write("**Фактический ответ:**")
                                            st.write(selected_test.get('rag_answer', ''))
                                            
                                            # Статус с цветовой индикацией
                                            is_correct = selected_test.get('score', 0) == 1
                                            if is_correct:
                                                st.success("**Оценка:** ПРАВИЛЬНО")
                                            else:
                                                st.error("**Оценка:** НЕПРАВИЛЬНО")
                                        
                                        # Объяснение оценки
                                        explanation = selected_test.get('explanation', '')
                                        if explanation:
                                            st.write("**Объяснение оценки:**")
                                            st.info(explanation)
                                        
                                        # Метрики теста
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Время поиска", f"{selected_test.get('search_time_ms', 0)} мс")
                                        with col2:
                                            st.metric("Время генерации", f"{selected_test.get('generation_time_ms', 0)} мс")
                                        with col3:
                                            st.metric("Общее время", f"{selected_test.get('total_time_ms', 0)} мс")
                            
                                # Общий анализ
                                if "overall_analysis" in eval_result:
                                    st.subheader("Общий анализ")
                                    st.write(eval_result["overall_analysis"])
                                
                                # Сохранение результатов
                                st.subheader("Сохранение результатов")
                                session_id = eval_result.get("test_session_id", "unknown")
                                st.info(f"Результаты сохранены в PostgreSQL. ID сессии: {session_id}")
                        
                except pd.errors.EmptyDataError:
                    st.error("CSV файл пустой или содержит только заголовки. Пожалуйста, добавьте данные в файл.")
                except pd.errors.ParserError as e:
                    st.error(f"Ошибка парсинга CSV файла: {e}. Проверьте формат файла.")
                except UnicodeDecodeError:
                    st.error("Ошибка кодировки файла. Попробуйте сохранить файл в UTF-8.")
                except Exception as e:
                    st.error(f"Ошибка при тестировании: {e}")
    
    elif mode == "Разметка ответов":
        st.header("Ручная разметка ответов")
        
        st.markdown("""
        **Ручная разметка результатов тестирования**
        
        Здесь вы можете проверить и оценить ответы, сгенерированные системой RAG.
        Ваши оценки помогут улучшить качество системы.
        """)
        
        try:
            postgres_storage = PostgreSQLStorage()
            
            # Проверяем и выполняем миграцию если необходимо
            try:
                # Пытаемся получить неоцененные тесты - это покажет, нужна ли миграция
                postgres_storage.create_table()
                postgres_storage.migrate_table_for_ratings()
                logger.info("Миграция для ручной разметки проверена/выполнена")
            except Exception as migration_error:
                logger.warning(f"Ошибка при проверке миграции: {migration_error}")
            
            # Получаем неоцененные тесты
            unrated_tests = postgres_storage.get_unrated_tests(limit=100)
            
            if not unrated_tests:
                st.info("Все тесты уже оценены! Запустите новые тесты для оценки.")
            else:
                st.success(f"Найдено {len(unrated_tests)} неоцененных тестов")
                
                # Выбор теста для оценки
                test_options = [
                    f"Тест {test['id']}: {test['question'][:80]}..." 
                    if len(test['question']) > 80 
                    else f"Тест {test['id']}: {test['question']}"
                    for test in unrated_tests
                ]
                
                selected_test_idx = st.selectbox(
                    "Выберите тест для оценки:",
                    range(len(test_options)),
                    format_func=lambda x: test_options[x],
                    key="rating_test_selector"
                )
                
                if selected_test_idx is not None:
                    selected_test = unrated_tests[selected_test_idx]
                    
                    # Отображение информации о тесте
                    st.subheader(f"Тест #{selected_test['id']}")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Вопрос:**")
                        st.info(selected_test['question'])
                        
                        st.write("**Ожидаемый ответ:**")
                        st.success(selected_test['expected_answer'])
                        
                        # Дополнительная информация
                        st.write("**Дополнительная информация:**")
                        st.write(f"- **Индекс:** {selected_test['index_name']}")
                        st.write(f"- **Время тестирования:** {selected_test['test_timestamp']}")
                        st.write(f"- **ID сессии:** {selected_test['test_session_id']}")
                        
                        # Метрики производительности
                        if selected_test.get('search_time_ms'):
                            st.write(f"- **Время поиска:** {selected_test['search_time_ms']} мс")
                        if selected_test.get('generation_time_ms'):
                            st.write(f"- **Время генерации:** {selected_test['generation_time_ms']} мс")
                        if selected_test.get('total_time_ms'):
                            st.write(f"- **Общее время:** {selected_test['total_time_ms']} мс")
                    
                    with col2:
                        st.write("**Фактический ответ системы:**")
                        st.warning(selected_test['actual_answer'])
                        
                        # Автоматическая оценка (если есть)
                        metrics = selected_test.get('metrics', {})
                        if isinstance(metrics, str):
                            try:
                                metrics = json.loads(metrics)
                            except:
                                metrics = {}
                        
                        if metrics:
                            st.write("**Автоматическая оценка:**")
                            if 'score' in metrics:
                                auto_score = metrics['score']
                                if auto_score == 1:
                                    st.success("Автоматическая оценка: ПРАВИЛЬНО")
                                else:
                                    st.error("Автоматическая оценка: НЕПРАВИЛЬНО")
                            
                            if 'explanation' in metrics:
                                st.write("**Объяснение автоматической оценки:**")
                                st.text(metrics['explanation'])
                    
                    # Форма для ручной оценки
                    st.markdown("---")
                    st.subheader("Ваша оценка")
                    
                    with st.form(key=f"rating_form_{selected_test['id']}"):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            user_rating = st.radio(
                                "Оценка ответа:",
                                options=[True, False],
                                format_func=lambda x: "ПРАВИЛЬНО" if x else "НЕПРАВИЛЬНО",
                                key=f"rating_{selected_test['id']}"
                            )
                        
                        with col2:
                            user_comment = st.text_area(
                                "Комментарий (необязательно):",
                                placeholder="Укажите причину оценки, замечания или предложения...",
                                height=100,
                                key=f"comment_{selected_test['id']}"
                            )
                        
                        submitted = st.form_submit_button("Сохранить оценку", type="primary")
                        
                        if submitted:
                            # Сохраняем оценку в базу данных
                            try:
                                success = postgres_storage.save_user_rating(
                                    test_id=selected_test['id'],
                                    is_correct=user_rating,
                                    comment=user_comment.strip() if user_comment.strip() else None,
                                    rated_by="streamlit_user"
                                )
                                
                                if success:
                                    rating_text = "ПРАВИЛЬНО" if user_rating else "НЕПРАВИЛЬНО"
                                    st.success(f"Оценка сохранена: {rating_text}")
                                    
                                    # Обновляем страницу для загрузки следующего теста
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Ошибка при сохранении оценки")
                                    
                            except Exception as e:
                                st.error(f"Ошибка при сохранении оценки: {e}")
                    
                    # Навигация
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        if st.button("Предыдущий тест", disabled=selected_test_idx == 0):
                            st.session_state.rating_test_selector = selected_test_idx - 1
                            st.rerun()
                    
                    with col2:
                        st.write(f"Тест {selected_test_idx + 1} из {len(unrated_tests)}")
                    
                    with col3:
                        if st.button("Следующий тест", disabled=selected_test_idx == len(unrated_tests) - 1):
                            st.session_state.rating_test_selector = selected_test_idx + 1
                            st.rerun()
                
                # Статистика оценок
                st.markdown("---")
                st.subheader("Статистика оценок")
                
                try:
                    stats = postgres_storage.get_test_statistics_with_ratings()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Всего тестов", stats.get('total_tests', 0))
                    with col2:
                        st.metric("Оценено", stats.get('rated_tests', 0))
                    with col3:
                        st.metric("Не оценено", stats.get('unrated_count', 0))
                    with col4:
                        user_accuracy = stats.get('user_accuracy')
                        if user_accuracy is not None:
                            st.metric("Точность по оценкам", f"{user_accuracy:.1%}")
                        else:
                            st.metric("Точность по оценкам", "—")
                    
                    # Прогресс оценки
                    total_tests = stats.get('total_tests', 0)
                    rated_tests = stats.get('rated_tests', 0)
                    
                    if total_tests > 0:
                        progress = rated_tests / total_tests
                        st.progress(progress)
                        st.write(f"Прогресс оценки: {rated_tests}/{total_tests} ({progress:.1%})")
                        
                except Exception as e:
                    st.error(f"Ошибка при получении статистики: {e}")
                    
        except Exception as e:
            st.error(f"Ошибка подключения к базе данных: {e}")
            st.info("""
            **Возможные причины:**
            - PostgreSQL не запущен
            - Неверные настройки подключения
            - Не выполнена миграция базы данных
            
            **Решение:**
            1. Убедитесь, что PostgreSQL запущен
            2. Проверьте настройки в файле .env
            3. Выполните миграцию: `python migrate_ratings.py`
            """)
    
    elif mode == "Аналитика запросов":
        st.header("Аналитика пользовательских запросов")
        
        st.markdown("""
        **Анализ использования RAG системы**
        
        Здесь вы можете просмотреть статистику пользовательских запросов,
        популярные темы и эффективность системы.
        """)
        
        try:
            postgres_storage = PostgreSQLStorage()
            
            # Проверяем и создаем таблицу если нужно
            try:
                postgres_storage.create_user_queries_table()
                logger.info("Таблица user_queries проверена/создана")
            except Exception as e:
                logger.warning(f"Ошибка при создании таблицы user_queries: {e}")
            
            # Общая статистика
            st.subheader("Общая статистика")
            
            stats = postgres_storage.get_user_queries_statistics()
            
            if stats.get('total_queries', 0) == 0:
                st.info("Пока нет данных о пользовательских запросах. Выполните несколько поисковых запросов для накопления статистики.")
            else:
                # Основные метрики
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Всего запросов", stats.get('total_queries', 0))
                with col2:
                    st.metric("Уникальных пользователей", stats.get('unique_users', 0))
                with col3:
                    success_rate = 0
                    if stats.get('total_queries', 0) > 0:
                        success_rate = (stats.get('successful_queries', 0) / stats.get('total_queries', 1)) * 100
                    st.metric("Успешность", f"{success_rate:.1f}%")
                with col4:
                    avg_time = stats.get('avg_total_time_ms', 0) or 0
                    st.metric("Среднее время", f"{avg_time:.0f} мс")
                
                # Дополнительные метрики
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Уникальных сессий", stats.get('unique_sessions', 0))
                with col2:
                    st.metric("Индексов используется", stats.get('indices_used', 0))
                with col3:
                    st.metric("Неуспешных запросов", stats.get('failed_queries', 0))
                
                # Временные метрики
                st.subheader("Производительность")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    search_time = stats.get('avg_search_time_ms', 0) or 0
                    st.metric("Среднее время поиска", f"{search_time:.0f} мс")
                with col2:
                    gen_time = stats.get('avg_generation_time_ms', 0) or 0
                    st.metric("Среднее время генерации", f"{gen_time:.0f} мс")
                with col3:
                    total_time = stats.get('avg_total_time_ms', 0) or 0
                    st.metric("Среднее общее время", f"{total_time:.0f} мс")
                
                # Популярные запросы
                st.subheader("Популярные запросы")
                
                popular_queries = postgres_storage.get_popular_queries(limit=15)
                
                if popular_queries:
                    # Создаем датафрейм для отображения
                    popular_data = []
                    for query in popular_queries:
                        popular_data.append({
                            'Запрос': query['query_text'][:100] + '...' if len(query['query_text']) > 100 else query['query_text'],
                            'Количество': query['query_count'],
                            'Пользователей': query['unique_users'],
                            'Среднее время (мс)': f"{query.get('avg_time_ms', 0):.0f}",
                            'Последнее использование': query['last_used'].strftime('%Y-%m-%d %H:%M') if query['last_used'] else 'Н/Д'
                        })
                    
                    popular_df = pd.DataFrame(popular_data)
                    st.dataframe(popular_df, use_container_width=True)
                    
                    # График популярных запросов
                    if len(popular_queries) > 0:
                        st.subheader("График популярности")
                        
                        # Берем топ-10 для графика
                        top_queries = popular_queries[:10]
                        query_labels = [q['query_text'][:30] + '...' if len(q['query_text']) > 30 else q['query_text'] for q in top_queries]
                        query_counts = [q['query_count'] for q in top_queries]
                        
                        # Создаем датафрейм для графика
                        chart_data = pd.DataFrame({
                            'Запрос': query_labels,
                            'Количество запросов': query_counts
                        })
                        
                        st.bar_chart(chart_data.set_index('Запрос'))
                else:
                    st.info("Популярные запросы пока отсутствуют.")
                
                # Последние запросы
                st.subheader("Последние запросы")
                
                recent_queries = postgres_storage.get_recent_queries(limit=20)
                
                if recent_queries:
                    # Создаем датафрейм для последних запросов
                    recent_data = []
                    for query in recent_queries:
                        status = "Успешно" if query['success'] else "Ошибка"
                        
                        recent_data.append({
                            'Время': query['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                            'Запрос': query['query_text'][:80] + '...' if len(query['query_text']) > 80 else query['query_text'],
                            'Пользователь': query['user_id'],
                            'Индекс': query['index_name'] or 'Н/Д',
                            'Статус': status,
                            'Время выполнения (мс)': query.get('total_time_ms', 0) or 0
                        })
                    
                    recent_df = pd.DataFrame(recent_data)
                    st.dataframe(recent_df, use_container_width=True)
                    
                    # Детальный просмотр запроса
                    st.subheader("Детальный просмотр")
                    
                    query_options = [
                        f"{q['timestamp'].strftime('%H:%M:%S')} - {q['query_text'][:50]}..." 
                        if len(q['query_text']) > 50 
                        else f"{q['timestamp'].strftime('%H:%M:%S')} - {q['query_text']}"
                        for q in recent_queries
                    ]
                    
                    selected_query_idx = st.selectbox(
                        "Выберите запрос для детального просмотра:",
                        range(len(query_options)),
                        format_func=lambda x: query_options[x],
                        key="analytics_query_selector"
                    )
                    
                    if selected_query_idx is not None:
                        selected_query = recent_queries[selected_query_idx]
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write("**Детали запроса:**")
                            st.write(f"**Пользователь:** {selected_query['user_id']}")
                            st.write(f"**Время:** {selected_query['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"**Индекс:** {selected_query['index_name'] or 'Не указан'}")
                            st.write(f"**Статус:** {'Успешно' if selected_query['success'] else 'Ошибка'}")
                            
                            if selected_query.get('error_message'):
                                st.error(f"**Ошибка:** {selected_query['error_message']}")
                        
                        with col2:
                            st.write("**Производительность:**")
                            if selected_query.get('search_time_ms'):
                                st.write(f"**Время поиска:** {selected_query['search_time_ms']} мс")
                            if selected_query.get('generation_time_ms'):
                                st.write(f"**Время генерации:** {selected_query['generation_time_ms']} мс")
                            if selected_query.get('total_time_ms'):
                                st.write(f"**Общее время:** {selected_query['total_time_ms']} мс")
                        
                        st.write("**Полный текст запроса:**")
                        st.info(selected_query['query_text'])
                        
                        if selected_query.get('response_text'):
                            st.write("**Ответ системы:**")
                            st.success(selected_query['response_text'][:500] + '...' if len(selected_query['response_text']) > 500 else selected_query['response_text'])
                        
                        # Технологии
                        if selected_query.get('technologies_used'):
                            try:
                                technologies = json.loads(selected_query['technologies_used']) if isinstance(selected_query['technologies_used'], str) else selected_query['technologies_used']
                                st.write("**Использованные технологии:**")
                                
                                tech_info = []
                                if technologies.get('hyde'):
                                    tech_info.append("HyDE")
                                if technologies.get('reranking'):
                                    tech_info.append("Reranking")
                                
                                weights = f"Семантический: {technologies.get('semantic_weight', 'Н/Д')}, Ключевой: {technologies.get('keyword_weight', 'Н/Д')}"
                                
                                st.write(f"**Технологии:** {', '.join(tech_info) if tech_info else 'Базовый поиск'}")
                                st.write(f"**Веса поиска:** {weights}")
                                
                            except Exception as e:
                                st.write(f"**Технологии:** Ошибка парсинга данных: {e}")
                else:
                    st.info("Последние запросы отсутствуют.")
        
        except Exception as e:
            st.error(f"Ошибка при получении аналитики: {e}")
            st.info("""
            **Возможные причины:**
            - PostgreSQL не запущен
            - Таблица user_queries не создана
            - Ошибка подключения к базе данных
            
            **Решение:**
            1. Убедитесь, что PostgreSQL запущен
            2. Выполните несколько поисковых запросов для накопления данных
            """)
    
    # Информация о системе в сайдбаре
    with st.sidebar:
        st.markdown("---")
        st.subheader("О системе")
        st.markdown("""
        **RAG Система v2.0**
        
        **Технологии:**
        - OpenSearch для векторного поиска
        - Yandex GPT для генерации ответов
        - ColBERT для реранжирования
        - FastColBERT для быстрой оценки
        - PostgreSQL для хранения результатов тестирования
        
        **Возможности:**
        - Гибридный поиск (семантический + ключевой)
        - HyDE для улучшения запросов
        - Использует косинусное сходство векторов
        - Семантическое разбиение документов
        - Автоматическое тестирование качества
        - Ручная разметка результатов
        - Полная трассировка результатов
        """)


def main():
    """Основная функция запуска приложения."""
    create_streamlit_interface()


if __name__ == "__main__":
    main() 