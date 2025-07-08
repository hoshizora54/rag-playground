"""Streamlit интерфейс для RAG системы с подключением к FastAPI бэкенду."""

import streamlit as st
import pandas as pd
import asyncio
from typing import List, Dict, Any
import io
import time
import json
import logging

# Импорт API клиента
from app.client.api_client import api_client
from app.logics.postgres_storage import postgres_storage

logger = logging.getLogger(__name__)


def run_async_fn(fn, *args):
    """Выполняет асинхронную функцию в синхронном контексте Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(fn(*args))
    finally:
        loop.close()


async def check_api_connection():
    """Проверяет подключение к API."""
    health = await api_client.health_check()
    return health.get("status") == "healthy"


async def upload_document_async(file, index_name: str, use_semantic_chunking: bool = False, 
                                semantic_threshold: float = 0.8) -> str:
    """Асинхронная загрузка документа через API."""
    try:
        # Читаем содержимое файла
        file_data = file.read()
        filename = file.name
        
        # Вызываем API
        result = await api_client.upload_document(
            file_data=file_data,
            filename=filename,
            index_name=index_name,
            use_semantic_chunking=use_semantic_chunking,
            semantic_threshold=semantic_threshold
        )
        
        if result.get("error"):
            return f"Ошибка: {result.get('message', 'Неизвестная ошибка')}"
        
        if result.get("success"):
            message = result.get("message", "Документ успешно загружен")
            # chunks_count находится на верхнем уровне ответа, а не в data
            chunks_count = result.get("chunks_count", 0)
            return f"{message}. Создано {chunks_count} чанков."
        else:
            return f"Ошибка: {result.get('message', 'Неизвестная ошибка')}"
            
    except Exception as e:
        return f"Ошибка при загрузке: {str(e)}"


async def get_indices_async() -> List[str]:
    """Получает список доступных индексов через API."""
    try:
        indices = await api_client.get_indices()
        return indices
    except Exception as e:
        logger.error(f"Ошибка получения индексов: {e}")
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
    """Асинхронный поиск и генерация ответа через API."""
    
    user_query = {
        "query_text": query_text,
        "index_name": index_name,
        "sematic": semantic_weight,
        "keyword": keyword_weight,
        "size": size,
        "use_hyde": use_hyde,
        "hyde_num_hypotheses": hyde_num_hypotheses,
        "reranking": reranking,
        "k": size,  # Для совместимости
        "fields": ["text"],  # Поля для поиска по умолчанию
        "trashold": 0.0,  # Пороги по умолчанию
        "trashold_bertscore": 0.0,
        "query_embed": None  # Будет сгенерирован на бэкенде
    }
    
    # Получаем результат через API
    result = await api_client.generate_answer(user_query)
    return result


async def evaluate_rag_async(
    test_data: List[Dict[str, str]], 
    index_name: str,
    search_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Асинхронная оценка RAG через API."""
    
    # Создаем CSV файл из тестовых данных
    df = pd.DataFrame(test_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')
    
    # Вызываем API
    result = await api_client.evaluate_rag_batch(
        file_data=csv_data,
        filename="test_data.csv",
        index_name=index_name,
        search_params=search_params
    )
    
    if result.get("error"):
        raise Exception(f"Ошибка API: {result.get('message')}")
    
    return result.get("data", {})


def create_streamlit_interface():
    """Создает интерфейс Streamlit."""
    
    # Настройка страницы
    st.set_page_config(
        page_title="RAG PDF Система",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Заголовок
    st.title("📚 RAG PDF Система")
    st.markdown("---")
    
    # Проверка подключения к API
    if "api_status" not in st.session_state:
        with st.spinner("Проверяю подключение к API..."):
            api_connected = run_async_fn(check_api_connection)
            st.session_state.api_status = api_connected
    
    # Отображение статуса API
    if st.session_state.api_status:
        st.success("✅ Подключение к API установлено")
    else:
        st.error("❌ Не удается подключиться к API. Убедитесь, что бэкенд запущен на localhost:8000")
        st.info("Запустите бэкенд командой: `uvicorn main:app --reload`")
        if st.button("Повторить проверку"):
            st.session_state.api_status = run_async_fn(check_api_connection)
            st.experimental_rerun()
        return
    
    # Боковая панель с режимами
    with st.sidebar:
        st.header("🎛️ Панель управления")
        
        mode = st.selectbox(
            "Выберите режим:",
            [
                "Поиск и ответы",
                "Загрузка документов", 
                "Тестирование качества",
                "Разметка ответов",
                "Аналитика пользователей"
            ]
        )
        
        if mode == "Поиск и ответы":
            st.subheader("Параметры поиска")
            
            # Выбор индекса
            with st.spinner("Получаю список индексов..."):
                indices = run_async_fn(get_indices_async)
            
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
                    
                    if result.get("error"):
                        st.error(f"Ошибка API: {result.get('message')}")
                        return
                    
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
                            st.subheader("Найденные изображения")
                            for i, img in enumerate(images, 1):
                                st.image(img.get("base64_data", ""), caption=f"Изображение {i}")
                
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
                    "Порог косинусного сходства",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="Чем выше значение, тем более похожие предложения группируются в один чанк"
                )
            
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
            with st.spinner("Получаю список индексов..."):
                indices = run_async_fn(get_indices_async)
            
            if indices:
                for idx in indices:
                    st.text(f"📁 {idx}")
            else:
                st.info("Индексы не найдены")
    
    elif mode == "Тестирование качества":
        st.header("Тестирование качества RAG")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Загрузка тестового файла")
            
            # Загрузка CSV файла
            test_file = st.file_uploader(
                "Выберите CSV файл с тестовыми данными",
                type=['csv'],
                help="CSV файл должен содержать колонки 'question' и 'expected_answer'"
            )
            
            if test_file:
                try:
                    # Проверяем размер файла перед чтением
                    file_size = test_file.size if hasattr(test_file, 'size') else 0
                    if file_size == 0:
                        st.error("Файл пуст. Проверьте содержимое файла.")
                    else:
                        # Сбрасываем курсор в начало файла
                        test_file.seek(0)
                        df = pd.read_csv(test_file, encoding='utf-8')
                        
                        st.write("**Предварительный просмотр данных:**")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Проверяем наличие нужных колонок
                        columns = df.columns.tolist()
                        question_cols = [col for col in columns if 'question' in col.lower() or 'вопрос' in col.lower()]
                        answer_cols = [col for col in columns if 'answer' in col.lower() or 'ответ' in col.lower()]
                        
                        if not question_cols or not answer_cols:
                            st.error("В CSV файле должны быть колонки с вопросами (question/вопрос) и ответами (answer/ответ)")
                        else:
                            st.success(f"Найдено {len(df)} строк для тестирования")
                            st.info(f"Колонка вопросов: {question_cols[0]}")
                            st.info(f"Колонка ответов: {answer_cols[0]}")
                
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {e}")
        
        with col2:
            st.subheader("Настройки тестирования")
            
            # Выбор индекса для тестирования
            with st.spinner("Получаю список индексов..."):
                test_indices = run_async_fn(get_indices_async)
            
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
                    
                    columns = df.columns.tolist()
                    question_cols = [col for col in columns if 'question' in col.lower() or 'вопрос' in col.lower()]
                    answer_cols = [col for col in columns if 'answer' in col.lower() or 'ответ' in col.lower()]
                    
                    if not question_cols or not answer_cols:
                        st.error("В CSV файле должны быть колонки с вопросами и ответами")
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
                                'reranking': test_reranking,
                                'k': test_size
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
                                total_tests = eval_result.get("total_tests", 0)
                                st.metric("Всего тестов", total_tests)
                                
                            with col2:
                                correct_answers = eval_result.get("correct_answers", 0)
                                st.metric("Правильных ответов", correct_answers)
                                
                            with col3:
                                accuracy = eval_result.get("accuracy", 0)
                                st.metric("Точность", f"{accuracy:.1%}")
                                
                            with col4:
                                avg_time = eval_result.get("average_time", 0)
                                st.metric("Среднее время", f"{avg_time:.2f} сек")
                            
                            # Детальные результаты
                            st.subheader("Детальные результаты")
                            
                            test_results = eval_result.get("test_results", [])
                            if test_results:
                                # Преобразуем в DataFrame для отображения
                                results_data = []
                                for test in test_results:
                                    results_data.append({
                                        'ID': test.get('test_id', ''),
                                        'Вопрос': test.get('question', '')[:50] + '...' if len(test.get('question', '')) > 50 else test.get('question', ''),
                                        'Ожидаемый ответ': test.get('expected_answer', '')[:30] + '...' if len(test.get('expected_answer', '')) > 30 else test.get('expected_answer', ''),
                                        'Правильность': "✅" if test.get('score', 0) == 1 else "❌",
                                        'Время (сек)': f"{test.get('time_taken', 0):.2f}"
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Выбор теста для детального просмотра
                                test_options = [f"Тест {test['test_id']}: {test['question'][:50]}..." 
                                              for test in test_results]
                                
                                selected_test_idx = st.selectbox(
                                    "Выберите тест для детального просмотра:",
                                    range(len(test_options)),
                                    format_func=lambda x: test_options[x]
                                )
                                
                                if selected_test_idx is not None:
                                    selected_test = test_results[selected_test_idx]
                                    
                                    # Отображение деталей теста
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Вопрос:**")
                                        st.write(selected_test.get('question', ''))
                                        
                                        st.write("**Ожидаемый ответ:**")
                                        st.write(selected_test.get('expected_answer', ''))
                                    
                                    with col2:
                                        st.write("**Фактический ответ:**")
                                        st.write(selected_test.get('actual_answer', ''))
                                        
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
        
        # Получение неразмеченных тестов
        try:
            unrated_tests = postgres_storage.get_unrated_tests(limit=50)
            
            if not unrated_tests:
                st.info("Все доступные тесты уже размечены! 🎉")
                
                # Статистика разметки
                rating_stats = postgres_storage.get_rating_statistics()
                if rating_stats:
                    st.subheader("Статистика разметки")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего размечено", rating_stats.get('total_rated', 0))
                    with col2:
                        st.metric("Правильных", rating_stats.get('correct_count', 0))
                    with col3:
                        accuracy = 0
                        if rating_stats.get('total_rated', 0) > 0:
                            accuracy = (rating_stats.get('correct_count', 0) / rating_stats.get('total_rated', 1)) * 100
                        st.metric("Точность", f"{accuracy:.1f}%")
            else:
                st.success(f"Найдено {len(unrated_tests)} неразмеченных тестов")
                
                # Выбор теста для разметки
                test_options = [
                    f"Тест #{test['id']}: {test['question'][:80]}..." 
                    if len(test['question']) > 80 
                    else f"Тест #{test['id']}: {test['question']}"
                    for test in unrated_tests
                ]
                
                selected_test_idx = st.selectbox(
                    "Выберите тест для разметки:",
                    range(len(test_options)),
                    format_func=lambda x: test_options[x]
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
                        st.write("**Ответ системы RAG:**")
                        st.warning(selected_test['actual_answer'])
                        
                        # Форма для разметки
                        st.write("**Оценка качества ответа:**")
                        
                        is_correct = st.radio(
                            "Является ли ответ правильным?",
                            options=[True, False],
                            format_func=lambda x: "✅ Правильно" if x else "❌ Неправильно",
                            key=f"rating_{selected_test['id']}"
                        )
                        
                        comment = st.text_area(
                            "Комментарий (опционально):",
                            placeholder="Объясните почему ответ правильный или неправильный...",
                            key=f"comment_{selected_test['id']}"
                        )
                        
                        # Кнопка сохранения разметки
                        if st.button("Сохранить оценку", type="primary", key=f"save_{selected_test['id']}"):
                            try:
                                postgres_storage.save_user_rating(
                                    test_id=selected_test['id'],
                                    is_correct=is_correct,
                                    comment=comment,
                                    rated_by="streamlit_user"
                                )
                                st.success("Оценка сохранена! ✅")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Ошибка при сохранении оценки: {e}")
        
        except Exception as e:
            st.error(f"Ошибка при получении данных: {e}")
    
    elif mode == "Аналитика пользователей":
        st.header("Аналитика пользовательских запросов")
        
        st.markdown("""
        **Анализ активности пользователей**
        
        Здесь отображается статистика по запросам пользователей, включая популярные запросы,
        время отклика системы и общую производительность.
        """)
        
        try:
            # Получение общей статистики
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
                    avg_search = stats.get('avg_search_time_ms', 0) or 0
                    st.metric("Среднее время поиска", f"{avg_search:.0f} мс")
                with col2:
                    avg_generation = stats.get('avg_generation_time_ms', 0) or 0
                    st.metric("Среднее время генерации", f"{avg_generation:.0f} мс")
                with col3:
                    max_time = stats.get('max_total_time_ms', 0) or 0
                    st.metric("Максимальное время", f"{max_time:.0f} мс")
                
                # Популярные запросы
                st.subheader("Популярные запросы")
                
                popular_queries = postgres_storage.get_popular_queries(limit=10)
                
                if popular_queries:
                    popular_data = []
                    for query in popular_queries:
                        popular_data.append({
                            'Запрос': query['query_text'][:80] + '...' if len(query['query_text']) > 80 else query['query_text'],
                            'Количество запросов': query['query_count'],
                            'Последний раз': query['last_used'].strftime('%Y-%m-%d %H:%M:%S'),
                            'Среднее время (мс)': f"{query.get('avg_time_ms', 0):.0f}"
                        })
                    
                    popular_df = pd.DataFrame(popular_data)
                    st.dataframe(popular_df, use_container_width=True)
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
                else:
                    st.info("Последние запросы отсутствуют.")
        
        except Exception as e:
            st.error(f"Ошибка при получении аналитики: {e}")


def main():
    """Основная функция запуска приложения."""
    create_streamlit_interface()


if __name__ == "__main__":
    main() 