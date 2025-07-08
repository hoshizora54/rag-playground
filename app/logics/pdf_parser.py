"""Парсер PDF документов с использованием pdfium и Yandex GPT для векторизации."""

import io
import os
import re
import uuid
import requests
import numpy as np
from typing import List, Dict, Any, Tuple, Union
import pypdfium2 as pdfium
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import config
from app.core.logger import logger
from app.schemas.document import DocumentChunk


class PDFParser:
    """Класс для парсинга PDF документов через pdfium и векторизации через Yandex GPT."""
    
    def __init__(self):
        """Инициализация парсера PDF."""
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
        self.file_path = None  # Сохраняем путь к файлу для извлечения изображений
        
    def parse_pdf(self, file_path: str, use_semantic_chunking: bool = False, 
                  semantic_threshold: float = 0.8) -> List[DocumentChunk]:
        """
        Парсит PDF файл и возвращает список чанков с векторами.
        
        Args:
            file_path: Путь к PDF файлу
            use_semantic_chunking: Использовать семантическое разбиение на чанки
            semantic_threshold: Порог косинусного сходства для семантического разбиения
            
        Returns:
            Список чанков документа с векторами
        """
        try:
            chunks = []
            self.file_path = file_path  # Сохраняем путь для извлечения изображений
            
            # Открываем PDF через pdfium
            pdf = pdfium.PdfDocument(file_path)
            
            for page_num in range(len(pdf)):
                page = pdf.get_page(page_num)
                
                # Извлекаем текст страницы
                text = page.get_textpage().get_text_range()
                
                # Извлекаем изображения
                images = self._extract_images_from_page(page, page_num)
                
                # Выбираем метод разбиения
                if use_semantic_chunking:
                    logger.info(f"Применяем семантическое разбиение для страницы {page_num + 1}")
                    page_chunks = self._split_text_semantically(text, semantic_threshold, page_num + 1, images)
                    chunks.extend(page_chunks)
                else:
                    # Обычное разбиение на абзацы
                    paragraphs = self._split_text_into_paragraphs(text)
                    
                    for paragraph_idx, paragraph in enumerate(paragraphs):
                        if paragraph.strip():  # Пропускаем пустые абзацы
                            # Векторизуем текст через Yandex GPT
                            vector = self._create_embedding(paragraph)
                            
                            chunk = DocumentChunk(
                                id=str(uuid.uuid4()),
                                text=paragraph.strip(),
                                vector=vector,
                                page_number=page_num + 1,
                                paragraph_index=paragraph_idx,
                                images=images
                            )
                            chunks.append(chunk)
                        
                page.close()
                
            pdf.close()
            
            logger.info(f"Успешно извлечено {len(chunks)} чанков из PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге PDF: {e}")
            raise
    
    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """
        Разбивает текст на абзацы.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список абзацев
        """
        # Разбиваем по двойным переносам строк и другим разделителям абзацев
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        # Фильтруем пустые абзацы и очень короткие строки
        filtered_paragraphs = []
        for paragraph in paragraphs:
            cleaned = paragraph.strip()
            if len(cleaned) > 20:  # Минимальная длина абзаца
                filtered_paragraphs.append(cleaned)
                
        return filtered_paragraphs
    
    def _split_text_semantically(self, text: str, threshold: float, page_number: int, 
                                images: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Разбивает текст на семантические чанки.
        
        Args:
            text: Исходный текст страницы
            threshold: Порог косинусного сходства для объединения предложений
            page_number: Номер страницы
            images: Изображения со страницы
            
        Returns:
            Список семантических чанков
        """
        try:
            # 1. Разбиваем на предложения
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= 1:
                # Если мало предложений, возвращаем как один чанк
                vector = self._create_embedding(text) if text.strip() else []
                return [DocumentChunk(
                    id=str(uuid.uuid4()),
                    text=text.strip(),
                    vector=vector,
                    page_number=page_number,
                    paragraph_index=0,
                    images=images
                )] if text.strip() else []
            
            logger.info(f"Разбили текст на {len(sentences)} предложений")
            
            # 2. Создаем векторы для предложений
            sentence_vectors = []
            valid_sentences = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # Минимальная длина предложения
                    vector = self._create_embedding(sentence)
                    if vector:
                        sentence_vectors.append(vector)
                        valid_sentences.append(sentence.strip())
            
            if not valid_sentences:
                return []
            
            logger.info(f"Создали векторы для {len(valid_sentences)} предложений")
            
            # 3. Группируем предложения в семантические чанки
            chunks = []
            current_chunk_sentences = [valid_sentences[0]]
            current_chunk_vectors = [sentence_vectors[0]]
            
            for i in range(1, len(valid_sentences)):
                # Вычисляем косинусное сходство с предыдущим предложением
                prev_vector = np.array(sentence_vectors[i-1]).reshape(1, -1)
                curr_vector = np.array(sentence_vectors[i]).reshape(1, -1)
                
                similarity = cosine_similarity(prev_vector, curr_vector)[0][0]
                
                logger.debug(f"Сходство между предложениями {i-1} и {i}: {similarity:.3f}")
                
                if similarity >= threshold:
                    # Добавляем к текущему чанку
                    current_chunk_sentences.append(valid_sentences[i])
                    current_chunk_vectors.append(sentence_vectors[i])
                else:
                    # Завершаем текущий чанк и начинаем новый
                    if current_chunk_sentences:
                        chunk = self._create_semantic_chunk(
                            current_chunk_sentences, 
                            current_chunk_vectors,
                            page_number,
                            len(chunks),
                            images
                        )
                        if chunk:
                            chunks.append(chunk)
                    
                    # Начинаем новый чанк
                    current_chunk_sentences = [valid_sentences[i]]
                    current_chunk_vectors = [sentence_vectors[i]]
            
            # Добавляем последний чанк
            if current_chunk_sentences:
                chunk = self._create_semantic_chunk(
                    current_chunk_sentences,
                    current_chunk_vectors, 
                    page_number,
                    len(chunks),
                    images
                )
                if chunk:
                    chunks.append(chunk)
            
            logger.info(f"Создано {len(chunks)} семантических чанков из {len(valid_sentences)} предложений")
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка при семантическом разбиении: {e}")
            # Fallback к обычному разбиению
            paragraphs = self._split_text_into_paragraphs(text)
            fallback_chunks = []
            
            for idx, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    vector = self._create_embedding(paragraph)
                    chunk = DocumentChunk(
                        id=str(uuid.uuid4()),
                        text=paragraph.strip(),
                        vector=vector,
                        page_number=page_number,
                        paragraph_index=idx,
                        images=images
                    )
                    fallback_chunks.append(chunk)
            
            return fallback_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Разбивает текст на предложения.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список предложений
        """
        # Разбиваем по точкам, восклицательным и вопросительным знакам
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Очищаем и фильтруем
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) > 5:  # Минимальная длина предложения
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    def _create_semantic_chunk(self, sentences: List[str], vectors: List[List[float]], 
                              page_number: int, chunk_index: int, 
                              images: List[Dict[str, Any]]) -> DocumentChunk:
        """
        Создает семантический чанк из группы предложений.
        
        Args:
            sentences: Список предложений
            vectors: Векторы предложений
            page_number: Номер страницы
            chunk_index: Индекс чанка
            images: Изображения
            
        Returns:
            Семантический чанк
        """
        try:
            # Объединяем предложения в текст
            chunk_text = '. '.join(sentences)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            
            # Создаем усредненный вектор или векторизуем весь текст заново
            if len(vectors) > 1:
                # Усредняем векторы предложений
                chunk_vector = np.mean(vectors, axis=0).tolist()
            else:
                chunk_vector = vectors[0]
            
            return DocumentChunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                vector=chunk_vector,
                page_number=page_number,
                paragraph_index=chunk_index,
                images=images
            )
            
        except Exception as e:
            logger.error(f"Ошибка при создании семантического чанка: {e}")
            return None
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создает векторное представление текста через Yandex GPT API.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Вектор embeddings
        """
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
                
                if not embedding:
                    logger.warning("Получен пустой вектор от Yandex GPT API")
                    return [0.0] * config.VECTOR_DIMENSION
                    
                return embedding
            else:
                logger.error(f"Ошибка Yandex GPT API: {response.status_code} - {response.text}")
                return [0.0] * config.VECTOR_DIMENSION
                
        except Exception as e:
            logger.error(f"Ошибка при создании embedding: {e}")
            # Возвращаем пустой вектор в случае ошибки
            return [0.0] * config.VECTOR_DIMENSION
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Извлекает изображения со страницы PDF.
        
        Args:
            page: Страница PDF (pypdfium2)
            page_num: Номер страницы
            
        Returns:
            Список словарей с информацией об изображениях
        """
        try:
            images = []
            
            # pypdfium2 не поддерживает извлечение отдельных изображений
            # Используем PyMuPDF для извлечения изображений как гибридный подход
            try:
                import fitz  # PyMuPDF
                
                if not self.file_path:
                    logger.warning("Путь к файлу не сохранен, извлечение изображений невозможно")
                    return []
                
                # Открываем тот же документ через PyMuPDF
                doc_pymupdf = fitz.open(self.file_path)
                page_pymupdf = doc_pymupdf[page_num]
                
                # Получаем список изображений на странице
                image_list = page_pymupdf.get_images()
                
                if image_list:
                    logger.info(f"Найдено {len(image_list)} изображений на странице {page_num + 1}")
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # img = [xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter]
                            xref = img[0]
                            
                            # Извлекаем изображение
                            base_image = doc_pymupdf.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Генерируем имя файла
                            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                            
                            # Создаем PIL изображение
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            
                            # Добавляем информацию об изображении
                            image_info = {
                                "filename": image_filename,
                                "pil_image": pil_image,
                                "page_number": page_num + 1,
                                "index": img_index + 1,
                                "width": pil_image.width,
                                "height": pil_image.height,
                                "format": image_ext.upper()
                            }
                            
                            images.append(image_info)
                            logger.debug(f"Извлечено изображение: {image_filename}")
                            
                        except Exception as e:
                            logger.warning(f"Ошибка при извлечении изображения {img_index} со страницы {page_num + 1}: {e}")
                            continue
                
                doc_pymupdf.close()
                
            except ImportError:
                logger.warning("PyMuPDF не установлен. Попробуем альтернативный метод.")
                
                # Альтернативный вариант: рендерим всю страницу как изображение
                try:
                    logger.info(f"Рендерим страницу {page_num + 1} как изображение")
                    
                    # Рендерим страницу в изображение
                    pix = page.render(scale=2.0)  # Увеличиваем разрешение
                    pil_image = pix.to_pil()
                    
                    # Генерируем имя файла для полной страницы
                    page_image_filename = f"page_{page_num + 1}_full.png"
                    
                    # Добавляем информацию об изображении страницы
                    image_info = {
                        "filename": page_image_filename,
                        "pil_image": pil_image,
                        "page_number": page_num + 1,
                        "index": 1,
                        "width": pil_image.width,
                        "height": pil_image.height,
                        "format": "PNG"
                    }
                    
                    images.append(image_info)
                    logger.info(f"Создано изображение страницы: {page_image_filename}")
                    
                except Exception as render_error:
                    logger.error(f"Ошибка при рендеринге страницы {page_num + 1}: {render_error}")
            
            return images
            
        except Exception as e:
            logger.error(f"Критическая ошибка при извлечении изображений со страницы {page_num + 1}: {e}")
            return []