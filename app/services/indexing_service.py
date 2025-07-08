"""Сервис для индексации документов."""

import uuid
import tempfile
import os
from typing import List, Dict, Any
from fastapi import HTTPException, UploadFile

from app.core.logger import logger
from app.logics.pdf_parser import PDFParser
from app.logics.opensearch import opensearch_worker
from app.logics.minio import minio_worker
from app.schemas.document import DocumentUploadRequest, DocumentUploadResponse, DocumentChunk


class IndexingService:
    """Сервис для индексации PDF документов."""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
    
    async def process_document(self, file: UploadFile, index_name: str, 
                              use_semantic_chunking: bool = False,
                              semantic_threshold: float = 0.8) -> DocumentUploadResponse:
        """
        Обрабатывает и индексирует PDF документ.
        
        Args:
            file: Загружаемый PDF файл
            index_name: Имя индекса для сохранения
            use_semantic_chunking: Использовать семантическое разбиение на чанки
            semantic_threshold: Порог косинусного сходства для семантического разбиения
            
        Returns:
            Информация о результате индексации
        """
        try:
            # Проверяем тип файла
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Поддерживаются только PDF файлы")
            
            # Генерируем уникальный ID документа
            document_id = str(uuid.uuid4())
            
            # Читаем содержимое файла
            file_content = await file.read()
            
            # Сохраняем во временный файл для парсинга
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Парсим PDF и извлекаем чанки
                chunks = self.pdf_parser.parse_pdf(
                    temp_file_path, 
                    use_semantic_chunking=use_semantic_chunking,
                    semantic_threshold=semantic_threshold
                )
                
                if not chunks:
                    raise HTTPException(status_code=400, detail="Не удалось извлечь текст из PDF")
                
                # Загружаем оригинальный PDF в MinIO
                pdf_path = minio_worker.upload_pdf(file_content, file.filename)
                
                # Собираем и сохраняем все изображения в MinIO
                all_saved_images = self._save_images_to_minio(chunks, file.filename)
                
                # Очищаем чанки от PIL объектов для индексации в OpenSearch
                clean_chunks = self._clean_chunks_for_indexing(chunks)
                
                # Подготавливаем данные для сохранения в MinIO
                document_data = {
                    "document_id": document_id,
                    "file_name": file.filename,
                    "chunks_count": len(chunks),
                    "pdf_path": pdf_path,
                    "chunks": [
                        {
                            "id": chunk.id,
                            "text": chunk.text,
                            "page_number": chunk.page_number,
                            "paragraph_index": chunk.paragraph_index,
                            "images": chunk.images  # Уже очищенные имена файлов
                        }
                        for chunk in clean_chunks
                    ]
                }
                
                # Сохраняем JSON данные в MinIO
                json_path = minio_worker.upload_json(document_data, file.filename)
                
                # Индексируем документ в OpenSearch (только очищенные чанки)
                success = opensearch_worker.index_document(
                    index_name=index_name,
                    chunks=clean_chunks,
                    document_id=document_id,
                    file_name=file.filename
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail="Ошибка при индексации документа")
                
                logger.info(f"Документ успешно обработан: {file.filename}")
                
                return DocumentUploadResponse(
                    document_id=document_id,
                    chunks_count=len(chunks),
                    minio_path=json_path
                )
                
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при обработке документа: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки документа: {str(e)}")
    
    async def delete_document(self, document_id: str, index_name: str, file_name: str):
        """
        Удаляет документ из индекса и MinIO.
        
        Args:
            document_id: ID документа
            index_name: Имя индекса
            file_name: Имя файла
        """
        try:
            # Удаляем из OpenSearch
            opensearch_worker.delete_document(index_name, document_id)
            
            # Удаляем из MinIO
            minio_worker.delete_folder(file_name)
            
            logger.info(f"Документ удален: {document_id}")
            
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка удаления документа: {str(e)}")
    
    def _save_images_to_minio(self, chunks: List[DocumentChunk], file_name: str) -> Dict[str, str]:
        """
        Сохраняет все изображения из чанков в MinIO.
        
        Args:
            chunks: Список чанков документа
            file_name: Имя исходного файла
            
        Returns:
            Словарь сохраненных изображений (путь -> MinIO путь)
        """
        saved_images = {}
        
        try:
            for chunk in chunks:
                if chunk.images and isinstance(chunk.images, list):
                    for image_info in chunk.images:
                        if isinstance(image_info, dict) and "pil_image" in image_info:
                            image_filename = image_info["filename"]
                            pil_image = image_info["pil_image"]
                            
                            # Сохраняем изображение в MinIO
                            try:
                                minio_path = minio_worker.upload_image(
                                    image=pil_image,
                                    file_name=file_name,
                                    image_name=image_filename
                                )
                                saved_images[image_filename] = minio_path
                                logger.debug(f"Изображение сохранено в MinIO: {minio_path}")
                                
                            except Exception as e:
                                logger.error(f"Ошибка при сохранении изображения {image_filename}: {e}")
                                continue
            
            logger.info(f"Сохранено {len(saved_images)} изображений в MinIO")
            return saved_images
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении изображений в MinIO: {e}")
            return {}
    
    def _get_image_names_for_chunk(self, images: List) -> List[str]:
        """
        Извлекает имена файлов изображений из данных чанка.
        
        Args:
            images: Список изображений (может быть dict или str)
            
        Returns:
            Список имен файлов изображений
        """
        image_names = []
        
        if not images:
            return image_names
            
        for image in images:
            if isinstance(image, dict) and "filename" in image:
                image_names.append(image["filename"])
            elif isinstance(image, str):

                import os
                image_names.append(os.path.basename(image))
        
        return image_names
    
    def _clean_chunks_for_indexing(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Создает копии чанков без PIL объектов для индексации в OpenSearch.
        
        Args:
            chunks: Исходные чанки с PIL объектами
            
        Returns:
            Очищенные чанки только с именами файлов изображений
        """
        clean_chunks = []
        
        for chunk in chunks:
            # Получаем только имена файлов изображений
            image_names = self._get_image_names_for_chunk(chunk.images)
            
            # Создаем новый чанк без PIL объектов
            clean_chunk = DocumentChunk(
                id=chunk.id,
                text=chunk.text,
                vector=chunk.vector,
                page_number=chunk.page_number,
                paragraph_index=chunk.paragraph_index,
                images=image_names  # Только имена файлов
            )
            
            clean_chunks.append(clean_chunk)
        
        return clean_chunks


indexing_service = IndexingService() 