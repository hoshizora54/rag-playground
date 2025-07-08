"""Модуль для работы с MinIO хранилищем."""

import io
import json
from typing import List, Dict, Any, Optional
from minio import Minio
from minio.error import S3Error
from PIL import Image

from app.core.config import config
from app.core.logger import logger
from app.schemas.document import DocumentChunk


class MinioWorker:
    """Класс для работы с MinIO хранилищем."""
    
    def __init__(self):
        """Инициализация подключения к MinIO."""
        self.client = Minio(
            endpoint=config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE
        )
        self.bucket_name = config.MINIO_BUCKET
        
        # Создаем bucket если его нет
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Проверяет и создает bucket если его нет."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Создан bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Ошибка при создании bucket: {e}")
            raise
    
    def upload_pdf(self, file_data: bytes, file_name: str) -> str:
        """
        Загружает PDF файл в MinIO.
        
        Args:
            file_data: Данные PDF файла
            file_name: Имя файла
            
        Returns:
            Путь к файлу в MinIO
        """
        try:
            folder_name = self._get_folder_name(file_name)
            object_path = f"{folder_name}/original.pdf"
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_path,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type="application/pdf"
            )
            
            logger.info(f"PDF файл загружен: {object_path}")
            return object_path
            
        except S3Error as e:
            logger.error(f"Ошибка при загрузке PDF: {e}")
            raise
    
    def upload_json(self, data: Dict[str, Any], file_name: str) -> str:
        """
        Загружает JSON данные в MinIO.
        
        Args:
            data: Данные для сохранения
            file_name: Имя исходного файла
            
        Returns:
            Путь к JSON файлу в MinIO
        """
        try:
            folder_name = self._get_folder_name(file_name)
            object_path = f"{folder_name}/data.json"
            
            json_data = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_path,
                data=io.BytesIO(json_data),
                length=len(json_data),
                content_type="application/json"
            )
            
            logger.info(f"JSON данные загружены: {object_path}")
            return object_path
            
        except S3Error as e:
            logger.error(f"Ошибка при загрузке JSON: {e}")
            raise
    
    def upload_image(self, image: Image.Image, file_name: str, image_name: str) -> str:
        """
        Загружает изображение в MinIO.
        
        Args:
            image: PIL изображение
            file_name: Имя исходного PDF файла
            image_name: Имя изображения
            
        Returns:
            Путь к изображению в MinIO
        """
        try:
            folder_name = self._get_folder_name(file_name)
            object_path = f"{folder_name}/images/{image_name}"
            
            # Конвертируем изображение в байты
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_path,
                data=io.BytesIO(img_data),
                length=len(img_data),
                content_type="image/png"
            )
            
            logger.info(f"Изображение загружено: {object_path}")
            return object_path
            
        except S3Error as e:
            logger.error(f"Ошибка при загрузке изображения: {e}")
            raise
    
    def get_json_data(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Получает JSON данные из MinIO.
        
        Args:
            file_name: Имя исходного файла
            
        Returns:
            Данные JSON или None если файл не найден
        """
        try:
            folder_name = self._get_folder_name(file_name)
            object_path = f"{folder_name}/data.json"
            
            response = self.client.get_object(self.bucket_name, object_path)
            data = json.loads(response.data.decode('utf-8'))
            
            logger.info(f"JSON данные получены: {object_path}")
            return data
            
        except S3Error as e:
            logger.warning(f"JSON файл не найден: {object_path}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении JSON: {e}")
            return None
    
    def get_image(self, file_name: str, image_name: str) -> Optional[bytes]:
        """
        Получает изображение из MinIO.
        
        Args:
            file_name: Имя исходного файла
            image_name: Имя изображения
            
        Returns:
            Данные изображения или None если не найдено
        """
        try:
            folder_name = self._get_folder_name(file_name)
            object_path = f"{folder_name}/images/{image_name}"
            
            response = self.client.get_object(self.bucket_name, object_path)
            return response.data
            
        except S3Error as e:
            logger.warning(f"Изображение не найдено: {object_path}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении изображения: {e}")
            return None
    
    def list_files(self, prefix: str = "") -> List[str]:
        """
        Получает список файлов в MinIO.
        
        Args:
            prefix: Префикс для фильтрации
            
        Returns:
            Список путей к файлам
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            return [obj.object_name for obj in objects]
            
        except S3Error as e:
            logger.error(f"Ошибка при получении списка файлов: {e}")
            return []
    
    def get_image_url(self, file_name: str, image_name: str, expires: int = 3600) -> Optional[str]:
        """
        Получает presigned URL для изображения.
        
        Args:
            file_name: Имя исходного файла
            image_name: Имя изображения
            expires: Время жизни URL в секундах (по умолчанию 1 час)
            
        Returns:
            Presigned URL для изображения или None если не найдено
        """
        try:
            folder_name = self._get_folder_name(file_name)
            object_path = f"{folder_name}/images/{image_name}"
            
            # Проверяем существование файла
            try:
                self.client.stat_object(self.bucket_name, object_path)
            except S3Error:
                logger.warning(f"Изображение не найдено: {object_path}")
                return None
            
            # Генерируем presigned URL
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_path,
                expires=expires
            )
            
            logger.info(f"Сгенерирован URL для изображения: {object_path}")
            return url
            
        except S3Error as e:
            logger.error(f"Ошибка при генерации URL для изображения: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при генерации URL: {e}")
            return None
    
    def get_multiple_images(self, document_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Получает данные изображений из MinIO для нескольких изображений из разных документов.
        
        Args:
            document_images: Список словарей с информацией об изображениях
                            Каждый словарь должен содержать 'file_name' и 'images'
            
        Returns:
            Список словарей с данными изображений
        """
        result_images = []
        
        for doc_info in document_images:
            file_name = doc_info.get('file_name')
            images = doc_info.get('images', [])
            
            if not file_name or not images:
                continue
                
            for image_name in images:
                image_data = self.get_image(file_name, image_name)
                if image_data:
                    result_images.append({
                        'file_name': file_name,
                        'image_name': image_name,
                        'image_data': image_data
                    })
        
        logger.info(f"Получено {len(result_images)} изображений из MinIO")
        return result_images
    
    def delete_folder(self, file_name: str):
        """
        Удаляет папку с файлами.
        
        Args:
            file_name: Имя исходного файла
        """
        try:
            folder_name = self._get_folder_name(file_name)
            
            # Получаем все объекты в папке
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=f"{folder_name}/",
                recursive=True
            )
            
            # Удаляем все объекты
            for obj in objects:
                self.client.remove_object(self.bucket_name, obj.object_name)
            
            logger.info(f"Папка удалена: {folder_name}")
            
        except S3Error as e:
            logger.error(f"Ошибка при удалении папки: {e}")
            raise
    
    def _get_folder_name(self, file_name: str) -> str:
        """
        Генерирует имя папки на основе имени файла.
        
        Args:
            file_name: Имя файла
            
        Returns:
            Имя папки
        """
        # Убираем расширение и заменяем точки на подчеркивания
        base_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
        return base_name.replace('.', '_').replace(' ', '_')


# Глобальный экземпляр для работы с MinIO
minio_worker = MinioWorker() 