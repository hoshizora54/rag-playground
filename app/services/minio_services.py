"""Дополнительный сервис для работы с MinIO."""

from minio import Minio
import io
from app.core.config import config


class MinioService:
    """Дополнительный сервис для работы с MinIO: загрузка, скачивание и управление файлами."""
    
    def __init__(self):
        self.client = Minio(
            endpoint=config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE
        )
        if not self.client.bucket_exists(config.MINIO_BUCKET):
            self.client.make_bucket(config.MINIO_BUCKET)
        self.bucket_name = config.MINIO_BUCKET

    def upload_file(self, file_data: bytes, file_name: str, content_type: str = "application/octet-stream") -> str:
        """Загружает файл в MinIO."""
        file_path = f"instructions/rag/{file_name}"
        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=file_path,
            data=io.BytesIO(file_data),
            length=len(file_data),
            content_type=content_type
        )
        return file_path

    def download_file(self, file_name: str, local_path: str) -> str:
        """Скачивает файл из MinIO."""
        self.client.fget_object(
            bucket_name=self.bucket_name,
            object_name=file_name,
            file_path=local_path
        )
        return local_path

    def list_files(self, prefix: str = "") -> list:
        """Возвращает список файлов в MinIO."""
        objects = self.client.list_objects(
            bucket_name=self.bucket_name,
            prefix=prefix,
            recursive=True
        )
        return [obj.object_name for obj in objects]
    
    def get_image(self, file_name: str, image_name: str) -> bytes:
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
            
        except Exception as e:
            return None
    
    def get_multiple_images(self, document_images: list) -> list:
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
        
        return result_images
    
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


# Глобальный экземпляр
minio_worker = MinioService()