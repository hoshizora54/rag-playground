"""Схемы для работы с документами."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from .base import BaseResponse


class DocumentChunk(BaseModel):
    """Схема чанка документа."""
    id: str
    text: str
    vector: List[float]
    page_number: int
    paragraph_index: int
    images: Union[List[str], List[Dict[str, Any]]] = []  # Имена файлов или объекты изображений


class DocumentUploadRequest(BaseModel):
    """Запрос на загрузку документа."""
    file_name: str
    index_name: str


class DocumentUploadResponse(BaseResponse):
    """Ответ на загрузку документа."""
    document_id: str
    chunks_count: int
    minio_path: str


class IndexListResponse(BaseModel):
    """Схема для списка индексов."""
    indices: List[str]


class OpenSearchInfoResponse(BaseModel):
    """Схема информации об OpenSearch."""
    opensearch_info: Dict[str, Any]


class FolderPathRequest(BaseModel):
    """Запрос пути к папке."""
    folder_path: str
    index: str 