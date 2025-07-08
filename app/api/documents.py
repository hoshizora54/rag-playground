"""API роутер для работы с документами."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List

from app.services.indexing_service import indexing_service
from app.schemas.document import DocumentUploadResponse, IndexListResponse, OpenSearchInfoResponse
from app.services.opensearch_service import opensearch_service
from app.core.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    use_semantic_chunking: bool = Form(False),
    semantic_threshold: float = Form(0.8)
):
    """
    Загружает и индексирует PDF документ.
    
    Args:
        file: PDF файл для загрузки
        index_name: Имя индекса для сохранения документа
        use_semantic_chunking: Использовать семантическое разбиение на чанки
        semantic_threshold: Порог косинусного сходства (0.1-1.0)
        
    Returns:
        Информация о результате загрузки
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    logger.info(f"Загрузка документа: {file.filename} в индекс: {index_name}")
    logger.info(f"Семантическое разбиение: {use_semantic_chunking}, порог: {semantic_threshold}")
    
    result = await indexing_service.process_document(
        file, 
        index_name, 
        use_semantic_chunking=use_semantic_chunking,
        semantic_threshold=semantic_threshold
    )
    return result


@router.delete("/delete/{document_id}")
async def delete_document(
    document_id: str,
    index_name: str,
    file_name: str
):
    """
    Удаляет документ из индекса и хранилища.
    
    Args:
        document_id: ID документа
        index_name: Имя индекса
        file_name: Имя файла
    """
    logger.info(f"Удаление документа: {document_id}")
    
    await indexing_service.delete_document(document_id, index_name, file_name)
    return {"message": "Документ успешно удален", "document_id": document_id}


@router.get("/indices", response_model=IndexListResponse)
async def get_indices():
    """
    Получает список всех доступных индексов.
    
    Returns:
        Список индексов
    """
    indices = opensearch_service.get_indices_list()
    return IndexListResponse(indices=indices)


@router.get("/opensearch/info", response_model=OpenSearchInfoResponse)
async def get_opensearch_info():
    """
    Получает информацию о состоянии OpenSearch.
    
    Returns:
        Информация об OpenSearch кластере
    """
    info = opensearch_service.get_cluster_info()
    return OpenSearchInfoResponse(opensearch_info=info)


@router.post("/indices/{index_name}")
async def create_index(index_name: str):
    """
    Создает новый индекс.
    
    Args:
        index_name: Имя нового индекса
    """
    logger.info(f"Создание индекса: {index_name}")
    
    success = opensearch_service.create_new_index(index_name)
    if success:
        return {"message": f"Индекс {index_name} успешно создан"}
    else:
        raise HTTPException(status_code=500, detail=f"Ошибка при создании индекса {index_name}") 