from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .base import BaseResponse

class UserQuery(BaseModel):
    """Запрос пользователя для поиска."""
    model_config = {"arbitrary_types_allowed": True}
    
    query_text: str = Field(..., description="Текст запроса пользователя")
    index_name: str = Field(..., description="Название индекса для поиска")
    query_embed: Optional[List[float]] = Field(default=None, description="Векторное представление запроса")
    k: int = Field(default=5, description="Количество ближайших векторов")
    size: int = Field(default=10, description="Количество результатов")
    sematic: float = Field(default=0.7, description="Вес семантического поиска")
    keyword: float = Field(default=0.3, description="Вес ключевого поиска")
    fields: List[str] = Field(default=["text"], description="Поля для поиска")
    reranking: bool = Field(default=False, description="Использовать ColBERT реранжирование")
    trashold: float = Field(default=0.5, description="Порог фильтрации")
    trashold_bertscore: float = Field(default=0.8, description="Порог BERTScore")
    
    # Продвинутые технологии
    use_hyde: bool = Field(default=False, description="Использовать HyDE (Hypothetical Document Embeddings)")
    hyde_num_hypotheses: int = Field(default=3, description="Количество гипотез для HyDE")

class SearchResult(BaseModel):
    """Результат поиска."""
    query: str
    results: List[Dict[str, Any]]
    search_time: float
    
class SearchResponse(BaseResponse):
    """Ответ поиска."""
    data: SearchResult


class CheckIndex(BaseModel):
    """Запрос на проверку файла в индексе."""
    source: str
    index_name: str


class IndexCheckResponse(BaseModel):
    """Ответ на проверку файла в индексе."""
    status: bool 