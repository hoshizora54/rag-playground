"""Модуль для работы с OpenSearch."""

import requests
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError

from app.core.config import config
from app.core.logger import logger
from app.schemas.document import DocumentChunk


class OpenSearchWorker:
    """Класс для работы с OpenSearch - индексация и поиск документов."""
    
    def __init__(self):
        """Инициализация подключения к OpenSearch."""
        self.client = OpenSearch(
            hosts=[{'host': config.OPENSEARCH_HOST, 'port': config.OPENSEARCH_PORT}],
            http_auth=(config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection
        )
        self.headers = {
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "Content-Type": "application/json"
        }
    
    def create_index(self, index_name: str) -> bool:
        """
        Создает индекс для хранения документов.
        
        Args:
            index_name: Имя индекса
            
        Returns:
            True если индекс создан успешно
        """
        try:
            # Схема индекса с двумя основными полями: text и vector
            index_body = {
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "vector": {
                            "type": "knn_vector",
                            "dimension": config.VECTOR_DIMENSION,  # 256 для Yandex GPT
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        "document_id": {
                            "type": "keyword"
                        },
                        "chunk_id": {
                            "type": "keyword"
                        },
                        "page_number": {
                            "type": "integer"
                        },
                        "paragraph_index": {
                            "type": "integer"
                        },
                        "images": {
                            "type": "keyword"
                        },
                        "file_name": {
                            "type": "keyword"
                        }
                    }
                }
            }
            
            if not self.client.indices.exists(index_name):
                self.client.indices.create(index=index_name, body=index_body)
                logger.info(f"Индекс создан: {index_name}")
                return True
            else:
                logger.info(f"Индекс уже существует: {index_name}")
                return True
                
        except RequestError as e:
            logger.error(f"Ошибка при создании индекса {index_name}: {e}")
            return False
    
    def index_document(self, index_name: str, chunks: List[DocumentChunk], 
                      document_id: str, file_name: str) -> bool:
        """
        Индексирует документ в OpenSearch.
        
        Args:
            index_name: Имя индекса
            chunks: Список чанков документа
            document_id: ID документа
            file_name: Имя файла
            
        Returns:
            True если индексация прошла успешно
        """
        try:
            # Создаем индекс если его нет
            self.create_index(index_name)
            
            for chunk in chunks:
                doc_body = {
                    "text": chunk.text,
                    "vector": chunk.vector,
                    "document_id": document_id,
                    "chunk_id": chunk.id,
                    "page_number": chunk.page_number,
                    "paragraph_index": chunk.paragraph_index,
                    "images": chunk.images,
                    "file_name": file_name
                }
                
                self.client.index(
                    index=index_name,
                    id=chunk.id,
                    body=doc_body
                )
            
            # Обновляем индекс
            self.client.indices.refresh(index=index_name)
            
            logger.info(f"Документ проиндексирован: {document_id}, чанков: {len(chunks)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации документа: {e}")
            return False
    
    def execute_search(self, query_id: int, query_text: str, index: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Выполняет гибридный поиск в OpenSearch.
        
        Args:
            query_id: ID запроса
            query_text: Текст запроса
            index: Имя индекса
            **kwargs: Дополнительные параметры поиска
            
        Returns:
            Список результатов поиска
        """
        try:
            # Получаем вектор запроса если он не предоставлен
            query_embed = kwargs.get("query_embed")
            if not query_embed:
                query_embed = self._create_embedding(query_text)
            
            # Параметры поиска
            size = kwargs.get("size", 10)
            semantic_weight = kwargs.get("sematic", 0.7)  # Вес семантического поиска
            keyword_weight = kwargs.get("keyword", 0.3)   # Вес ключевого поиска
            fields = kwargs.get("fields", ["text"])       # Поля для текстового поиска
            
            # Логируем информацию об embedding для диагностики
            if query_embed:
                logger.info(f"Используется embedding: размерность={len(query_embed)}, первые 3 значения={query_embed[:3] if len(query_embed) >= 3 else query_embed}")
            else:
                logger.warning("Embedding не предоставлен для поиска!")
            
            # Гибридный поиск с векторным и текстовым компонентами
            search_body = {
                "size": size,
                "query": {
                    "function_score": {
                        "query": {
                            "bool": {
                                "should": [
                                    # Векторный поиск
                                    {
                                        "function_score": {
                                            "query": {
                                                "knn": {
                                                    "vector": {
                                                        "vector": query_embed,
                                                        "k": size
                                                    }
                                                }
                                            },
                                            "weight": semantic_weight
                                        }
                                    },
                                    # Текстовый поиск
                                    {
                                        "function_score": {
                                            "query": {
                                                "multi_match": {
                                                    "query": query_text,
                                                    "fields": fields,
                                                    "type": "most_fields"
                                                }
                                            },
                                            "weight": keyword_weight
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
                "_source": ["text", "chunk_id", "page_number", "paragraph_index", 
                           "images", "file_name", "document_id"]
            }
            
            response = self.client.search(index=index, body=search_body)
            
            # Обрабатываем результаты
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "score": hit['_score'],
                    "text": hit['_source']['text'],
                    "chunk_id": hit['_source']['chunk_id'],
                    "page_number": hit['_source']['page_number'],
                    "paragraph_index": hit['_source']['paragraph_index'],
                    "images": hit['_source'].get('images', []),
                    "file_name": hit['_source']['file_name'],
                    "document_id": hit['_source']['document_id']
                }
                results.append(result)
            
            logger.info(f"Поиск выполнен: найдено {len(results)} результатов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}")
            return []
    
    def check_file_in_index(self, file_name: str, index_name: str) -> bool:
        """
        Проверяет наличие файла в индексе.
        
        Args:
            file_name: Имя файла
            index_name: Имя индекса
            
        Returns:
            True если файл найден в индексе
        """
        try:
            search_body = {
                "query": {
                    "term": {
                        "file_name": file_name
                    }
                },
                "size": 1
            }
            
            response = self.client.search(index=index_name, body=search_body)
            return response['hits']['total']['value'] > 0
            
        except Exception as e:
            logger.error(f"Ошибка при проверке файла в индексе: {e}")
            return False
    
    def delete_document(self, index_name: str, document_id: str) -> bool:
        """
        Удаляет документ из индекса.
        
        Args:
            index_name: Имя индекса
            document_id: ID документа
            
        Returns:
            True если удаление прошло успешно
        """
        try:
            delete_body = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                }
            }
            
            response = self.client.delete_by_query(index=index_name, body=delete_body)
            self.client.indices.refresh(index=index_name)
            
            logger.info(f"Документ удален: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}")
            return False
    
    def list_indices(self) -> List[str]:
        """
        Получает список всех индексов.
        
        Returns:
            Список имен индексов
        """
        try:
            indices = self.client.indices.get_alias("*")
            return list(indices.keys())
            
        except Exception as e:
            logger.error(f"Ошибка при получении списка индексов: {e}")
            return []
    
    def get_opensearch_info(self) -> Dict[str, Any]:
        """
        Получает информацию о состоянии OpenSearch.
        
        Returns:
            Информация о кластере
        """
        try:
            info = self.client.info()
            cluster_health = self.client.cluster.health()
            
            return {
                "cluster_name": info.get("cluster_name"),
                "version": info.get("version", {}).get("number"),
                "status": cluster_health.get("status"),
                "number_of_nodes": cluster_health.get("number_of_nodes"),
                "number_of_data_nodes": cluster_health.get("number_of_data_nodes")
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении информации об OpenSearch: {e}")
            return {}
    
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


# Глобальный экземпляр для работы с OpenSearch
opensearch_worker = OpenSearchWorker()
