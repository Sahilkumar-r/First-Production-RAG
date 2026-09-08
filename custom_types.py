import pydantic
from typing import List, Dict, Any, Optional

class RAGChunkAndSrc(pydantic.BaseModel):
    # Now accepts a list of rich dicts containing enhanced_content and original_content metadata
    chunks: List[Dict[str, Any]]
    source_id: Optional[str] = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: List[Any]
    sources: List[str]

class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int
