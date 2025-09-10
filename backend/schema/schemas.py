from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    similarity_score: float

class RetrievalOutput(BaseModel):
    chunks: list[RetrievedChunk]


class DraftAnswer(BaseModel):
    answer: str
    cited_sources: list[str]  # list of document_ids or chunk_ids


class VerifiedSource(BaseModel):
    document_id: str
    chunk_id: str | None = None
    excerpt: str

class AttributionOutput(BaseModel):
    verified_answer: str
    sources: list[VerifiedSource]
    unsupported: list[str] = []  # statements not backed by any chunk


class SummaryOutput(BaseModel):
    summary_type: str  # bullet_list | short_summary | executive_summary
    summary: str



class SourceItem(BaseModel):
    doc_id: str
    snippet: str
    page: int | None = None
    confidence: float | None = None

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    metadata: dict


class IntentOutput(BaseModel):
    intent: Literal["qa", "summarize", "lookup", "compare"] = Field(
        ..., description="High-level intent of the user query"
    )
    entities: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of parsed entities (e.g., document IDs, doc titles, compare targets)"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Model confidence (0.0-1.0) if provided; otherwise null"
    )






# class AnsweringInput(BaseModel):
#     query: str
#     chunks: List[RetrievedChunk]
