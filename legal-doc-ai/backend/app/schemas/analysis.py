from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    page: int | None = None


class ExtractedEntity(BaseModel):
    label: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceSpan


class CoreExtraction(BaseModel):
    case_name: ExtractedEntity | None = None
    parties: list[ExtractedEntity] = Field(default_factory=list)
    judges: list[ExtractedEntity] = Field(default_factory=list)
    court_names: list[ExtractedEntity] = Field(default_factory=list)
    important_dates: list[ExtractedEntity] = Field(default_factory=list)
    legal_sections_cited: list[ExtractedEntity] = Field(default_factory=list)
    punishment_sentence: list[ExtractedEntity] = Field(default_factory=list)
    final_order: ExtractedEntity | None = None


class KeyPoint(BaseModel):
    label: str
    sentence: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: SourceSpan


class RetrievalHit(BaseModel):
    doc_id: str
    score: float
    snippet: str


class AnalysisResult(BaseModel):
    summary_extractive: str
    summary_abstractive: str
    key_points: list[KeyPoint]
    next_steps: list[str]
    extraction: CoreExtraction
    retrieval_context: list[RetrievalHit]
    disclaimer: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result: AnalysisResult | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
    status_url: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatSource(BaseModel):
    page: int | None = None
    snippet: str


class ChatAnswer(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    device: str
