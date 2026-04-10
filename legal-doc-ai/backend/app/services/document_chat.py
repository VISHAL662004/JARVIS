from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from app.config import settings
from app.schemas.analysis import AnalysisResult, ChatAnswer, ChatSource, CoreExtraction
from app.services.pdf_ingestion import PageText, PDFIngestionService
from app.utils.text import remove_boilerplate


@dataclass
class DocumentChunk:
    page: int | None
    text: str


@dataclass
class ChatDocumentContext:
    full_text: str
    clean_text: str
    page_marked_text: str
    page_texts: list[PageText]
    chunks: list[DocumentChunk]


class DocumentChatStore:
    def __init__(self) -> None:
        self._contexts: dict[str, ChatDocumentContext] = {}
        self._lock = asyncio.Lock()

    async def register(self, job_id: str, context: ChatDocumentContext) -> None:
        async with self._lock:
            self._contexts[job_id] = context

    async def get(self, job_id: str) -> ChatDocumentContext | None:
        async with self._lock:
            return self._contexts.get(job_id)


class DocumentChatService:
    def __init__(self) -> None:
        self.ingestion = PDFIngestionService()

    def build_context(self, pdf_bytes: bytes) -> ChatDocumentContext:
        doc = self.ingestion.extract_text(pdf_bytes)
        full_text = doc.full_text
        clean_text = remove_boilerplate(full_text)
        return ChatDocumentContext(
            full_text=full_text,
            clean_text=clean_text,
            page_marked_text=doc.page_marked_text or full_text,
            page_texts=doc.pages,
            chunks=self._build_chunks(doc.pages),
        )

    def answer(
        self,
        *,
        context: ChatDocumentContext,
        question: str,
        analysis: AnalysisResult | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ChatAnswer:
        ranked_chunks = self._rank_chunks(question, context.chunks)
        relevant_sources = [ChatSource(page=chunk.page, snippet=chunk.text[:280].strip()) for chunk in ranked_chunks[:3]]
        selected_context = self._compose_context(context, ranked_chunks)
        conversation = self._compose_history(history or [])

        if not settings.groq_api_key:
            return ChatAnswer(
                answer=self._fallback_answer(question, ranked_chunks, analysis),
                sources=relevant_sources,
                disclaimer=settings.legal_disclaimer,
            )

        prompt = self._build_prompt(
            question=question,
            context=selected_context,
            analysis=analysis,
            conversation=conversation,
            sources=relevant_sources,
        )

        payload = {
            "model": settings.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a legal document question-answering assistant. Answer only from the provided "
                        "document context and analysis. If the document does not contain enough information, say so clearly. "
                        "Return only valid JSON with keys answer and sources. answer must be plain text and concise. "
                        "sources must be an array of objects with page and snippet. Do not mention policy or chain of thought."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                settings.groq_api_url,
                headers=headers,
                json=payload,
                timeout=settings.groq_timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_json_content(content)
            answer = self._normalize_text(str(parsed.get("answer", "")))
            parsed_sources = self._parse_sources(parsed.get("sources", []))
            if parsed_sources:
                return ChatAnswer(answer=answer, sources=parsed_sources, disclaimer=settings.legal_disclaimer)
            return ChatAnswer(answer=answer, sources=relevant_sources, disclaimer=settings.legal_disclaimer)
        except Exception:
            return ChatAnswer(
                answer=self._fallback_answer(question, ranked_chunks, analysis),
                sources=relevant_sources,
                disclaimer=settings.legal_disclaimer,
            )

    def _build_chunks(self, pages: list[PageText], chunk_size: int = 1200, overlap: int = 180) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            if len(text) <= chunk_size:
                chunks.append(DocumentChunk(page=page.page, text=text))
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(DocumentChunk(page=page.page, text=chunk))
                if end >= len(text):
                    break
                start = max(end - overlap, start + 1)
        return chunks

    def _rank_chunks(self, question: str, chunks: list[DocumentChunk], limit: int = 4) -> list[DocumentChunk]:
        if not chunks:
            return []

        query_terms = self._tokenize(question)
        if not query_terms:
            return chunks[:limit]

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            chunk_terms = self._tokenize(chunk.text)
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            density = overlap / max(len(chunk_terms), 1)
            scored.append((overlap + density, chunk))

        if not scored:
            return chunks[:limit]

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    def _compose_context(self, context: ChatDocumentContext, chunks: list[DocumentChunk]) -> str:
        selected = "\n\n".join(f"[Page {chunk.page or 'Unknown'}]\n{chunk.text}" for chunk in chunks)
        if not selected:
            selected = context.page_marked_text
        return self._truncate(selected, settings.groq_context_chars)

    def _compose_history(self, history: list[dict[str, str]]) -> str:
        if not history:
            return "N/A"
        lines: list[str] = []
        for turn in history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = self._normalize_text(turn.get("content", ""))
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "N/A"

    def _build_prompt(
        self,
        *,
        question: str,
        context: str,
        analysis: AnalysisResult | None,
        conversation: str,
        sources: list[ChatSource],
    ) -> str:
        analysis_text = self._format_analysis(analysis)
        source_text = "\n".join(f"- Page {source.page or 'Unknown'}: {source.snippet}" for source in sources) or "N/A"
        return "\n\n".join(
            [
                "Task: answer the user's question using the uploaded legal PDF.",
                "Rules:",
                "- Use only the provided document context and analysis.",
                "- If the answer is not supported by the PDF, say that the document does not provide enough information.",
                "- Keep the answer direct and grounded in the text.",
                "- Mention page numbers when helpful.",
                "- Return valid JSON only.",
                f"Conversation so far:\n{conversation}",
                f"Analysis context:\n{analysis_text}",
                f"Relevant excerpts:\n{source_text}",
                f"Document context:\n{context}",
                f"Question: {question}",
            ]
        )

    def _format_analysis(self, analysis: AnalysisResult | None) -> str:
        if analysis is None:
            return "N/A"
        sections = [
            f"Summary: {analysis.summary_abstractive}",
            f"Extractive summary: {analysis.summary_extractive}",
            f"Key points: {self._join_key_points(analysis)}",
            f"Next steps: {'; '.join(analysis.next_steps) if analysis.next_steps else 'N/A'}",
            f"Entities: {self._format_entities(analysis.extraction)}",
        ]
        return self._truncate("\n".join(sections), 5000)

    def _join_key_points(self, analysis: AnalysisResult) -> str:
        if not analysis.key_points:
            return "N/A"
        return " | ".join(f"{item.label}: {item.sentence}" for item in analysis.key_points[:8])

    def _format_entities(self, extraction: CoreExtraction) -> str:
        parts: list[str] = []
        if extraction.case_name:
            parts.append(f"Case name: {extraction.case_name.value}")
        if extraction.parties:
            parts.append(f"Parties: {self._entity_values(extraction.parties)}")
        if extraction.judges:
            parts.append(f"Judges: {self._entity_values(extraction.judges)}")
        if extraction.court_names:
            parts.append(f"Courts: {self._entity_values(extraction.court_names)}")
        if extraction.legal_sections_cited:
            parts.append(f"Sections cited: {self._entity_values(extraction.legal_sections_cited)}")
        if extraction.final_order:
            parts.append(f"Final order: {extraction.final_order.value}")
        return " | ".join(parts) if parts else "N/A"

    def _fallback_answer(
        self,
        question: str,
        chunks: list[DocumentChunk],
        analysis: AnalysisResult | None,
    ) -> str:
        if chunks:
            excerpt = chunks[0].text[:700].strip()
            page = chunks[0].page or "Unknown"
            return f"I could not reach Groq, but the most relevant part I found is on page {page}: {excerpt}"
        if analysis:
            return (
                "I could not reach Groq. Based on the existing analysis, the strongest available answer is: "
                f"{analysis.summary_abstractive[:800].strip()}"
            )
        return f"I could not find enough text in the PDF to answer: {question}"

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}
            data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}

    def _parse_sources(self, raw_sources: Any) -> list[ChatSource]:
        if not isinstance(raw_sources, list):
            return []
        sources: list[ChatSource] = []
        for item in raw_sources[:3]:
            if not isinstance(item, dict):
                continue
            snippet = self._normalize_text(str(item.get("snippet", "")))
            if not snippet:
                continue
            sources.append(ChatSource(page=self._safe_int(item.get("page")), snippet=snippet))
        return sources

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _entity_values(self, entities: list[Any]) -> str:
        values = [getattr(entity, "value", "").strip() for entity in entities if getattr(entity, "value", "").strip()]
        return "; ".join(values) if values else "N/A"

    def _safe_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None


document_chat_store = DocumentChatStore()
document_chat_service = DocumentChatService()