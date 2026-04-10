from __future__ import annotations

from app.config import settings
from app.schemas.analysis import AnalysisResult
from app.services.entity_extractor import EntityExtractionService
from app.services.document_chat import ChatDocumentContext, DocumentChunk
from app.services.groq_summary import groq_summary
from app.services.keypoint_extractor import KeyPointExtractionService
from app.services.next_steps import NextStepsService
from app.services.pdf_ingestion import PDFIngestionService
from app.services.rag_service import RAGService
from app.services.segmentation import RhetoricalSegmentationService
from app.services.summarizer import SummarizationService
from app.utils.text import remove_boilerplate


class AnalysisPipeline:
    def __init__(self) -> None:
        self.ingestion = PDFIngestionService()
        self.segmentation = RhetoricalSegmentationService()
        self.entity_extractor = EntityExtractionService()
        self.key_points = KeyPointExtractionService()
        self.summarizer = SummarizationService()
        self.rag = RAGService()
        self.next_steps = NextStepsService()

    def run(self, pdf_bytes: bytes) -> AnalysisResult:
        result, _ = self.run_with_context(pdf_bytes)
        return result

    def run_with_context(self, pdf_bytes: bytes) -> tuple[AnalysisResult, ChatDocumentContext]:
        doc = self.ingestion.extract_text(pdf_bytes)
        text = doc.full_text
        page_marked_text = doc.page_marked_text or text
        filtered_text = remove_boilerplate(text)
        chat_context = ChatDocumentContext(
            full_text=text,
            clean_text=filtered_text,
            page_marked_text=page_marked_text,
            page_texts=doc.pages,
            chunks=self._build_chunks(doc.pages),
        )

        extraction = self.entity_extractor.extract(filtered_text)
        key_points = self.key_points.extract(filtered_text)
        segments = self.segmentation.segment(filtered_text)
        local_extractive = self.summarizer.summarize_extractive(filtered_text)

        retrieval_hits = self.rag.search(local_extractive or filtered_text[:1500])
        retrieval_context = "\n\n".join(hit.snippet for hit in retrieval_hits)

        summary_abstractive_input = filtered_text
        if retrieval_context:
            summary_abstractive_input = f"{filtered_text[:4000]}\n\nRelated precedents:\n{retrieval_context}"

        groq_extractive, groq_abstractive = groq_summary.summarize_pair(
            text=page_marked_text,
            local_summary=local_extractive,
            extraction=extraction,
            key_points=key_points,
            segments=segments,
            retrieval_hits=retrieval_hits,
        )

        summary_extractive = groq_extractive or local_extractive
        summary_abstractive = groq_abstractive or self.summarizer.summarize_abstractive(summary_abstractive_input)
        suggestions = self.next_steps.suggest(filtered_text, extraction, retrieval_hits)

        result = AnalysisResult(
            summary_extractive=summary_extractive,
            summary_abstractive=summary_abstractive,
            key_points=key_points,
            next_steps=suggestions,
            extraction=extraction,
            retrieval_context=retrieval_hits,
            disclaimer=settings.legal_disclaimer,
        )

        return result, chat_context

    def _build_chunks(self, pages: list[object], chunk_size: int = 1200, overlap: int = 180) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for page in pages:
            text = getattr(page, "text", "").strip()
            page_number = getattr(page, "page", None)
            if not text:
                continue
            if len(text) <= chunk_size:
                chunks.append(DocumentChunk(page=page_number, text=text))
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(DocumentChunk(page=page_number, text=chunk))
                if end >= len(text):
                    break
                start = max(end - overlap, start + 1)
        return chunks


analysis_pipeline = AnalysisPipeline()
