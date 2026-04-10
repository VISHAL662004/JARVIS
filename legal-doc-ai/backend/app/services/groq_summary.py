from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.config import settings
from app.schemas.analysis import CoreExtraction, KeyPoint, RetrievalHit
from app.services.segmentation import Segment


class GroqSummaryService:
    def summarize_pair(
        self,
        *,
        text: str,
        local_summary: str,
        extraction: CoreExtraction,
        key_points: list[KeyPoint],
        segments: list[Segment],
        retrieval_hits: list[RetrievalHit],
    ) -> tuple[str, str]:
        if not settings.groq_api_key:
            return "", ""

        prompt = self._build_prompt(
            text=text,
            local_summary=local_summary,
            extraction=extraction,
            key_points=key_points,
            segments=segments,
            retrieval_hits=retrieval_hits,
        )

        payload = {
            "model": settings.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write polished legal document summaries. Return only valid JSON with "
                        "keys summary_extractive and summary_abstractive. Each value must be plain text, "
                        "self-contained, and free of markdown, bullets, code fences, or disclaimers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1100,
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
            return (
                self._normalize_text(parsed.get("summary_extractive", "")),
                self._normalize_text(parsed.get("summary_abstractive", "")),
            )
        except Exception:
            return "", ""

    def _build_prompt(
        self,
        *,
        text: str,
        local_summary: str,
        extraction: CoreExtraction,
        key_points: list[KeyPoint],
        segments: list[Segment],
        retrieval_hits: list[RetrievalHit],
    ) -> str:
        return "\n\n".join(
            [
                "Task: rewrite the case into two high-quality summaries for a legal analysis app.",
                "Requirements:",
                "- Return valid JSON only.",
                "- summary_extractive: 220-320 words, compact but complete, in plain English.",
                "- summary_abstractive: 500-800 words, fully developed, standard legal-summary style, and more detailed than the extractive version.",
                "- Use multiple paragraphs in summary_abstractive if helpful.",
                "- Mention the case name, court, parties, bench, procedural history, main facts, legal sections, issues, reasoning, and final order when available.",
                "- Do not copy sentences verbatim unless they are short and essential; rewrite in polished prose.",
                "- Use the full document text below as the primary source, with extracted metadata as support.",
                f"Local summary seed: {local_summary or 'N/A'}",
                f"Case name: {self._entity_value(extraction.case_name)}",
                f"Parties: {self._entity_values(extraction.parties)}",
                f"Judges: {self._entity_values(extraction.judges)}",
                f"Courts: {self._entity_values(extraction.court_names)}",
                f"Important dates: {self._entity_values(extraction.important_dates)}",
                f"Sections cited: {self._entity_values(extraction.legal_sections_cited)}",
                f"Punishment sentence: {self._entity_values(extraction.punishment_sentence)}",
                f"Final order: {self._entity_value(extraction.final_order)}",
                f"Key points: {self._format_key_points(key_points)}",
                f"Text segments: {self._format_segments(segments)}",
                f"Relevant precedents: {self._format_retrieval_hits(retrieval_hits)}",
                f"Full document text:\n{text}",
            ]
        )

    def _format_key_points(self, key_points: list[KeyPoint], limit: int = 10) -> str:
        if not key_points:
            return "N/A"
        items = [f"{kp.label}: {kp.sentence}" for kp in key_points[:limit]]
        return " | ".join(items)

    def _format_segments(self, segments: list[Segment], limit: int = 18) -> str:
        if not segments:
            return "N/A"
        items: list[str] = []
        total_chars = 0
        for segment in segments[:limit]:
            piece = f"{segment.label}: {segment.text.strip()}"
            if total_chars + len(piece) > 2500:
                break
            items.append(piece)
            total_chars += len(piece)
        return " | ".join(items) if items else "N/A"

    def _format_retrieval_hits(self, retrieval_hits: list[RetrievalHit], limit: int = 3) -> str:
        if not retrieval_hits:
            return "N/A"
        items = [f"{hit.doc_id} ({hit.score:.3f}): {hit.snippet}" for hit in retrieval_hits[:limit]]
        return " | ".join(items)

    def _entity_value(self, entity: Any) -> str:
        if entity is None:
            return "N/A"
        value = getattr(entity, "value", "")
        return value or "N/A"

    def _entity_values(self, entities: list[Any]) -> str:
        if not entities:
            return "N/A"
        values = [getattr(entity, "value", "").strip() for entity in entities if getattr(entity, "value", "").strip()]
        return "; ".join(values) if values else "N/A"

    def _parse_json_content(self, content: str) -> dict[str, str]:
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
        return {
            "summary_extractive": str(data.get("summary_extractive", "")),
            "summary_abstractive": str(data.get("summary_abstractive", "")),
        }

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()


groq_summary = GroqSummaryService()