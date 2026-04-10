from __future__ import annotations

import io
from dataclasses import dataclass

import pdfplumber

from app.utils.text import sanitize_text


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class DocumentText:
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def page_marked_text(self) -> str:
        return "\n\n".join(
            f"[Page {page.page}]\n{page.text}" for page in self.pages if page.text
        )


class PDFIngestionService:
    def extract_text(self, pdf_bytes: bytes) -> DocumentText:
        pages: list[PageText] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = sanitize_text(text)
                pages.append(PageText(page=i + 1, text=text))

        if all(not p.text for p in pages):
            ocr_pages = self._ocr_fallback(pdf_bytes)
            if ocr_pages:
                pages = ocr_pages

        return DocumentText(pages=pages)

    def _ocr_fallback(self, pdf_bytes: bytes) -> list[PageText]:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
        except Exception:
            return []

        images = convert_from_bytes(pdf_bytes, dpi=200)
        pages: list[PageText] = []
        for i, image in enumerate(images):
            text = sanitize_text(pytesseract.image_to_string(image))
            pages.append(PageText(page=i + 1, text=text))
        return pages
