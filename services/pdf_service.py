import io
from typing import List

from pypdf import PdfReader


class PdfService:
    """Service responsible for extracting textual content from PDF files."""

    def extract_text(self, file_bytes: bytes) -> str:
        """Return the concatenated text of all pages in the given PDF bytes."""
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts: List[str] = []
        for page in reader.pages:
            # PdfReader may return None if a page is empty
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        result = "\n".join(text_parts)
        # if PdfReader had trouble (e.g. scanned/image PDF) the result might
        # consist solely of whitespace/newlines; callers should check.
        return result
