"""
Docling-based Document Parser
Uses Docling Serve HTTP API for structured extraction, with pdfplumber fallback.

Docling Serve provides:
- Layout-aware text extraction (reading order preserved)
- Table structure recognition
- Heading hierarchy detection
- Figure/caption extraction
- Section tree construction
"""
import logging
import io
import hashlib
import json
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DOCLING_SERVE_URL = "http://localhost:5001"


@dataclass
class ParsedElement:
    """A structured element from a document."""
    text: str
    element_type: str  # paragraph, heading, table, list, figure, formula
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_number: Optional[int] = None
    section_path: str = ""
    confidence: float = 1.0


@dataclass
class ParsedDocument:
    """Fully parsed document with structured elements."""
    filename: str
    title: str
    elements: List[ParsedElement]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_hash: str = ""


class DoclingParser:
    """
    Document parser using Docling Serve HTTP API for structured extraction.
    Falls back to pdfplumber/python-pptx/python-docx if the service is unavailable.
    """

    def __init__(self, serve_url: str = DOCLING_SERVE_URL):
        self.serve_url = serve_url.rstrip("/")
        self._docling_available = False
        self._http_client = httpx.Client(timeout=120.0)
        self._fallback_parser = FallbackParser()

        # Check if Docling Serve is reachable
        try:
            resp = self._http_client.get(f"{self.serve_url}/health", timeout=3.0)
            if resp.status_code == 200:
                self._docling_available = True
                logger.info(f"Docling Serve available at {self.serve_url}")
            else:
                logger.warning(f"Docling Serve returned {resp.status_code}, using fallback")
        except Exception as e:
            logger.warning(f"Docling Serve not reachable ({e}), using pdfplumber fallback")

    def parse_file(self, filepath: Path) -> Optional[ParsedDocument]:
        """Parse a document file into structured elements."""
        file_hash = self._compute_hash(filepath)
        ext = filepath.suffix.lower()

        # Try Docling Serve first for PDFs
        if ext == '.pdf' and self._docling_available:
            try:
                return self._serve_parse_pdf(filepath, file_hash)
            except Exception as e:
                logger.warning(f"Docling Serve PDF parse failed: {e}, falling back")

        # Fallback parser for all formats
        return self._fallback_parser.parse_file(filepath, file_hash)

    def _serve_parse_pdf(self, filepath: Path, file_hash: str) -> ParsedDocument:
        """Parse PDF via Docling Serve HTTP API."""
        with open(filepath, "rb") as f:
            files = {"files": (filepath.name, f, "application/pdf")}
            resp = self._http_client.post(
                f"{self.serve_url}/v1alpha/convert/file",
                files=files,
            )
        resp.raise_for_status()
        result = resp.json()

        return self._convert_serve_result(result, filepath, file_hash)

    def _convert_serve_result(self, result: dict, filepath: Path, file_hash: str) -> ParsedDocument:
        """Convert Docling Serve JSON response to ParsedDocument."""
        elements = []
        full_text_parts = []

        # Docling Serve returns {"document": {"texts": [...], "tables": [...], ...}}
        doc_data = result.get("document", result)

        # Extract texts
        texts = doc_data.get("texts", [])
        for item in texts:
            text = item.get("text", "")
            if not text.strip():
                continue
            label = item.get("label", "paragraph").lower()
            elem_type = self._map_label(label)
            prov = item.get("prov", [{}])
            page = prov[0].get("page_no") if prov else None

            elements.append(ParsedElement(
                text=text, element_type=elem_type,
                metadata={
                    "source": filepath.name, "file_type": "pdf",
                    "docling_label": label, "page": page,
                },
                page_number=page,
            ))
            full_text_parts.append(text)

        # Extract tables
        tables = doc_data.get("tables", [])
        for t_item in tables:
            table_text = self._format_serve_table(t_item)
            if table_text.strip():
                prov = t_item.get("prov", [{}])
                page = prov[0].get("page_no") if prov else None
                elements.append(ParsedElement(
                    text=table_text, element_type="table",
                    metadata={"source": filepath.name, "file_type": "pdf", "page": page},
                    page_number=page,
                ))
                full_text_parts.append(table_text)

        full_text = "\n\n".join(full_text_parts)
        title = filepath.stem.replace('_', ' ').replace('-', ' ').title()

        return ParsedDocument(
            filename=filepath.name, title=title, elements=elements, full_text=full_text,
            file_hash=file_hash,
            metadata={"source": filepath.name, "file_type": "pdf",
                      "total_elements": len(elements), "parser": "docling-serve"}
        )

    def _map_label(self, label: str) -> str:
        """Map Docling label to our element type."""
        mapping = {
            "heading": "heading", "title": "heading", "section_header": "heading",
            "table": "table", "table_header": "heading",
            "list_item": "list", "ordered_list": "list", "unordered_list": "list",
            "figure": "figure", "image": "figure", "caption": "figure",
            "formula": "formula", "code": "code",
            "footnote": "paragraph", "page_footer": "paragraph",
        }
        return mapping.get(label, "paragraph")

    def _format_serve_table(self, table_data: dict) -> str:
        """Format a Docling Serve table into readable text."""
        rows = []
        grid = table_data.get("grid", [])
        for row in grid:
            row_text = " | ".join(str(cell.get("text", "")) if cell else "" for cell in row)
            if row_text.strip():
                rows.append(row_text)
        return "\n".join(rows)

    def _compute_hash(self, filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read, b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def health_check(self) -> dict:
        """Check Docling Serve health."""
        try:
            resp = self._http_client.get(f"{self.serve_url}/health", timeout=3.0)
            return {"available": True, "status_code": resp.status_code, "url": self.serve_url}
        except Exception as e:
            return {"available": False, "error": str(e), "url": self.serve_url}


class FallbackParser:
    """Fallback parser using pdfplumber/PIL when Docling Serve is unavailable."""

    def parse_file(self, filepath: Path, file_hash: str) -> Optional[ParsedDocument]:
        ext = filepath.suffix.lower()
        if ext == '.pdf':
            return self.parse_pdf(filepath, file_hash)
        elif ext == '.pptx':
            return self.parse_pptx(filepath, file_hash)
        elif ext in ('.docx', '.doc'):
            return self.parse_docx(filepath, file_hash)
        elif ext in ('.png', '.jpg', '.jpeg'):
            return self.parse_image(filepath, file_hash)
        elif ext == '.txt':
            return self.parse_text(filepath, file_hash)
        return None

    def parse_pdf(self, filepath: Path, file_hash: str) -> ParsedDocument:
        import pdfplumber
        elements, full_text_parts = [], []
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(layout=True)
                if text:
                    elements.append(ParsedElement(text=text, element_type="paragraph",
                        metadata={"source": filepath.name, "file_type": "pdf", "page": i + 1},
                        page_number=i + 1))
                    full_text_parts.append(text)
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if table and any(any(c for c in row) for row in table):
                        table_lines = [" | ".join(str(c) if c else "" for c in row) for row in table]
                        table_text = "\n".join(table_lines)
                        elements.append(ParsedElement(text=table_text, element_type="table",
                            metadata={"source": filepath.name, "file_type": "pdf", "page": i + 1},
                            page_number=i + 1))
        return ParsedDocument(filename=filepath.name, title=filepath.stem, elements=elements,
            full_text="\n\n".join(full_text_parts), file_hash=file_hash,
            metadata={"source": filepath.name, "file_type": "pdf", "total_pages": total_pages, "parser": "pdfplumber"})

    def parse_pptx(self, filepath: Path, file_hash: str) -> ParsedDocument:
        from pptx import Presentation
        elements, full_text_parts = [], []
        prs = Presentation(filepath)
        for slide_idx, slide in enumerate(prs.slides):
            parts = [f"=== Slide {slide_idx + 1} ==="]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            parts.append(t)
                if shape.has_table:
                    for row in shape.table.rows:
                        parts.append(" | ".join(c.text.strip() for c in row.cells))
            text = "\n".join(parts)
            if text.strip():
                elements.append(ParsedElement(text=text, element_type="slide",
                    metadata={"source": filepath.name, "file_type": "pptx", "slide": slide_idx + 1}))
                full_text_parts.append(text)
        return ParsedDocument(filename=filepath.name, title=filepath.stem, elements=elements,
            full_text="\n\n".join(full_text_parts), file_hash=file_hash,
            metadata={"source": filepath.name, "file_type": "pptx", "parser": "python-pptx"})

    def parse_docx(self, filepath: Path, file_hash: str) -> ParsedDocument:
        from docx import Document as DocxDocument
        elements, full_text_parts = [], []
        doc = DocxDocument(filepath)
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            etype = "heading" if para.style.name.startswith("Heading") else "paragraph"
            elements.append(ParsedElement(text=text, element_type=etype,
                metadata={"source": filepath.name, "file_type": "docx"}))
            full_text_parts.append(text)
        return ParsedDocument(filename=filepath.name, title=filepath.stem, elements=elements,
            full_text="\n\n".join(full_text_parts), file_hash=file_hash,
            metadata={"source": filepath.name, "file_type": "docx", "parser": "python-docx"})

    def parse_image(self, filepath: Path, file_hash: str) -> Optional[ParsedDocument]:
        try:
            from PIL import Image
            import pytesseract
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image)
            if not text.strip():
                return None
            elements = [ParsedElement(text=text.strip(), element_type="image_ocr",
                metadata={"source": filepath.name, "file_type": "image_ocr"})]
            return ParsedDocument(filename=filepath.name, title=filepath.stem, elements=elements,
                full_text=text.strip(), file_hash=file_hash,
                metadata={"source": filepath.name, "file_type": "image_ocr"})
        except Exception as e:
            logger.warning(f"OCR failed for {filepath.name}: {e}")
            return None

    def parse_text(self, filepath: Path, file_hash: str) -> ParsedDocument:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        elements = [ParsedElement(text=content, element_type="text",
            metadata={"source": filepath.name, "file_type": "txt"})]
        return ParsedDocument(filename=filepath.name, title=filepath.stem, elements=elements,
            full_text=content, file_hash=file_hash,
            metadata={"source": filepath.name, "file_type": "txt"})
