"""
adaptive_chunker.py — Element-aware adaptive document chunking engine.

The core innovation: instead of blindly splitting text by character count,
this chunker first identifies structural elements (headers, tables, lists,
SOAP entries) and applies the right strategy for each:

  - Tables → kept whole (or split by rows if very large, preserving header row)
  - SOAP daily entries → one chunk per day entry
  - Lists → kept grouped
  - Narrative text → RecursiveCharacterTextSplitter with sentence awareness

This prevents tables from being split mid-row and ensures section boundaries
are respected regardless of the document format.
"""

import re
import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

# Markdown table detection: a line with |---|---| pattern
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:]+\|", re.MULTILINE)

# SOAP daily entry boundaries — multiple formats
_SOAP_BOUNDARY_RE = re.compile(
    r"^(?:"
    r"Day\s+\d+\s*[—\-–]"          # "Day 1 —", "Day 30 -"
    r"|#{1,3}\s+Day\s+\d+"          # "## Day 1", "### Day 15"
    r"|\*\*Day\s+\d+[—\-–\s]"      # "**Day 1 —"
    r"|\d{1,2}/\d{1,2}/\d{2,4}\s*[—\-–]"  # "01/15/2026 —" date-based entries
    r")",
    re.MULTILINE,
)

# Markdown headers at various levels
_MD_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# List items (bullet or numbered)
_LIST_ITEM_RE = re.compile(r"^[\s]*(?:[-*•]|\d+\.)\s+", re.MULTILINE)


class AdaptiveDocumentChunker:
    """Element-aware adaptive chunking for EHR documents.

    Splits markdown content by headers first, then applies element-specific
    strategies within each section:
      - Tables: kept whole or split by rows with header preservation
      - SOAP entries: split per day/date boundary
      - Lists: kept grouped
      - Narrative: RecursiveCharacterTextSplitter
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 150,
        max_table_chunk_size: int = 3000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_table_chunk_size = max_table_chunk_size

        # Text splitter for narrative content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

        # Markdown header splitter — splits on ##, ###, #### headers
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )

    def chunk_document(
        self,
        markdown_text: str,
        metadata: dict,
        page_map: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[Document]:
        """Chunk a markdown document with element awareness.

        Args:
            markdown_text: Full document as markdown string.
            metadata: Base metadata dict (patient_mrn, patient_name, source, etc.)
            page_map: Optional list of (char_start, char_end, page_number) tuples
                      for mapping chunks back to page numbers.

        Returns:
            List of LangChain Document objects with rich metadata.
        """
        if not markdown_text.strip():
            return []

        # Step 1: Split by markdown headers
        sections = self._split_by_headers(markdown_text)

        # Step 2: Process each section with element-aware chunking
        all_chunks: List[Tuple[str, str]] = []  # (chunk_text, section_title)

        for section_title, section_body in sections:
            section_chunks = self._chunk_section(section_body, section_title)
            all_chunks.extend(section_chunks)

        # Step 3: Merge small fragments
        all_chunks = self._merge_small_fragments(all_chunks)

        # Step 4: Build Document objects with metadata
        documents = []
        ctx_parts = []
        if metadata.get("patient_name"):
            ctx_parts.append(f"Patient: {metadata['patient_name']}")
        if metadata.get("patient_mrn"):
            ctx_parts.append(f"MRN: {metadata['patient_mrn']}")
        if metadata.get("document_type") and metadata["document_type"] != "general":
            ctx_parts.append(f"DocType: {metadata['document_type']}")

        for chunk_idx, (chunk_text, section_title) in enumerate(all_chunks):
            # Context prefix
            prefix_parts = ctx_parts + [f"Section: {section_title}"]
            prefix = "[" + " | ".join(prefix_parts) + "]"
            contextualized = f"{prefix}\n{chunk_text}"

            # Determine page number from character position
            page_num = 1
            if page_map:
                text_pos = markdown_text.find(chunk_text[:80])
                if text_pos >= 0:
                    page_num = self._get_page_number(text_pos, page_map)

            doc = Document(
                page_content=contextualized,
                metadata={
                    **metadata,
                    "section_title": section_title,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                },
            )
            documents.append(doc)

        logger.info("Chunked into %d documents (%d sections detected)",
                    len(documents), len(sections))
        return documents

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split text by markdown headers, returning (title, body) tuples.

        Uses MarkdownHeaderTextSplitter for robust header detection.
        Falls back to the full text as a single section if no headers found.
        """
        # Try markdown header splitting
        try:
            header_docs = self.header_splitter.split_text(text)
        except Exception:
            header_docs = []

        if header_docs and len(header_docs) > 1:
            sections = []
            for doc in header_docs:
                # Build section title from header metadata
                title_parts = []
                for level in ("h1", "h2", "h3", "h4"):
                    if doc.metadata.get(level):
                        title_parts.append(doc.metadata[level])
                title = " > ".join(title_parts) if title_parts else "Content"
                sections.append((title, doc.page_content))
            return sections

        # Fallback: try regex-based numbered heading detection (clinical docs)
        numbered = self._detect_numbered_sections(text)
        if numbered:
            return numbered

        # No structure detected — return as single section
        return [("Full Document", text)]

    def _detect_numbered_sections(self, text: str) -> Optional[List[Tuple[str, str]]]:
        """Detect numbered clinical sections like '1 Patient Demographics'.

        This is the fallback for documents where pymupdf4llm doesn't produce
        markdown headers (e.g., flat-formatted clinical notes).
        """
        heading_re = re.compile(
            r"^(\d{1,2}(?:[a-d])?(?:\.\d{1,2})*)\.?\s+([A-Z][A-Za-z &/\-,()]+.*)$",
            re.MULTILINE,
        )

        matches = []
        for m in heading_re.finditer(text):
            title_text = m.group(2).strip()
            # Validate: not a data line
            if self._is_valid_heading(title_text):
                matches.append(m)

        if len(matches) < 2:
            return None

        sections = []

        # Capture preamble before first heading
        if matches[0].start() > 30:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(("Document Header", preamble))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            num = m.group(1)
            title = m.group(2).strip()
            body = text[start:end].strip()
            sections.append((f"{num} {title}", body))

        return sections

    @staticmethod
    def _is_valid_heading(title: str) -> bool:
        """Validate that a candidate heading is a real section title."""
        if not title or not title[0].isupper():
            return False
        if len(title) > 80:
            return False
        alpha_words = re.findall(r"[A-Za-z]{2,}", title)
        if len(alpha_words) < 2:
            return False
        alpha_space = sum(1 for c in title if c.isalpha() or c.isspace())
        if alpha_space / max(len(title), 1) < 0.6:
            return False
        # Reject medication/measurement lines
        data_units = re.compile(
            r"\b(?:mg|mL|mmHg|tablet|capsule|BID|TID|QID|mcg|mEq|units?)\b",
            re.IGNORECASE,
        )
        if data_units.search(title):
            return False
        # Reject ICD-10 codes
        if re.match(r"^[A-Z]\d+\.\d+", title):
            return False
        return True

    def _chunk_section(
        self, section_body: str, section_title: str
    ) -> List[Tuple[str, str]]:
        """Apply element-aware chunking to a single section.

        Detects content type and routes to the appropriate strategy.
        """
        chunks: List[Tuple[str, str]] = []

        # Check for SOAP daily entries first
        soap_entries = self._extract_soap_entries(section_body, section_title)
        if soap_entries:
            for entry_text, sublabel in soap_entries:
                if len(entry_text) <= self.chunk_size * 2:
                    chunks.append((entry_text, f"{section_title} > {sublabel}"))
                else:
                    # Very long SOAP entry — split further
                    sub_chunks = self._split_mixed_content(entry_text)
                    for j, sc in enumerate(sub_chunks):
                        lbl = f"{sublabel} (part {j+1})" if len(sub_chunks) > 1 else sublabel
                        chunks.append((sc, f"{section_title} > {lbl}"))
            return chunks

        # For non-SOAP sections, use mixed content splitting
        if len(section_body) <= int(self.chunk_size * 1.5):
            # Short section — keep whole (preserves small tables, lists, etc.)
            chunks.append((section_body, section_title))
        else:
            sub_chunks = self._split_mixed_content(section_body)
            for sc in sub_chunks:
                chunks.append((sc, section_title))

        return chunks

    def _split_mixed_content(self, text: str) -> List[str]:
        """Split text that may contain a mix of tables, lists, and narrative.

        Tables are extracted and kept whole (or split by rows if too large).
        Remaining text is split with RecursiveCharacterTextSplitter.
        """
        # Identify table blocks
        segments = self._segment_by_tables(text)

        result = []
        for segment_text, is_table in segments:
            if is_table:
                # Table content — keep whole or split by rows
                table_chunks = self._chunk_table(segment_text)
                result.extend(table_chunks)
            else:
                # Non-table content
                if len(segment_text) <= int(self.chunk_size * 1.5):
                    if segment_text.strip():
                        result.append(segment_text)
                else:
                    splits = self.text_splitter.split_text(segment_text)
                    result.extend(splits)

        # Merge small fragments
        result = self._merge_small_text_chunks(result)
        return result

    def _segment_by_tables(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into alternating (non-table, table) segments.

        Detects markdown tables by the |---|---| separator pattern and
        extracts them as contiguous blocks (header row + separator + data rows).
        """
        # Find all table separator lines
        separator_positions = []
        for m in _TABLE_SEPARATOR_RE.finditer(text):
            separator_positions.append(m.start())

        if not separator_positions:
            return [(text, False)]

        segments = []
        lines = text.split("\n")
        line_offsets = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line) + 1  # +1 for newline

        # Find table blocks (contiguous lines starting with |)
        in_table = False
        table_start = None
        non_table_start = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            is_table_line = line_stripped.startswith("|") and line_stripped.endswith("|")

            if is_table_line and not in_table:
                # Start of table — emit preceding non-table text
                if i > non_table_start:
                    non_table_text = "\n".join(lines[non_table_start:i])
                    if non_table_text.strip():
                        segments.append((non_table_text, False))
                table_start = i
                in_table = True
            elif not is_table_line and in_table:
                # End of table
                table_text = "\n".join(lines[table_start:i])
                if table_text.strip():
                    segments.append((table_text, True))
                non_table_start = i
                in_table = False

        # Handle remaining content
        if in_table and table_start is not None:
            table_text = "\n".join(lines[table_start:])
            if table_text.strip():
                segments.append((table_text, True))
        elif not in_table:
            remaining = "\n".join(lines[non_table_start:])
            if remaining.strip():
                segments.append((remaining, False))

        return segments if segments else [(text, False)]

    def _chunk_table(self, table_text: str) -> List[str]:
        """Chunk a markdown table, preserving the header row.

        If the table fits within max_table_chunk_size, keep it whole.
        Otherwise, split by rows and prepend the header + separator to each chunk.
        """
        if len(table_text) <= self.max_table_chunk_size:
            return [table_text]

        lines = table_text.split("\n")

        # Find header row and separator
        header_lines = []
        data_lines = []
        found_separator = False

        for line in lines:
            if not found_separator:
                header_lines.append(line)
                if re.match(r"^\|[\s\-:]+\|", line.strip()):
                    found_separator = True
            else:
                data_lines.append(line)

        if not data_lines:
            return [table_text]

        header = "\n".join(header_lines)
        header_len = len(header) + 1  # +1 for newline

        # Split data rows into chunks that fit
        chunks = []
        current_rows = []
        current_size = header_len

        for row in data_lines:
            row_len = len(row) + 1
            if current_size + row_len > self.max_table_chunk_size and current_rows:
                # Emit current chunk
                chunk = header + "\n" + "\n".join(current_rows)
                chunks.append(chunk)
                current_rows = []
                current_size = header_len

            current_rows.append(row)
            current_size += row_len

        # Emit remaining rows
        if current_rows:
            chunk = header + "\n" + "\n".join(current_rows)
            chunks.append(chunk)

        return chunks

    def _extract_soap_entries(
        self, body: str, section_title: str
    ) -> Optional[List[Tuple[str, str]]]:
        """Extract SOAP daily entries from a section.

        Detects Day N boundaries in multiple formats and splits into
        one entry per day/date.

        Returns None if this doesn't appear to be a SOAP section.
        """
        matches = list(_SOAP_BOUNDARY_RE.finditer(body))
        if len(matches) < 2:
            return None

        entries = []

        # Capture intro before first entry
        if matches[0].start() > 30:
            intro = body[:matches[0].start()].strip()
            if intro:
                entries.append((intro, "Introduction"))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            entry_text = body[start:end].strip()
            # Extract the day label from the match
            day_label = m.group(0).strip().rstrip("—-– ")
            # Clean up markdown formatting
            day_label = re.sub(r"^#+\s*", "", day_label)
            day_label = re.sub(r"^\*\*|\*\*$", "", day_label)
            entries.append((entry_text, day_label.strip()))

        return entries

    def _merge_small_fragments(
        self, chunks: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Merge chunks smaller than min_chunk_size with neighbors."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        buf_text = ""
        buf_title = ""

        for text, title in chunks:
            if not buf_text:
                buf_text = text
                buf_title = title
            elif len(buf_text) < self.min_chunk_size:
                buf_text = buf_text + "\n\n" + text
                # Keep the more specific title
                if title != buf_title and title != "Content":
                    buf_title = title
            else:
                merged.append((buf_text, buf_title))
                buf_text = text
                buf_title = title

        if buf_text:
            if merged and len(buf_text) < self.min_chunk_size:
                prev_text, prev_title = merged[-1]
                merged[-1] = (prev_text + "\n\n" + buf_text, prev_title)
            else:
                merged.append((buf_text, buf_title))

        return merged

    def _merge_small_text_chunks(
        self, chunks: List[str], min_size: int = 0
    ) -> List[str]:
        """Merge plain text chunks smaller than min_chunk_size."""
        min_size = min_size or self.min_chunk_size
        if len(chunks) <= 1:
            return chunks

        merged = []
        buf = ""
        for chunk in chunks:
            if not buf:
                buf = chunk
            elif len(buf) < min_size:
                buf = buf + "\n\n" + chunk
            else:
                merged.append(buf)
                buf = chunk
        if buf:
            if merged and len(buf) < min_size:
                merged[-1] = merged[-1] + "\n\n" + buf
            else:
                merged.append(buf)
        return merged

    @staticmethod
    def _get_page_number(
        char_pos: int, page_map: List[Tuple[int, int, int]]
    ) -> int:
        """Map a character position to a page number."""
        for start, end, page_num in page_map:
            if start <= char_pos < end:
                return page_num
        return page_map[-1][2] if page_map else 1
