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

PATCH NOTES (applied over original):
  1. _SOAP_BOUNDARY_RE — added arrow (→), colon (:), and "Hospital Day N"
     variants that were causing day-boundary bleed in ARNI uptitration notes.
  2. _SentenceSnapTextSplitter — wraps RecursiveCharacterTextSplitter and
     snaps the overlap window to the nearest sentence boundary, eliminating
     mid-sentence chunk starts ("bilaterally. No wheezing..." artifacts).
  3. _dedup_chunks — sub-document content-hash dedup removes duplicate
     passages (repeated notes sections) before indexing.
  4. chunk_overlap reduced from 200 → 80 in tools_config.yml recommendation.
     The neighbor-expansion in tool_clinical_notes_rag already handles
     cross-chunk context continuity; large overlap was counterproductive.
"""

import re
import hashlib
import logging
import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

# Markdown table detection: a line with |---|---| pattern
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:]+\|", re.MULTILINE)

# ── PATCH 1: Expanded SOAP daily entry boundary regex ────────────────────────
# Original only matched [—\-–] as separators and missed:
#   - "Day 14 → 01/28/2026: Day 14 → Two Week Mark" (arrow separator)
#   - "01/28/2026:" (date with colon, no dash)
#   - "Hospital Day 3 —" (common inpatient note variant)
_SOAP_BOUNDARY_RE = re.compile(
    r"^(?:"
    # "Day 1 —", "Day 30 -", "Day 14 →", "Day 5:", "Day 1 "
    r"Day\s+\d+\s*(?:[—\-–:]|\s*→|\s*\u2014|\s*\u2013)"
    # "## Day 1", "### Day 15"
    r"|#{1,3}\s+Day\s+\d+"
    # "**Day 1 —", "**Day 14 →"
    r"|\*\*Day\s+\d+\s*(?:[—\-–:\s]|\s*→)"
    # "01/15/2026 —", "01/28/2026:", "1/5/26 -"
    r"|\d{1,2}/\d{1,2}/\d{2,4}\s*(?:[—\-–:]|\s*→)"
    # "Hospital Day 3 —", "Hospital Day 14:"
    r"|Hospital\s+Day\s+\d+\s*(?:[—\-–:]|\s*→)?"
    # "POD 1 —" (post-op day variant)
    r"|POD\s+\d+\s*(?:[—\-–:]|\s*→)?"
    r")",
    re.MULTILINE,
)

# Markdown headers at various levels
_MD_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# List items (bullet or numbered)
_LIST_ITEM_RE = re.compile(r"^[\s]*(?:[-*•]|\d+\.)\s+", re.MULTILINE)

# Sentence boundary: end of sentence followed by space and capital letter
# Used by _SentenceSnapTextSplitter to find clean overlap start points
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ── PATCH 2: Sentence-snapping text splitter ─────────────────────────────────

class _SentenceSnapTextSplitter:
    """
    Wraps RecursiveCharacterTextSplitter and snaps the overlap window to the
    nearest sentence boundary.

    Problem with standard overlap: a 200-char window landing mid-sentence
    produces chunks that start with "bilaterally. No wheezing..." — the
    opening word has no referent. This makes the chunk useless in isolation
    and poisons the reranker.

    Solution: after splitting, trim each chunk's start to the first sentence
    boundary within a tolerance window. The tolerance is min(overlap,
    max_snap_chars) so we never snap more than necessary.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 80,
        max_snap_chars: int = 120,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_snap_chars = max_snap_chars

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        raw_chunks = self._splitter.split_text(text)
        return [self._snap_to_sentence(c) for c in raw_chunks]

    @staticmethod
    def _snap_to_sentence(chunk: str, max_snap: int = 120) -> str:
        """
        If the chunk starts mid-sentence (no capital after punctuation at
        the very start), find the first sentence boundary within max_snap
        chars and trim to it.

        We only snap if the chunk does NOT already start cleanly:
          - Starts with a capital letter after punctuation, OR
          - Starts with a markdown heading (##), OR
          - Starts with a list marker (- , * , 1. )
        """
        if not chunk:
            return chunk

        stripped = chunk.lstrip()

        # Already starts cleanly
        if (
            stripped.startswith("##")       # heading
            or stripped.startswith("| ")    # table
            or re.match(r"^[-*•]\s", stripped)  # list
            or re.match(r"^\d+\.\s", stripped)  # numbered list
        ):
            return chunk

        # Find first sentence boundary within snap window
        search_region = chunk[:max_snap]
        match = _SENTENCE_END_RE.search(search_region, pos=1)  # pos=1: skip pos 0
        if match:
            return chunk[match.end():]

        return chunk


# ── Main Chunker ──────────────────────────────────────────────────────────────

class AdaptiveDocumentChunker:
    """Element-aware adaptive chunking for EHR documents.

    Splits markdown content by headers first, then applies element-specific
    strategies within each section:
      - Tables: kept whole or split by rows with header preservation
      - SOAP entries: split per day/date boundary
      - Lists: kept grouped
      - Narrative: SentenceSnapTextSplitter (sentence-boundary overlap)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 80,
        min_chunk_size: int = 150,
        max_table_chunk_size: int = 3000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_table_chunk_size = max_table_chunk_size

        self._semantic_fallback_embedding_model = os.environ.get(
            "MEDGRAPH_SEMANTIC_CHUNKER_EMBEDDING_MODEL",
            "NeuML/pubmedbert-base-embeddings",
        )
        self._semantic_fallback_breakpoint_percentile = int(
            os.environ.get("MEDGRAPH_SEMANTIC_CHUNKER_BREAKPOINT_PERCENTILE", "88")
        )

        # ── PATCH 2: Use sentence-snapping splitter instead of bare RCTS ──
        self.text_splitter = _SentenceSnapTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_snap_chars=min(chunk_overlap, 120),
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

        # ── PATCH 3: Sub-document duplicate removal ─────────────────────────
        all_chunks = self._dedup_chunks(all_chunks)

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

    # ── PATCH 3: Sub-document dedup ──────────────────────────────────────────

    @staticmethod
    def _dedup_chunks(
        chunks: List[Tuple[str, str]],
        min_len: int = 100,
    ) -> List[Tuple[str, str]]:
        """
        Remove near-duplicate chunks within the same document.

        Problem: some PDFs contain repeated note sections (e.g., a vitals
        summary appearing once in prose and once as a near-duplicate table).
        Document-level content hash (in __init__.py) catches whole-document
        duplicates but not sub-document duplicate passages.

        Strategy: normalize each chunk text (lowercase + whitespace collapse),
        hash it, and drop any chunk whose normalized hash has already appeared.
        Chunks shorter than min_len are exempt (too short to be meaningful dups).
        """
        seen: set = set()
        result: List[Tuple[str, str]] = []

        for chunk_text, section_title in chunks:
            if len(chunk_text) < min_len:
                result.append((chunk_text, section_title))
                continue

            normalized = re.sub(r"\s+", " ", chunk_text.lower().strip())
            chunk_hash = hashlib.md5(normalized.encode()).hexdigest()

            if chunk_hash in seen:
                logger.debug(
                    "[Chunker] Dropping duplicate chunk in section '%s' (%d chars)",
                    section_title, len(chunk_text),
                )
                continue

            seen.add(chunk_hash)
            result.append((chunk_text, section_title))

        return result

    # ── Header splitting (unchanged, kept for reference) ──────────────────────

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split text by markdown headers, returning (title, body) tuples."""
        try:
            header_docs = self.header_splitter.split_text(text)
        except Exception:
            header_docs = []

        if header_docs and len(header_docs) > 1:
            sections = []
            for doc in header_docs:
                title_parts = []
                for level in ("h1", "h2", "h3", "h4"):
                    if doc.metadata.get(level):
                        title_parts.append(doc.metadata[level])
                title = " > ".join(title_parts) if title_parts else "Content"
                sections.append((title, doc.page_content))
            return sections

        numbered = self._detect_numbered_sections(text)
        if numbered:
            return numbered

        keyword_sections = self._detect_keyword_sections(text)
        if keyword_sections:
            return keyword_sections

        semantic_parts = self._split_semantic_fallback(text)
        if semantic_parts and len(semantic_parts) > 1:
            return [(f"Narrative Semantic Part {i+1}", part)
                    for i, part in enumerate(semantic_parts)]

        return [("Full Document", text)]

    def _split_semantic_fallback(self, text: str) -> Optional[List[str]]:
        """
        Split unstructured narrative text into semantic parts using local embeddings.

        Production-safety:
        - Returns None (caller falls back to "Full Document") if embeddings cannot be loaded.
        - Caps the number of embedding units to avoid pathological runtime on very large notes.
        """
        if not text or len(text.strip()) < 2000:
            return None

        try:
            import numpy as np
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            return None

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self._semantic_fallback_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 16,
                },
            )
        except Exception:
            return None

        raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not raw_paragraphs:
            return None

        target_unit_chars = max(800, min(1600, self.chunk_size))
        units: list[str] = []
        cur = []
        cur_chars = 0
        for p in raw_paragraphs:
            cur.append(p)
            cur_chars += len(p)
            if cur_chars >= target_unit_chars:
                units.append("\n\n".join(cur))
                cur = []
                cur_chars = 0
        if cur:
            units.append("\n\n".join(cur))

        max_units = int(os.environ.get("MEDGRAPH_SEMANTIC_CHUNKER_MAX_UNITS", "80"))
        if len(units) > max_units:
            units = units[:max_units]
        if len(units) < 2:
            return None

        try:
            unit_vectors = embeddings.embed_documents(units)
        except Exception:
            return None

        import numpy as np
        vecs = np.array(unit_vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(units):
            return None

        adjacency = vecs[:-1] * vecs[1:]
        cos_sims = adjacency.sum(axis=1)
        dists = 1.0 - cos_sims

        if dists.size == 0:
            return None

        threshold = float(np.percentile(dists, self._semantic_fallback_breakpoint_percentile))
        cut_indices = [i for i, d in enumerate(dists) if d >= threshold]
        if not cut_indices:
            return None

        min_segment_chars = int(self.min_chunk_size * 1.2)
        segments: list[str] = []
        current_units: list[str] = []
        current_chars = 0

        def _flush():
            nonlocal current_units, current_chars
            seg = "\n\n".join(current_units).strip()
            if seg:
                segments.append(seg)
            current_units = []
            current_chars = 0

        for unit_idx, unit in enumerate(units):
            current_units.append(unit)
            current_chars += len(unit)

            if unit_idx in {i + 1 for i in cut_indices}:
                if current_chars >= min_segment_chars:
                    _flush()
                    if len(segments) >= 20:
                        break

        if current_units and (not segments or len(segments) < 20):
            _flush()

        if len(segments) < 2:
            return None
        return segments

    def _detect_keyword_sections(self, text: str) -> Optional[List[Tuple[str, str]]]:
        """Detect common clinical note headings."""
        keyword_re = re.compile(
            r"^(?P<label>"
            r"Subjective|Objective"
            r"|Assessment(?:\s*[&/]\s*|\s+and\s+)?Plan"
            r"|Assessment"
            r"|A\s*/\s*P|A/P"
            r"|HPI|History of Present Illness"
            r"|ROS|Review of Systems"
            r"|Physical Exam(?:ination)?"
            r"|Impression|Plan|Chief Complaint"
            r"|Past Medical History|PMH"
            r"|Family History|FH"
            r"|Social History|SH"
            r"|Vital Signs?|VS"
            r"|(?:Current\s+)?Medications?"
            r"|Allergies?"
            r"|Discharge (?:Instructions?|Summary|Diagnosis|Medications?)"
            r"|Follow[\-\s]?Up(?: Instructions?| Plan| Appointments?)?"
            r"|Admission Diagnosis"
            r"|Problem List"
            r")\s*:?(?:\s+(?P<inline>[^\n].*))?$",
            flags=re.IGNORECASE | re.MULTILINE,
        )

        matches = [m for m in keyword_re.finditer(text) if (m.group("label") or "").strip()]
        if len(matches) < 2:
            return None

        sections: list[tuple[str, str]] = []

        if matches[0].start() > 30:
            pre = text[: matches[0].start()].strip()
            if pre:
                sections.append(("Preamble", pre))

        for i, m in enumerate(matches):
            label = m.group("label").strip()
            inline_body = (m.group("inline") or "").strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            trailing_body = text[start:end].strip()
            parts = []
            if inline_body:
                parts.append(inline_body)
            if trailing_body:
                parts.append(trailing_body)
            body = "\n".join(parts).strip()
            if body:
                sections.append((label, body))

        return sections if len(sections) >= 2 else None

    def _detect_numbered_sections(self, text: str) -> Optional[List[Tuple[str, str]]]:
        """Detect numbered clinical sections like '1 Patient Demographics'."""
        heading_re = re.compile(
            r"^(\d{1,2}(?:[a-d])?(?:\.\d{1,2})*)\.?\s+([A-Z][A-Za-z &/\-,()]+.*)$",
            re.MULTILINE,
        )

        matches = []
        for m in heading_re.finditer(text):
            title_text = m.group(2).strip()
            if self._is_valid_heading(title_text):
                matches.append(m)

        if len(matches) < 2:
            return None

        sections = []

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
        data_units = re.compile(
            r"\b(?:mg|mL|mmHg|tablet|capsule|BID|TID|QID|mcg|mEq|units?)\b",
            re.IGNORECASE,
        )
        if data_units.search(title):
            return False
        if re.match(r"^[A-Z]\d+\.\d+", title):
            return False
        return True

    def _chunk_section(
        self, section_body: str, section_title: str
    ) -> List[Tuple[str, str]]:
        """Apply element-aware chunking to a single section."""
        chunks: List[Tuple[str, str]] = []

        soap_entries = self._extract_soap_entries(section_body, section_title)
        if soap_entries:
            for entry_text, sublabel in soap_entries:
                if len(entry_text) <= self.chunk_size * 2:
                    chunks.append((entry_text, f"{section_title} > {sublabel}"))
                else:
                    sub_chunks = self._split_mixed_content(entry_text)
                    for j, sc in enumerate(sub_chunks):
                        lbl = f"{sublabel} (part {j+1})" if len(sub_chunks) > 1 else sublabel
                        chunks.append((sc, f"{section_title} > {lbl}"))
            return chunks

        if len(section_body) <= int(self.chunk_size * 1.5):
            chunks.append((section_body, section_title))
        else:
            sub_chunks = self._split_mixed_content(section_body)
            for sc in sub_chunks:
                chunks.append((sc, section_title))

        return chunks

    def _split_mixed_content(self, text: str) -> List[str]:
        """Split text that may contain a mix of tables, lists, and narrative."""
        segments = self._segment_by_tables(text)

        result = []
        for segment_text, is_table in segments:
            if is_table:
                table_chunks = self._chunk_table(segment_text)
                result.extend(table_chunks)
            else:
                if len(segment_text) <= int(self.chunk_size * 1.5):
                    if segment_text.strip():
                        result.append(segment_text)
                else:
                    splits = self.text_splitter.split_text(segment_text)
                    result.extend(splits)

        result = self._merge_small_text_chunks(result)
        return result

    def _segment_by_tables(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into alternating (non-table, table) segments."""
        separator_positions = []
        for m in _TABLE_SEPARATOR_RE.finditer(text):
            separator_positions.append(m.start())

        if not separator_positions:
            return [(text, False)]

        segments = []
        lines = text.split("\n")
        in_table = False
        table_start = None
        non_table_start = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            is_table_line = line_stripped.startswith("|") and line_stripped.endswith("|")

            if is_table_line and not in_table:
                if i > non_table_start:
                    non_table_text = "\n".join(lines[non_table_start:i])
                    if non_table_text.strip():
                        segments.append((non_table_text, False))
                table_start = i
                in_table = True
            elif not is_table_line and in_table:
                table_text = "\n".join(lines[table_start:i])
                if table_text.strip():
                    segments.append((table_text, True))
                non_table_start = i
                in_table = False

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
        """Chunk a markdown table, preserving the header row."""
        if len(table_text) <= self.max_table_chunk_size:
            return [table_text]

        lines = table_text.split("\n")
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
        header_len = len(header) + 1

        chunks = []
        current_rows = []
        current_size = header_len

        for row in data_lines:
            row_len = len(row) + 1
            if current_size + row_len > self.max_table_chunk_size and current_rows:
                chunk = header + "\n" + "\n".join(current_rows)
                chunks.append(chunk)
                current_rows = []
                current_size = header_len

            current_rows.append(row)
            current_size += row_len

        if current_rows:
            chunk = header + "\n" + "\n".join(current_rows)
            chunks.append(chunk)

        return chunks

    def _extract_soap_entries(
        self, body: str, section_title: str
    ) -> Optional[List[Tuple[str, str]]]:
        """Extract SOAP daily entries from a section."""
        matches = list(_SOAP_BOUNDARY_RE.finditer(body))
        if len(matches) < 2:
            return None

        entries = []

        if matches[0].start() > 30:
            intro = body[:matches[0].start()].strip()
            if intro:
                entries.append((intro, "Introduction"))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            entry_text = body[start:end].strip()
            day_label = m.group(0).strip().rstrip("—-–: →")
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
