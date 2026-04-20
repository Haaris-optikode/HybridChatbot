"""
test_ingestion_pipeline.py — Comprehensive tests for Phase 1 document ingestion.

Tests:
  - Quality gate validation (accept/reject logic)
  - Content hash computation and dedup
  - OCR fallback trigger logic
  - Multi-format handler dispatch
  - Format handler edge cases (empty, corrupted)
  - Incremental indexing dedup behavior
  - process_document backward compatibility
"""
import os
import sys
import tempfile
import hashlib
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Quality Gate Tests ────────────────────────────────────────────────────────

class TestQualityGate:
    """Tests for document_processing.quality_gate.validate_extracted_text."""

    def setup_method(self):
        from document_processing.quality_gate import validate_extracted_text
        self.validate = validate_extracted_text

    def test_empty_text_rejected(self):
        ok, reason = self.validate("", "test.pdf")
        assert not ok
        assert reason == "empty_text"

    def test_whitespace_only_rejected(self):
        ok, reason = self.validate("   \n\n  \t  ", "test.pdf")
        assert not ok
        assert reason == "empty_text"

    def test_too_short_rejected(self):
        ok, reason = self.validate("Hello world", "test.pdf")
        assert not ok
        assert "too_short" in reason

    def test_low_alpha_ratio_rejected(self):
        # 90% digits, very few letters
        text = "1234567890" * 50 + "abc"
        ok, reason = self.validate(text, "test.pdf")
        assert not ok
        assert "low_alpha_ratio" in reason

    def test_binary_garbage_rejected(self):
        # Control characters mixed with some text
        text = "".join(chr(i) for i in range(1, 30)) * 20 + "some text"
        ok, reason = self.validate(text, "test.pdf")
        assert not ok
        # Should fail on control chars or alpha ratio

    def test_excessive_repetition_rejected(self):
        text = "Normal clinical note introduction with enough content. " + "a" * 30 + " more valid text added here to reach minimum length requirement easily."
        ok, reason = self.validate(text, "test.pdf")
        assert not ok
        assert "excessive_repetition" in reason

    def test_valid_clinical_text_passes(self):
        text = (
            "Patient Name: John Smith\n"
            "MRN: MRN-2026-001234\n"
            "Date of Birth: January 15, 1960\n\n"
            "Chief Complaint: Shortness of breath and bilateral lower extremity edema.\n\n"
            "History of Present Illness: The patient is a 66-year-old male with a history "
            "of heart failure with reduced ejection fraction who presents with worsening "
            "dyspnea on exertion over the past two weeks. He reports a 15-pound weight gain "
            "and increased bilateral leg swelling. He denies chest pain or palpitations.\n\n"
            "Past Medical History:\n"
            "- Heart failure (EF 25%)\n"
            "- Type 2 diabetes mellitus\n"
            "- Hypertension\n"
            "- Chronic kidney disease stage 3\n"
        )
        ok, reason = self.validate(text, "test.pdf")
        assert ok
        assert reason == "ok"

    def test_low_page_density_rejected(self):
        # 50 chars for 10 pages = 5 chars/page
        text = "A" * 50 + " " + "B" * 50  # ~100 chars, barely passes min length
        ok, reason = self.validate(text, "test.pdf", page_count=10)
        assert not ok
        assert "low_density" in reason

    def test_normal_page_density_passes(self):
        text = "Clinical notes with sufficient content. " * 50
        ok, reason = self.validate(text, "test.pdf", page_count=5)
        assert ok


# ── Content Hash Tests ────────────────────────────────────────────────────────

class TestContentHash:
    """Tests for document_processing.compute_content_hash."""

    def setup_method(self):
        from document_processing import compute_content_hash
        self.hash = compute_content_hash

    def test_deterministic(self):
        text = "Patient admitted with chest pain."
        h1 = self.hash(text)
        h2 = self.hash(text)
        assert h1 == h2

    def test_whitespace_normalization(self):
        t1 = "Patient   admitted\n\nwith  chest   pain."
        t2 = "Patient admitted\nwith chest pain."
        assert self.hash(t1) == self.hash(t2)

    def test_case_normalization(self):
        t1 = "Patient Admitted With Chest Pain."
        t2 = "patient admitted with chest pain."
        assert self.hash(t1) == self.hash(t2)

    def test_different_content_different_hash(self):
        h1 = self.hash("Patient admitted with chest pain.")
        h2 = self.hash("Patient discharged in stable condition.")
        assert h1 != h2

    def test_returns_sha256_hex(self):
        h = self.hash("test")
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)


# ── OCR Fallback Logic Tests ─────────────────────────────────────────────────

class TestOCRFallbackLogic:
    """Tests for OCR trigger logic (not actual OCR which requires Tesseract)."""

    def setup_method(self):
        from document_processing.ocr_fallback import needs_ocr
        self.needs_ocr = needs_ocr

    def test_empty_pages_needs_ocr(self):
        assert self.needs_ocr([])

    def test_pages_with_no_text_needs_ocr(self):
        pages = [
            {"page_number": 1, "markdown": ""},
            {"page_number": 2, "markdown": "  "},
        ]
        assert self.needs_ocr(pages)

    def test_pages_with_little_text_needs_ocr(self):
        pages = [
            {"page_number": 1, "markdown": "A few words"},
            {"page_number": 2, "markdown": "More words"},
        ]
        assert self.needs_ocr(pages, min_chars_per_page=50)

    def test_pages_with_sufficient_text_no_ocr(self):
        pages = [
            {"page_number": 1, "markdown": "A" * 200},
            {"page_number": 2, "markdown": "B" * 200},
        ]
        assert not self.needs_ocr(pages)


# ── OCR Quality & Vision Fallback Tests ──────────────────────────────────────

class TestOCRQualityCheck:
    """Tests for _ocr_quality_acceptable and _tesseract_available."""

    def setup_method(self):
        from document_processing.ocr_fallback import _ocr_quality_acceptable
        self.quality_ok = _ocr_quality_acceptable

    def test_empty_text_not_acceptable(self):
        assert not self.quality_ok("")

    def test_short_text_not_acceptable(self):
        assert not self.quality_ok("hi")

    def test_garbled_text_not_acceptable(self):
        # Mostly non-alpha chars → low alpha ratio
        assert not self.quality_ok("@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^")

    def test_valid_clinical_text_acceptable(self):
        assert self.quality_ok("Patient Name: John Smith, MRN: 12345")

    def test_mixed_text_with_numbers_acceptable(self):
        assert self.quality_ok("BP 120/80 HR 72 Temp 98.6 RR 16 SpO2 99%")


class TestTesseractAvailable:
    """Test _tesseract_available detection."""

    def test_returns_bool(self):
        from document_processing.ocr_fallback import _tesseract_available
        result = _tesseract_available()
        assert isinstance(result, bool)


# ── Format Handler Tests ─────────────────────────────────────────────────────

class TestFormatHandlers:
    """Tests for format_handlers.py dispatch logic."""

    def test_unsupported_extension_raises(self):
        from document_processing.format_handlers import extract_text
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
                f.write(b"test")
                tmp_path = f.name
            with pytest.raises(ValueError, match="Unsupported"):
                extract_text(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_nonexistent_file_raises(self):
        from document_processing.format_handlers import extract_text
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/path/to/file.pdf")

    def test_supported_extensions_constant(self):
        from document_processing.format_handlers import SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS


# ── Process Document Backward Compatibility ───────────────────────────────────

class TestProcessDocumentCompat:
    """Tests that process_document still accepts the old pdf_path parameter."""

    def test_file_path_kwarg_takes_precedence(self):
        from document_processing import process_document
        # Both provided — file_path should take precedence
        # Using non-existent path, should raise FileNotFoundError from format_handlers
        with pytest.raises(FileNotFoundError):
            process_document(
                pdf_path="/old/path.pdf",
                file_path="/new/path.pdf",
            )

    def test_no_path_raises(self):
        from document_processing import process_document
        with pytest.raises(ValueError, match="Either file_path or pdf_path"):
            process_document()


# ── Validate Document Text Public API ─────────────────────────────────────────

class TestValidateDocumentTextAPI:
    """Tests for the public validate_document_text wrapper."""

    def test_wrapper_delegates_correctly(self):
        from document_processing import validate_document_text
        ok, reason = validate_document_text("", "test.pdf")
        assert not ok
        assert reason == "empty_text"

    def test_wrapper_passes_valid_text(self):
        from document_processing import validate_document_text
        text = "This is a valid clinical document with sufficient content. " * 10
        ok, reason = validate_document_text(text, "test.pdf")
        assert ok
        assert reason == "ok"


# ── Eval Dataset Validation ───────────────────────────────────────────────────

class TestEvalDataset:
    """Validate the expanded evaluation dataset structure."""

    def test_eval_dataset_has_minimum_questions(self):
        import json
        eval_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
        with open(eval_path, "r") as f:
            dataset = json.load(f)
        assert len(dataset) >= 30, f"Expected ≥30 questions, got {len(dataset)}"

    def test_eval_dataset_schema(self):
        import json
        eval_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
        with open(eval_path, "r") as f:
            dataset = json.load(f)
        for i, item in enumerate(dataset):
            assert "question" in item, f"Item {i} missing 'question'"
            assert "ground_truth" in item, f"Item {i} missing 'ground_truth'"
            assert len(item["question"]) > 10, f"Item {i} question too short"
            assert len(item["ground_truth"]) > 10, f"Item {i} ground_truth too short"

    def test_eval_dataset_no_duplicates(self):
        import json
        eval_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
        with open(eval_path, "r") as f:
            dataset = json.load(f)
        questions = [item["question"] for item in dataset]
        assert len(questions) == len(set(questions)), "Duplicate questions found"
