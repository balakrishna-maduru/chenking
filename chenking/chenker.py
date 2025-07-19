import logging
import time
from typing import Dict, Any


class Chenker:
    """
    Performs a series of checks on the input document.
    Each check returns structured data under a named key.
    """

    def __init__(self, **kwargs):
        # Configuration variables with default values
        self.min_word_count = kwargs.get("min_word_count", 10)
        self.max_word_count = kwargs.get("max_word_count", 50)  # Reduced for testing
        self.min_char_count = kwargs.get("min_char_count", 50)
        self.max_char_count = kwargs.get("max_char_count", 50000)
        self.max_line_count = kwargs.get("max_line_count", 1000)
        self.check_timeout = kwargs.get("check_timeout", 30)  # seconds
        self.enable_detailed_logging = kwargs.get("enable_detailed_logging", False)
        self.supported_formats = kwargs.get(
            "supported_formats", ["txt", "md", "html", "json"]
        )
        self.required_fields = kwargs.get("required_fields", ["content"])
        self.optional_fields = kwargs.get(
            "optional_fields", ["title", "author", "date", "metadata"]
        )
        
        # Page splitting configuration
        self.enable_page_splitting = kwargs.get("enable_page_splitting", True)
        self.max_pages_to_process = kwargs.get("max_pages_to_process", 100)
        self.min_words_per_page = kwargs.get("min_words_per_page", 5)
        
        # Chenk numbering configuration
        self.enable_chenk_numbering = kwargs.get("enable_chenk_numbering", True)
        self.chenk_counter = 0  # Initialize chenk counter

        # Setup logging based on configuration
        self._setup_logging()

        # Validate configuration
        self._validate_config()

        # Register more checks here if needed
        self.checks = {
            "basic_check": self._basic_check,
            "length_check": self._length_check,
            "format_check": self._format_check,
            "field_check": self._field_check,
            "content_quality_check": self._content_quality_check,
            "page_split_check": self._page_split_check,
        }

    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_level = logging.DEBUG if self.enable_detailed_logging else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.min_word_count < 0 or self.max_word_count < self.min_word_count:
            raise ValueError("Invalid word count configuration")

        if self.min_char_count < 0 or self.max_char_count < self.min_char_count:
            raise ValueError("Invalid character count configuration")

        if self.max_line_count <= 0:
            raise ValueError("Max line count must be positive")

        if self.check_timeout <= 0:
            raise ValueError("Check timeout must be positive")

        self.logger.debug("Configuration validated successfully")

    def run_checks(self, document: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all configured checks on the document with timeout protection."""
        results = {}
        
        # Reset chenk counter for each document
        self.chenk_counter = 0

        self.logger.info(f"Starting document checks with {len(self.checks)} check(s)")

        for check_name, check_func in self.checks.items():
            start_time = time.time()
            try:
                self.logger.debug(f"Running check: {check_name}")

                # Run check with timeout protection
                check_result = self._run_check_with_timeout(check_func, document)

                # Increment chenk counter
                self.chenk_counter += 1

                elapsed_time = time.time() - start_time
                results[check_name] = {
                    "chenk_number": self.chenk_counter if self.enable_chenk_numbering else None,
                    "data": check_result,
                    "status": "success",
                    "execution_time": elapsed_time,
                }

                self.logger.debug(
                    f"Check {check_name} (chenk #{self.chenk_counter}) completed in {elapsed_time:.2f}s"
                )

            except Exception as e:
                self.chenk_counter += 1  # Still increment even on error
                elapsed_time = time.time() - start_time
                results[check_name] = {
                    "chenk_number": self.chenk_counter if self.enable_chenk_numbering else None,
                    "data": {},
                    "status": "error",
                    "error": str(e),
                    "execution_time": elapsed_time,
                }
                self.logger.error(f"Check {check_name} (chenk #{self.chenk_counter}) failed: {str(e)}")

        self.logger.info("Document checks completed")
        return results

    def _run_check_with_timeout(self, check_func, document):
        """Run a check function with timeout protection."""
        # Note: For production use, consider using threading or asyncio for timeout
        # This is a simplified implementation
        start_time = time.time()
        result = check_func(document)

        if time.time() - start_time > self.check_timeout:
            self.logger.warning(f"Check exceeded timeout of {self.check_timeout}s")

        return result

    def _basic_check(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Example: check if content is non-empty."""
        content = document.get("content", "")
        word_count = len(content.strip().split()) if content.strip() else 0

        return {
            "has_content": bool(content.strip()),
            "word_count": word_count,
            "is_word_count_valid": self.min_word_count
            <= word_count
            <= self.max_word_count,
        }

    def _length_check(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Check document length against configured limits."""
        content = document.get("content", "")
        char_count = len(content)
        line_count = content.count("\n") + 1 if content else 0

        return {
            "char_count": char_count,
            "line_count": line_count,
            "is_char_count_valid": self.min_char_count
            <= char_count
            <= self.max_char_count,
            "is_line_count_valid": line_count <= self.max_line_count,
            "length_summary": {
                "within_char_limits": self.min_char_count
                <= char_count
                <= self.max_char_count,
                "within_line_limits": line_count <= self.max_line_count,
            },
        }

    def _format_check(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Check if document format is supported."""
        doc_format = document.get("format", "").lower()
        file_extension = document.get("file_extension", "").lower().lstrip(".")

        # Check both format field and file extension
        format_supported = (
            doc_format in self.supported_formats
            or file_extension in self.supported_formats
        )

        return {
            "format": doc_format,
            "file_extension": file_extension,
            "is_format_supported": format_supported,
            "supported_formats": self.supported_formats,
        }

    def _field_check(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Check if required and optional fields are present."""
        present_fields = list(document.keys())
        missing_required = [
            field for field in self.required_fields if field not in document
        ]
        present_optional = [
            field for field in self.optional_fields if field in document
        ]
        missing_optional = [
            field for field in self.optional_fields if field not in document
        ]

        return {
            "present_fields": present_fields,
            "required_fields_status": {
                "all_present": len(missing_required) == 0,
                "missing": missing_required,
                "present": [
                    field for field in self.required_fields if field in document
                ],
            },
            "optional_fields_status": {
                "present": present_optional,
                "missing": missing_optional,
            },
            "field_coverage": {
                "required_coverage": len(self.required_fields) - len(missing_required),
                "total_required": len(self.required_fields),
                "optional_coverage": len(present_optional),
                "total_optional": len(self.optional_fields),
            },
        }

    def _content_quality_check(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality checks on document content."""
        content = document.get("content", "")

        # Basic quality metrics
        sentences = content.split(".") if content else []
        paragraphs = content.split("\n\n") if content else []
        words = content.split() if content else []

        # Calculate readability metrics
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        avg_sentences_per_paragraph = (
            len(sentences) / len(paragraphs) if paragraphs else 0
        )

        # Check for common quality issues
        has_title = bool(document.get("title", "").strip())
        has_metadata = bool(document.get("metadata"))
        has_author = bool(document.get("author", "").strip())

        return {
            "structure_metrics": {
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "avg_sentences_per_paragraph": round(avg_sentences_per_paragraph, 2),
            },
            "metadata_completeness": {
                "has_title": has_title,
                "has_author": has_author,
                "has_metadata": has_metadata,
                "completeness_score": sum([has_title, has_author, has_metadata]) / 3,
            },
            "quality_indicators": {
                "content_density": len(words) / len(content) if content else 0,
                "structural_organization": len(paragraphs) > 1,
                "proper_length": self.min_word_count
                <= len(words)
                <= self.max_word_count,
            },
        }

    def _page_split_check(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split document content based on page information from Tika metadata.

        This check looks for page indicators in the content or metadata and
        splits the document accordingly, providing page-level analysis.
        """
        # Check if page splitting is enabled
        if not self.enable_page_splitting:
            return {
                "has_page_info": False,
                "page_count": 1,
                "pages": [],
                "split_method": "disabled",
                "page_statistics": {},
                "message": "Page splitting is disabled"
            }

        content = document.get("content", "")
        metadata = document.get("metadata", {})

        # Initialize result structure
        result = {
            "has_page_info": False,
            "page_count": 0,
            "pages": [],
            "split_method": "none",
            "page_statistics": {},
        }

        # Method 1: Check if Tika provided page count in metadata
        page_count = None
        if "page_count" in metadata:
            page_count = metadata["page_count"]
            result["has_page_info"] = True
            result["page_count"] = page_count
            result["split_method"] = "metadata_page_count"

        # Method 2: Look for page break markers in content (common Tika patterns)
        page_markers = [
            "\f",  # Form feed character (common page break)
            "\\f",  # Escaped form feed
            "\n\n--- Page ",  # Custom page markers
            "\n--- PAGE ",
            "\nPage ",
            "[PAGE BREAK]",
            "<!-- PAGE BREAK -->",
        ]

        pages_by_marker = []
        split_pattern = None

        for marker in page_markers:
            if marker in content:
                pages_by_marker = content.split(marker)
                split_pattern = marker
                break

        # Method 3: If we have page count but no markers, estimate page splits
        if page_count and page_count > 1 and not pages_by_marker:
            # Split content approximately by page count
            content_length = len(content)
            chars_per_page = content_length // page_count

            pages_by_marker = []
            for i in range(page_count):
                start_idx = i * chars_per_page
                end_idx = (i + 1) * chars_per_page if i < page_count - 1 else content_length
                page_content = content[start_idx:end_idx].strip()
                if page_content:
                    pages_by_marker.append(page_content)

            result["split_method"] = "estimated_by_length"
            split_pattern = "estimated_split"

        # Process the pages if we found any splits
        if pages_by_marker and len(pages_by_marker) > 1:
            result["has_page_info"] = True
            result["page_count"] = len(pages_by_marker)
            result["split_pattern"] = split_pattern

            # Analyze each page
            for i, page_content in enumerate(pages_by_marker, 1):
                page_content = page_content.strip()
                if not page_content:
                    continue

                words = page_content.split()
                sentences = [s.strip() for s in page_content.split(".") if s.strip()]

                page_info = {
                    "page_number": i,
                    "content": page_content,
                    "word_count": len(words),
                    "char_count": len(page_content),
                    "sentence_count": len(sentences),
                    "starts_with": page_content[:50] + "..." if len(page_content) > 50 else page_content,
                    "ends_with": "..." + page_content[-50:] if len(page_content) > 50 else page_content,
                }

                result["pages"].append(page_info)

        # If no page splits found, treat entire content as single page
        if not result["has_page_info"] or not result["pages"]:
            result["page_count"] = 1
            result["split_method"] = "single_page"

            words = content.split()
            sentences = [s.strip() for s in content.split(".") if s.strip()]

            single_page = {
                "page_number": 1,
                "content": content,
                "word_count": len(words),
                "char_count": len(content),
                "sentence_count": len(sentences),
                "starts_with": content[:50] + "..." if len(content) > 50 else content,
                "ends_with": "..." + content[-50:] if len(content) > 50 else content,
            }

            result["pages"] = [single_page]

        # Calculate page statistics
        if result["pages"]:
            word_counts = [page["word_count"] for page in result["pages"]]
            char_counts = [page["char_count"] for page in result["pages"]]

            result["page_statistics"] = {
                "total_pages": len(result["pages"]),
                "avg_words_per_page": sum(word_counts) / len(word_counts),
                "min_words_per_page": min(word_counts),
                "max_words_per_page": max(word_counts),
                "avg_chars_per_page": sum(char_counts) / len(char_counts),
                "min_chars_per_page": min(char_counts),
                "max_chars_per_page": max(char_counts),
                "page_length_variance": {
                    "consistent_length": max(word_counts) - min(word_counts) < 100,
                    "word_count_range": max(word_counts) - min(word_counts),
                },
            }

        return result
