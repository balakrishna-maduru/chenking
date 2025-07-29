#!/usr/bin/env python3
"""
Chenk - Main Document Processing Class

This module provides the core Chenk class for processing documents into structured
chunks (chenks) according to the Chenking specification. It integrates with the 
text splitter functionality to handle various document types and formats.

Input format:
{
    "id": "document_id",
    "content": "document text content",
    "metadata": {
        "title": "document_title",
        "mimetype": "text/markdown",
        "sourcemimetype": "application/pdf",
        ... other fields
    }
}

Output format:
{
    "id": "document_id",
    "content": {
        "chenks": [
            {
                "chenk_number": 1,
                "data": "chunk text",
                "start": 0,
                "words": 5,
                "char_count": 25
            },
            ...
        ]
    },
    "metadata": {
        "title": "document_title",
        "mimetype": "text/markdown",
        "sourcemimetype": "application/pdf",
        ... other fields
    }
}
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    from .text_splitter import (
        TextSplitter, 
        RecursiveCharacterTextSplitter, 
        CharacterTextSplitter,
        MarkdownTextSplitter,
        LatexTextSplitter,
        Language
    )
    from .markdown_processor import MarkdownProcessor
    from .chunk_utils import ChunkBuilder, ChunkValidator, ChunkMetrics
except ImportError:
    # Fallback for standalone usage
    from text_splitter import (
        TextSplitter, 
        RecursiveCharacterTextSplitter, 
        CharacterTextSplitter,
        MarkdownTextSplitter,
        LatexTextSplitter,
        Language
    )
    from markdown_processor import MarkdownProcessor
    from chunk_utils import ChunkBuilder, ChunkValidator, ChunkMetrics


@dataclass
class ChenkConfig:
    """Configuration for chenk processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_words_per_chenk: int = 5
    max_words_per_chenk: int = 500
    enable_smart_splitting: bool = True
    preserve_metadata: bool = True
    add_position_info: bool = True
    # LlamaIndex-style markdown processing
    include_markdown_metadata: bool = True
    include_prev_next_rel: bool = True
    return_each_line: bool = False


class Chenk:
    """
    Main class for processing documents into structured chunks (chenks).
    
    Handles various document types and formats, using appropriate text splitters
    based on the document's mimetype and content structure. Includes LlamaIndex-style
    markdown processing for enhanced document structure preservation.
    """
    
    def __init__(self, config: Optional[ChenkConfig] = None):
        """
        Initialize the Chenk processor.
        
        Args:
            config: Configuration for chenk processing
        """
        self.config = config or ChenkConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize text splitters for different formats
        self._initialize_splitters()
        
        # Initialize LlamaIndex-style markdown processor
        self.markdown_processor = MarkdownProcessor(self.config)
        
        # Initialize shared utilities
        self.chunk_builder = ChunkBuilder(self.config, self.config.preserve_metadata)
        self.chunk_validator = ChunkValidator(self.config)
        
        # Mimetype to splitter mapping
        self._splitter_mapping = {
            'text/markdown': self.markdown_splitter,
            'text/x-markdown': self.markdown_splitter,
            'application/x-tex': self.latex_splitter,
            'text/x-tex': self.latex_splitter,
            'text/plain': self.text_splitter,
            'text/html': self.recursive_splitter,
            'application/json': self.text_splitter,
            'text/csv': self.text_splitter,
            'application/rtf': self.text_splitter,
            'application/msword': self.text_splitter,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self.text_splitter,
            'application/pdf': self.recursive_splitter,  # For PDF-extracted text
        }
    
    def _initialize_splitters(self) -> None:
        """Initialize different text splitters for various content types."""
        base_kwargs = {
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'add_start_index': self.config.add_position_info,
        }
        
        # Recursive character splitter (default)
        self.recursive_splitter = RecursiveCharacterTextSplitter(**base_kwargs)
        
        # Character splitter for simple text
        self.text_splitter = CharacterTextSplitter(
            separator="\n\n",
            **base_kwargs
        )
        
        # Markdown splitter
        self.markdown_splitter = MarkdownTextSplitter(**base_kwargs)
        
        # LaTeX splitter
        self.latex_splitter = LatexTextSplitter(**base_kwargs)
        
        # Line-based splitter for structured data
        self.line_splitter = CharacterTextSplitter(
            separator="\n",
            **base_kwargs
        )
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document into chenks according to the specified format.
        
        Args:
            document: Input document with id, content, and metadata
            
        Returns:
            Processed document with structured chenks
            
        Raises:
            ValueError: If document format is invalid
        """
        # Validate input document
        self._validate_input_document(document)
        
        # Extract document components
        doc_id = document.get("id")
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        self.logger.info(f"Processing document: {doc_id}")
        
        # Check if this should use LlamaIndex-style processing
        mimetype = metadata.get("mimetype", "").lower()
        should_use_llamaindex_style = (
            self.config.include_markdown_metadata and 
            mimetype in ['text/markdown', 'text/x-markdown']
        )
        
        if should_use_llamaindex_style:
            # Use LlamaIndex-style markdown processing
            return self.process_markdown_document(document)
        
        # Determine appropriate splitter
        splitter = self._get_splitter_for_document(metadata)
        
        # Split content into chunks
        chunks = self._split_content(content, splitter)
        
        # Process chunks into chenks
        chenks = self._create_chenks(chunks, content)
        
        # Build output structure
        result = {
            "id": doc_id,
            "content": {
                "chenks": chenks
            },
            "metadata": metadata.copy() if self.config.preserve_metadata else {}
        }
        
        # Add processing metadata
        if self.config.preserve_metadata:
            result["metadata"]["processing_info"] = {
                "total_chenks": len(chenks),
                "total_words": sum(chenk["words"] for chenk in chenks),
                "total_chars": sum(chenk["char_count"] for chenk in chenks),
                "splitter_used": splitter.__class__.__name__
            }
        
        self.logger.info(f"Document {doc_id} processed into {len(chenks)} chenks")
        return result
    
    def _validate_input_document(self, document: Dict[str, Any]) -> None:
        """Validate input document structure."""
        if not isinstance(document, dict):
            raise ValueError("Document must be a dictionary")
        
        if "id" not in document:
            raise ValueError("Document must contain 'id' field")
        
        if "content" not in document:
            raise ValueError("Document must contain 'content' field")
        
        if not isinstance(document.get("content"), str):
            raise ValueError("Document content must be a string")
    
    def _get_splitter_for_document(self, metadata: Dict[str, Any]) -> TextSplitter:
        """
        Determine the appropriate text splitter based on document metadata.
        
        Args:
            metadata: Document metadata containing mimetype information
            
        Returns:
            Appropriate text splitter instance
        """
        # Primary mimetype (processed format)
        mimetype = metadata.get("mimetype", "").lower()
        
        # Source mimetype (original format)
        source_mimetype = metadata.get("sourcemimetype", "").lower()
        
        # Try to match primary mimetype first
        if mimetype in self._splitter_mapping:
            splitter = self._splitter_mapping[mimetype]
            self.logger.debug(f"Using splitter for mimetype: {mimetype}")
            return splitter
        
        # Fall back to source mimetype
        if source_mimetype in self._splitter_mapping:
            splitter = self._splitter_mapping[source_mimetype]
            self.logger.debug(f"Using splitter for source mimetype: {source_mimetype}")
            return splitter
        
        # Detect format from content if smart splitting is enabled
        if self.config.enable_smart_splitting:
            return self._detect_content_format(metadata)
        
        # Default fallback
        self.logger.debug("Using default recursive splitter")
        return self.recursive_splitter
    
    def _detect_content_format(self, metadata: Dict[str, Any]) -> TextSplitter:
        """
        Detect content format from metadata and return appropriate splitter.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Detected text splitter
        """
        # Check file extension
        title = metadata.get("title", "").lower()
        
        if title.endswith(('.md', '.markdown')):
            return self.markdown_splitter
        elif title.endswith(('.tex', '.latex')):
            return self.latex_splitter
        elif title.endswith(('.html', '.htm')):
            return self.recursive_splitter
        elif title.endswith(('.csv', '.tsv')):
            return self.line_splitter
        elif title.endswith(('.txt', '.text')):
            return self.text_splitter
        
        # Default to recursive splitter
        return self.recursive_splitter
    
    def _split_content(self, content: str, splitter: TextSplitter) -> List[str]:
        """
        Split content using the specified splitter.
        
        Args:
            content: Text content to split
            splitter: Text splitter to use
            
        Returns:
            List of text chunks
        """
        if not content.strip():
            return []
        
        try:
            chunks = splitter.split_text(content)
            
            # Filter chunks using shared validator
            filtered_chunks = self.chunk_validator.filter_chunks_by_size(chunks)
            
            # Handle large chunks that still exceed limits
            final_chunks = []
            for chunk in filtered_chunks:
                word_count = ChunkMetrics.calculate_word_count(chunk)
                if word_count > self.config.max_words_per_chenk:
                    # Further split large chunks
                    sub_chunks = self._split_large_chunk(chunk)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting content: {str(e)}")
            # Fallback to simple splitting
            return self._fallback_split(content)
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """
        Split a chunk that exceeds maximum word count.
        
        Args:
            chunk: Large chunk to split
            
        Returns:
            List of smaller chunks
        """
        # Use simple character splitter for large chunks
        simple_splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size // 2,
            chunk_overlap=self.config.chunk_overlap // 2,
            separator=" "
        )
        
        try:
            return simple_splitter.split_text(chunk)
        except Exception:
            # Final fallback: split by sentences
            sentences = re.split(r'[.!?]+', chunk)
            return [s.strip() for s in sentences if s.strip()]
    
    def _fallback_split(self, content: str) -> List[str]:
        """
        Fallback splitting method when other splitters fail.
        
        Args:
            content: Content to split
            
        Returns:
            List of chunks
        """
        # Simple paragraph-based splitting
        paragraphs = content.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            words = para.split()
            if len(words) <= self.config.max_words_per_chenk:
                chunks.append(para)
            else:
                # Split long paragraphs by sentences
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len((current_chunk + " " + sentence).split()) <= self.config.max_words_per_chenk:
                        current_chunk = (current_chunk + " " + sentence).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
        
        return chunks
    
    def _create_chenks(self, chunks: List[str], original_content: str) -> List[Dict[str, Any]]:
        """
        Create chenk objects from text chunks using shared utilities.
        
        Args:
            chunks: List of text chunks
            original_content: Original document content for position calculation
            
        Returns:
            List of chenk dictionaries
        """
        chenks = []
        
        for i, chunk in enumerate(chunks, 1):
            # Use shared chunk builder for consistent chenk creation
            chenk = self.chunk_builder.build_basic_chenk(
                text=chunk,
                original_content=original_content,
                chenk_number=i
            )
            
            # Handle position estimation for chunks that couldn't be found exactly
            if chenk["start"] == 0 and self.config.add_position_info and i > 1:
                chenk["start"] = self.chunk_builder.estimate_position_by_index(
                    chunk, original_content, i - 1, chunks
                )
            
            chenks.append(chenk)
        
        return chenks
    

    
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of input documents
            
        Returns:
            List of processed documents
        """
        results = []
        
        for doc in documents:
            try:
                result = self.process_document(doc)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing document {doc.get('id', 'unknown')}: {str(e)}")
                # Add error result
                results.append({
                    "id": doc.get("id", "unknown"),
                    "content": {"chenks": []},
                    "metadata": doc.get("metadata", {}),
                    "error": str(e)
                })
        
        return results
    
    def process_markdown_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a markdown document using LlamaIndex-style parsing.
        
        This method provides enhanced markdown processing that preserves
        document structure and header hierarchy, similar to LlamaIndex's
        MarkdownNodeParser functionality.
        
        Args:
            document: Input document with markdown content
            
        Returns:
            Processed document with markdown-aware chenks
        """
        # Validate input document
        self._validate_input_document(document)
        
        # Extract document components
        doc_id = document.get("id")
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        self.logger.info(f"Processing markdown document with LlamaIndex-style parsing: {doc_id}")
        
        # Check if this should use LlamaIndex-style processing
        mimetype = metadata.get("mimetype", "").lower()
        should_use_llamaindex_style = (
            self.config.include_markdown_metadata and 
            mimetype in ['text/markdown', 'text/x-markdown']
        )
        
        self.logger.info(f"Processing document {doc_id}: mimetype='{mimetype}', include_markdown_metadata={self.config.include_markdown_metadata}, should_use_llamaindex_style={should_use_llamaindex_style}")
        
        if should_use_llamaindex_style:
            # Use complete LlamaIndex-style markdown processing
            chenks = self.markdown_processor.process_markdown_text(content, document)
            
            # Build output structure
            result = {
                "id": doc_id,
                "content": {
                    "chenks": chenks
                },
                "metadata": metadata.copy() if self.config.preserve_metadata else {}
            }
            
            # Add processing metadata
            if self.config.preserve_metadata:
                result["metadata"]["processing_info"] = {
                    "total_chenks": len(chenks),
                    "total_words": sum(chenk["words"] for chenk in chenks),
                    "total_chars": sum(chenk["char_count"] for chenk in chenks),
                    "splitter_used": "MarkdownProcessor",
                    "markdown_sections": len(chenks),  # Each chenk is a section in LlamaIndex style
                    "processing_mode": "llamaindex_faithful",
                    "processor_class": self.markdown_processor.class_name()
                }
            
            self.logger.info(f"LlamaIndex-style processing completed for {doc_id}: {len(chenks)} chenks")
            return result
        else:
            # Fall back to regular processing
            self.logger.info(f"Using regular processing for document {doc_id}")
            return self.process_document(document)
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the processing capabilities of this Chenk instance.
        
        Returns:
            Dictionary describing available processing modes and configurations
        """
        return {
            "standard_processing": {
                "supported_mimetypes": list(self._splitter_mapping.keys()),
                "splitters_available": [
                    "RecursiveCharacterTextSplitter",
                    "CharacterTextSplitter", 
                    "MarkdownTextSplitter",
                    "LatexTextSplitter"
                ]
            },
            "llamaindex_style_processing": {
                "enabled": self.config.include_markdown_metadata,
                "supported_mimetypes": ["text/markdown", "text/x-markdown"],
                "features": [
                    "header_hierarchy_preservation",
                    "code_block_handling",
                    "metadata_aggregation",
                    "section_based_splitting"
                ]
            },
            "configuration": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "word_limits": {
                    "min": self.config.min_words_per_chenk,
                    "max": self.config.max_words_per_chenk
                },
                "features": {
                    "smart_splitting": self.config.enable_smart_splitting,
                    "metadata_preservation": self.config.preserve_metadata,
                    "position_tracking": self.config.add_position_info,
                    "markdown_metadata": self.config.include_markdown_metadata,
                    "prev_next_relations": self.config.include_prev_next_rel
                }
            }
        }


class ChenkDemo:
    """
    Demonstration class for Chenk processing capabilities.
    
    This class provides examples and demonstrations of different
    processing modes available in the Chenk system.
    """
    
    def __init__(self):
        """Initialize the demo class."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_sample_markdown_document(self) -> Dict[str, Any]:
        """
        Create a sample markdown document for testing.
        
        Returns:
            Sample markdown document in Chenking format
        """
        return {
            "id": "markdown-sample",
            "content": """# Document Title

This is the introduction paragraph with important information about the document.

## Section 1: Overview

This section contains overview information about the topic. It provides context
and background information that helps readers understand the content.

### Subsection 1.1: Technical Details

More detailed technical information in this subsection. This includes
specific implementation details and technical considerations.

```python
# Code block example that should be preserved
def hello_world():
    return "Hello, World!"

class SampleClass:
    def __init__(self):
        self.value = 42
```

### Subsection 1.2: Additional Information

Additional details in another subsection. This content complements
the technical details from the previous section.

## Section 2: Implementation

This section focuses on implementation aspects and practical considerations
for applying the concepts discussed in the overview.

### Subsection 2.1: Best Practices

Guidelines and best practices for implementation. These recommendations
are based on real-world experience and proven methodologies.

## Section 3: Conclusion

Final thoughts and conclusions about the document. This section
summarizes the key points and provides closing remarks.""",
            "metadata": {
                "title": "sample_markdown.md",
                "mimetype": "text/markdown",
                "sourcemimetype": "application/pdf",
                "author": "Chenking Demo",
                "created_date": "2025-07-29"
            }
        }
    
    def demonstrate_standard_processing(self) -> Dict[str, Any]:
        """
        Demonstrate standard Chenking processing.
        
        Returns:
            Processing results from standard mode
        """
        print("=== Standard Chenking Processing ===")
        
        # Create configuration for standard processing
        config = ChenkConfig(
            chunk_size=150,
            chunk_overlap=30,
            min_words_per_chenk=10,
            max_words_per_chenk=50,
            include_markdown_metadata=False  # Disable for standard processing
        )
        
        # Create processor and document
        processor = Chenk(config)
        document = self.create_sample_markdown_document()
        
        # Process document
        result = processor.process_document(document)
        
        # Display results
        print(f"Document ID: {result['id']}")
        print(f"Total chenks: {len(result['content']['chenks'])}")
        print(f"Splitter used: {result['metadata']['processing_info']['splitter_used']}")
        print()
        
        return result
    
    def demonstrate_llamaindex_processing(self) -> Dict[str, Any]:
        """
        Demonstrate LlamaIndex-style markdown processing.
        
        Returns:
            Processing results from LlamaIndex-style mode
        """
        print("=== LlamaIndex-Style Markdown Processing ===")
        
        # Create configuration for LlamaIndex-style processing
        config = ChenkConfig(
            chunk_size=200,
            chunk_overlap=50,
            min_words_per_chenk=5,
            max_words_per_chenk=100,
            include_markdown_metadata=True,
            include_prev_next_rel=True,
            return_each_line=False
        )
        
        # Create processor and document
        processor = Chenk(config)
        document = self.create_sample_markdown_document()
        
        # Process using LlamaIndex-style processing
        result = processor.process_markdown_document(document)
        
        # Display results
        print(f"Document ID: {result['id']}")
        print(f"Total chenks: {len(result['content']['chenks'])}")
        print(f"Processing method: {result['metadata']['processing_info']['splitter_used']}")
        print(f"Markdown sections: {result['metadata']['processing_info']['markdown_sections']}")
        print()
        
        return result
    
    def demonstrate_processing_capabilities(self) -> None:
        """Demonstrate the processing capabilities of the Chenk system."""
        print("=== Chenk Processing Capabilities ===")
        
        config = ChenkConfig()
        processor = Chenk(config)
        capabilities = processor.get_processing_capabilities()
        
        import json
        print(json.dumps(capabilities, indent=2))
        print()
    
    def compare_processing_modes(self) -> None:
        """Compare different processing modes side by side."""
        print("=== Processing Mode Comparison ===")
        
        # Get results from both modes
        standard_result = self.demonstrate_standard_processing()
        llamaindex_result = self.demonstrate_llamaindex_processing()
        
        print("Comparison Summary:")
        print(f"Standard Processing: {len(standard_result['content']['chenks'])} chenks")
        print(f"LlamaIndex Processing: {len(llamaindex_result['content']['chenks'])} chenks")
        print()
        
        # Show sample chenks from each mode
        print("Sample Chenk (Standard):")
        if standard_result['content']['chenks']:
            sample_chenk = standard_result['content']['chenks'][0]
            print(f"  Data: {sample_chenk['data'][:100]}...")
            print(f"  Words: {sample_chenk['words']}, Chars: {sample_chenk['char_count']}")
        
        print("\nSample Chenk (LlamaIndex):")
        if llamaindex_result['content']['chenks']:
            sample_chenk = llamaindex_result['content']['chenks'][0]
            print(f"  Data: {sample_chenk['data'][:100]}...")
            print(f"  Words: {sample_chenk['words']}, Chars: {sample_chenk['char_count']}")
            if 'markdown_metadata' in sample_chenk:
                print(f"  Headers: {sample_chenk['markdown_metadata']}")
        print()
    
    def run_full_demonstration(self) -> None:
        """Run the complete demonstration of Chenk capabilities."""
        print("🚀 Chenking System Demonstration")
        print("=" * 50)
        print()
        
        self.demonstrate_processing_capabilities()
        self.compare_processing_modes()
        
        print("✅ Demonstration completed successfully!")


def main() -> None:
    """Main function demonstrating modular Chenk processing."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create and run demonstration
    demo = ChenkDemo()
    demo.run_full_demonstration()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()