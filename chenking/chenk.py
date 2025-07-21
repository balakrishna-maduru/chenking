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


class Chenk:
    """
    Main class for processing documents into structured chunks (chenks).
    
    Handles various document types and formats, using appropriate text splitters
    based on the document's mimetype and content structure.
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
            
            # Filter chunks based on configuration
            filtered_chunks = []
            for chunk in chunks:
                word_count = len(chunk.split())
                if (self.config.min_words_per_chenk <= word_count <= self.config.max_words_per_chenk):
                    filtered_chunks.append(chunk)
                elif word_count > self.config.max_words_per_chenk:
                    # Further split large chunks
                    sub_chunks = self._split_large_chunk(chunk)
                    filtered_chunks.extend(sub_chunks)
                # Skip chunks that are too small
            
            return filtered_chunks
            
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
        Create chenk objects from text chunks.
        
        Args:
            chunks: List of text chunks
            original_content: Original document content for position calculation
            
        Returns:
            List of chenk dictionaries
        """
        chenks = []
        
        for i, chunk in enumerate(chunks, 1):
            # Calculate position in original content
            start_pos = original_content.find(chunk) if self.config.add_position_info else 0
            if start_pos == -1:
                # If exact match not found, estimate position
                start_pos = self._estimate_position(chunk, original_content, i - 1, chunks)
            
            # Calculate metrics
            words = chunk.split()
            word_count = len(words)
            char_count = len(chunk)
            
            chenk = {
                "chenk_number": i,
                "data": chunk,
                "start": start_pos,
                "words": word_count,
                "char_count": char_count
            }
            
            chenks.append(chenk)
        
        return chenks
    
    def _estimate_position(self, chunk: str, content: str, chunk_index: int, all_chunks: List[str]) -> int:
        """
        Estimate the position of a chunk in the original content.
        
        Args:
            chunk: Current chunk
            content: Original content
            chunk_index: Index of current chunk
            all_chunks: All chunks
            
        Returns:
            Estimated start position
        """
        if chunk_index == 0:
            return 0
        
        # Estimate based on previous chunks
        estimated_pos = 0
        for i in range(chunk_index):
            estimated_pos += len(all_chunks[i])
        
        # Add some buffer for overlaps and separators
        estimated_pos += chunk_index * self.config.chunk_overlap
        
        return min(estimated_pos, len(content))
    
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


def main():
    """Example usage of the Chenk class."""
    # Example input document with more substantial content
    example_doc = {
        "id": "1",
        "content": "Hello world and other text. This is a sample document that will be processed into chenks. Each chenk represents a meaningful chunk of the original content. The chunking process should create multiple segments that preserve the semantic meaning of the text while maintaining appropriate size limits for downstream processing tasks.",
        "metadata": {
            "title": "hello.pdf",
            "mimetype": "text/markdown",
            "sourcemimetype": "application/pdf",
            "author": "Test Author",
            "created_date": "2025-07-19"
        }
    }
    
    # Create chenk processor with realistic settings
    config = ChenkConfig(
        chunk_size=100,         # Reasonable chunk size
        chunk_overlap=20,       # Moderate overlap
        min_words_per_chenk=5,  # Minimum words per chunk
        max_words_per_chenk=30  # Maximum words per chunk
    )
    
    chenk_processor = Chenk(config)
    
    # Process document
    result = chenk_processor.process_document(example_doc)
    
    # Print result with better formatting
    import json
    print("=== Chenk Processing Results ===")
    print(f"Document ID: {result['id']}")
    print(f"Total chenks: {len(result['content']['chenks'])}")
    print(f"Splitter used: {result['metadata']['processing_info']['splitter_used']}")
    print("\nChenks:")
    print("-" * 50)
    
    for chenk in result['content']['chenks']:
        print(f"Chenk {chenk['chenk_number']}: {chenk['words']} words, {chenk['char_count']} chars")
        print(f"  Start: {chenk['start']}")
        print(f"  Data: {chenk['data']}")
        print()
    
    print("\nFull JSON output:")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()