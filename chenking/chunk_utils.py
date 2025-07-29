"""
Shared utilities for chunk/chenk creation and text processing.

This module contains common functionality used by both the main Chenk class
and the MarkdownProcessor to eliminate code duplication.
"""

from typing import Dict, Any, List, Optional
import logging


class ChunkMetrics:
    """Calculate metrics for text chunks."""
    
    @staticmethod
    def calculate_word_count(text: str) -> int:
        """Calculate word count for a text chunk."""
        return len(text.split())
    
    @staticmethod
    def calculate_char_count(text: str) -> int:
        """Calculate character count for a text chunk."""
        return len(text)
    
    @staticmethod
    def find_position_in_content(chunk: str, original_content: str, add_position_info: bool = True) -> int:
        """
        Find the start position of a chunk in the original content.
        
        Args:
            chunk: Text chunk to find position for
            original_content: Original document content
            add_position_info: Whether to calculate position info
            
        Returns:
            Start position in original content (0 if not found or disabled)
        """
        if not add_position_info:
            return 0
        
        start_pos = original_content.find(chunk)
        return start_pos if start_pos != -1 else 0


class ChunkBuilder:
    """Build chenk/node objects from text chunks with consistent structure."""
    
    def __init__(self, config: Any = None, include_metadata: bool = True):
        self.config = config
        self.include_metadata = include_metadata
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_basic_chenk(
        self, 
        text: str, 
        original_content: str, 
        chenk_number: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a basic chenk object with standard metrics.
        
        Args:
            text: Chunk text content
            original_content: Original document content for position calculation
            chenk_number: Optional chenk number for sequencing
            additional_metadata: Optional additional metadata to include
            
        Returns:
            Chenk dictionary with standard structure
        """
        # Calculate metrics
        word_count = ChunkMetrics.calculate_word_count(text)
        char_count = ChunkMetrics.calculate_char_count(text)
        
        # Find position
        add_position_info = getattr(self.config, 'add_position_info', True)
        start_pos = ChunkMetrics.find_position_in_content(text, original_content, add_position_info)
        
        # Build basic chenk structure
        chenk = {
            "data": text,
            "words": word_count,
            "char_count": char_count,
            "start": start_pos,
        }
        
        # Add chenk number if provided
        if chenk_number is not None:
            chenk["chenk_number"] = chenk_number
        
        # Add metadata if enabled and provided
        if self.include_metadata and additional_metadata:
            chenk["metadata"] = additional_metadata.copy()
        
        return chenk
    
    def build_markdown_node(
        self,
        text_split: str,
        node: Dict[str, Any],
        markdown_metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Build a node specifically for markdown processing (LlamaIndex style).
        
        Args:
            text_split: The text content for this node
            node: Original document node
            markdown_metadata: Header metadata for this split
            
        Returns:
            Chenking-style node/chenk dictionary with markdown metadata
        """
        # Get original content for position calculation
        original_content = node.get("content", "")
        
        # Build basic chenk
        chenk_node = self.build_basic_chenk(text_split, original_content)
        
        # Add markdown-specific metadata if enabled
        if self.include_metadata:
            # Combine original metadata with header metadata
            original_metadata = node.get("metadata", {})
            combined_metadata = {**original_metadata, **markdown_metadata}
            chenk_node["metadata"] = combined_metadata
            chenk_node["markdown_metadata"] = markdown_metadata
        
        return chenk_node
    
    def estimate_position_by_index(
        self, 
        chunk: str, 
        content: str, 
        chunk_index: int, 
        all_chunks: List[str]
    ) -> int:
        """
        Estimate the position of a chunk when exact matching fails.
        
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
        chunk_overlap = getattr(self.config, 'chunk_overlap', 0)
        estimated_pos += chunk_index * chunk_overlap
        
        return min(estimated_pos, len(content))


class ChunkAggregator:
    """Aggregate and combine chunks based on metadata."""
    
    @staticmethod
    def aggregate_lines_to_chunks(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine lines with common metadata into chunks (LlamaIndex style).
        
        This method aggregates consecutive lines that have the same markdown_metadata
        into single chunks, updating their content and metrics accordingly.
        
        Args:
            lines: List of line dictionaries with text and metadata
            
        Returns:
            Aggregated chunks following LlamaIndex logic
        """
        aggregated_chunks: List[Dict[str, Any]] = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1].get("markdown_metadata") == line.get("markdown_metadata")
            ):
                # If the last line in the aggregated list has the same metadata
                # as the current line, append the current content to the last line's content
                aggregated_chunks[-1]["data"] += "\n" + line["data"]
                # Update metrics
                aggregated_chunks[-1]["words"] += line["words"]
                aggregated_chunks[-1]["char_count"] += len("\n") + line["char_count"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return aggregated_chunks


class ChunkValidator:
    """Validate and filter chunks based on configuration."""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def filter_chunks_by_size(self, chunks: List[str]) -> List[str]:
        """
        Filter chunks based on word count limits.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Filtered list of chunks within size limits
        """
        if not self.config:
            return chunks
        
        min_words = getattr(self.config, 'min_words_per_chenk', 0)
        max_words = getattr(self.config, 'max_words_per_chenk', float('inf'))
        
        filtered_chunks = []
        for chunk in chunks:
            word_count = ChunkMetrics.calculate_word_count(chunk)
            if min_words <= word_count <= max_words:
                filtered_chunks.append(chunk)
            elif word_count > max_words:
                self.logger.warning(f"Chunk exceeds max words ({word_count} > {max_words}), consider splitting")
                # Could add automatic splitting here if needed
                filtered_chunks.append(chunk)  # Include for now
            # Skip chunks that are too small
        
        return filtered_chunks
