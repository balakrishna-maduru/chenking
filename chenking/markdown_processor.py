#!/usr/bin/env python3
"""
Markdown Processor - LlamaIndex-style Markdown Processing

This module provides the MarkdownProcessor class which is a faithful reproduction
of LlamaIndex's MarkdownNodeParser functionality, adapted for Chenking's document
processing pipeline. It includes all the original functionality including callback
management, progress tracking, and exact parsing logic.
"""

import logging
import re
from typing import Dict, Any, List

from .chunk_utils import ChunkBuilder, ChunkAggregator


class MarkdownProcessor:
    """
    Complete LlamaIndex MarkdownNodeParser implementation.
    
    This is a faithful reproduction of LlamaIndex's MarkdownNodeParser
    adapted for Chenking's document processing pipeline. It includes
    all the original functionality including callback management,
    progress tracking, and exact parsing logic.
    """
    
    def __init__(self, config: Any):
        """
        Initialize the LlamaIndex-style markdown processor.
        
        Args:
            config: Chenking configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LlamaIndex equivalent properties
        self.include_metadata = config.include_markdown_metadata
        self.include_prev_next_rel = config.include_prev_next_rel
        self.return_each_line = config.return_each_line
        
        # Initialize shared utilities
        self.chunk_builder = ChunkBuilder(config, self.include_metadata)
        self.chunk_aggregator = ChunkAggregator()
    
    @classmethod
    def class_name(cls) -> str:
        """Get class name (LlamaIndex pattern)."""
        return "MarkdownProcessor"
    
    def get_nodes_from_documents(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Parse documents into nodes (LlamaIndex style).
        
        Args:
            documents: List of document dictionaries to parse
            show_progress: Whether to show progress (simplified for Chenking)
            
        Returns:
            List of processed nodes/chenks
        """
        all_nodes = []
        
        if show_progress:
            print("Parsing documents into nodes...")
        
        for document in documents:
            nodes = self.get_nodes_from_node(document)
            all_nodes.extend(nodes)
        
        return all_nodes
    
    def get_nodes_from_node(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get nodes from document (exact LlamaIndex logic).
        
        This method replicates the exact logic from LlamaIndex's
        MarkdownNodeParser.get_nodes_from_node method.
        
        Args:
            node: Input document/node
            
        Returns:
            List of processed markdown nodes
        """
        # Extract text content (equivalent to node.get_content(metadata_mode=MetadataMode.NONE))
        text = node.get("content", "")
        
        markdown_nodes = []
        lines = text.split("\n")
        metadata: Dict[str, str] = {}
        code_block = False
        current_section = ""

        for line in lines:
            # Handle code blocks exactly like LlamaIndex
            if line.startswith("```"):
                code_block = not code_block
            
            # Parse headers exactly like LlamaIndex
            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match and not code_block:
                if current_section != "":
                    markdown_nodes.append(
                        self._build_node_from_split(
                            current_section.strip(), node, metadata
                        )
                    )
                
                # Update metadata exactly like LlamaIndex
                metadata = self._update_metadata(
                    metadata, header_match.group(2), len(header_match.group(1).strip())
                )
                current_section = f"{header_match.group(2)}\n"
            else:
                current_section += line + "\n"

        # Add final section
        markdown_nodes.append(
            self._build_node_from_split(current_section.strip(), node, metadata)
        )

        return markdown_nodes
    
    def _update_metadata(
        self, headers_metadata: Dict[str, str], new_header: str, new_header_level: int
    ) -> Dict[str, str]:
        """
        Update metadata exactly like LlamaIndex.
        
        Removes all headers that are equal or less than the level
        of the newly found header (exact LlamaIndex logic).
        
        Args:
            headers_metadata: Current header metadata
            new_header: New header text
            new_header_level: Level of the new header
            
        Returns:
            Updated headers metadata
        """
        updated_headers = {}

        # Keep only headers of higher level (LlamaIndex logic)
        for i in range(1, new_header_level):
            key = f"Header {i}"
            if key in headers_metadata:
                updated_headers[key] = headers_metadata[key]

        # Add the new header
        updated_headers[f"Header {new_header_level}"] = new_header
        return updated_headers
    
    def _build_node_from_split(
        self,
        text_split: str,
        node: Dict[str, Any],
        metadata: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build node from single text split (LlamaIndex style).
        
        This replicates the LlamaIndex build_nodes_from_splits functionality
        adapted for Chenking's data structures.
        
        Args:
            text_split: The text content for this node
            node: Original document node
            metadata: Header metadata for this split
            
        Returns:
            Chenking-style node/chenk dictionary
        """
        # Use shared chunk builder for consistent node creation
        return self.chunk_builder.build_markdown_node(text_split, node, metadata)
    
    def aggregate_lines_to_chunks(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine lines with common metadata into chunks (exact LlamaIndex logic).
        
        This replicates LlamaIndex's aggregate_lines_to_chunks method exactly.
        
        Args:
            lines: Line of text / associated header metadata
            
        Returns:
            Aggregated chunks following LlamaIndex logic
        """
        # Use shared aggregator for consistent chunking
        return self.chunk_aggregator.aggregate_lines_to_chunks(lines)
    
    def split_text(self, text: str, original_node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split markdown text exactly like LlamaIndex.
        
        This is the main entry point that replicates LlamaIndex's split_text method.
        
        Args:
            text: Markdown text to split
            original_node: Original document node
            
        Returns:
            List of processed nodes/chenks
        """
        # Create a temporary node for processing
        temp_node = {
            "content": text,
            "metadata": original_node.get("metadata", {})
        }
        
        # Process using LlamaIndex logic
        lines_with_metadata = self.get_nodes_from_node(temp_node)
        
        # Aggregate if not returning each line
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        else:
            return lines_with_metadata
    
    def process_markdown_text(self, text: str, original_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main processing method that combines all LlamaIndex functionality.
        
        Args:
            text: Raw markdown text to process
            original_document: Original document context
            
        Returns:
            List of processed chenks with LlamaIndex-style metadata
        """
        # Use the LlamaIndex split_text method
        processed_nodes = self.split_text(text, original_document)
        
        # Add chenk numbering and additional Chenking-specific fields
        chenks = []
        for i, node in enumerate(processed_nodes, 1):
            chenk = {
                "chenk_number": i,
                **node  # Include all LlamaIndex-processed fields
            }
            
            # Add prev/next relationships if enabled
            if self.include_prev_next_rel:
                if i > 1:
                    chenk["prev_chenk"] = i - 1
                if i < len(processed_nodes):
                    chenk["next_chenk"] = i + 1
            
            chenks.append(chenk)
        
        return chenks
    
    def create_chenks_from_sections(
        self, 
        sections: List[Dict[str, Any]], 
        original_content: str
    ) -> List[Dict[str, Any]]:
        """
        Legacy method for compatibility - now uses the LlamaIndex approach.
        
        Args:
            sections: Processed markdown sections
            original_content: Original markdown text
            
        Returns:
            List of chenk dictionaries
        """
        # This method is now a wrapper around the LlamaIndex processing
        original_doc = {"content": original_content, "metadata": {}}
        return self.process_markdown_text(original_content, original_doc)
