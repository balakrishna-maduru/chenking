#!/usr/bin/env python3
"""
Tika Document Processor for Chenking

This script processes documents extracted by Apache Tika and runs them through
the Chenking document validation and embedding pipeline.

Input format (JSON):
{
    "id": "document_identifier",
    "content": "extracted text content from Tika",
    "metadata": {
        "file_type": "pdf",
        "file_name": "example.pdf",
        "page_count": 10,
        "author": "John Doe",
        "creation_date": "2025-01-01",
        ...
    }
}

Usage:
    python tika_document_processor.py --input document.json
    python tika_document_processor.py --input documents/ --batch
    echo '{"id":"test","content":"sample text","metadata":{}}' | python tika_document_processor.py --stdin
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Union
import logging
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chenking.processor import Processor


class TikaDocumentProcessor:
    """
    Processor for handling Tika-extracted documents through Chenking pipeline.
    """
    
    def __init__(self, api_url: str = "http://localhost:8002/chenking/embedding", **processor_kwargs):
        """
        Initialize the Tika document processor.
        
        Args:
            api_url: URL for the embedding API service
            **processor_kwargs: Additional configuration for the Chenking Processor
        """
        self.processor = Processor(api_url=api_url, **processor_kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def validate_tika_document(self, document: Dict[str, Any]) -> bool:
        """
        Validate that the document has the expected Tika format.
        
        Args:
            document: Document dictionary from Tika
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["id", "content"]
        
        for field in required_fields:
            if field not in document:
                self.logger.error(f"Missing required field: {field}")
                return False
                
        if not isinstance(document.get("content"), str):
            self.logger.error("Content field must be a string")
            return False
            
        if not document["content"].strip():
            self.logger.warning(f"Document {document.get('id', 'unknown')} has empty content")
            
        return True
        
    def enhance_document_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the document with additional metadata for Chenking processing.
        
        Args:
            document: Original Tika document
            
        Returns:
            Enhanced document with Chenking-compatible metadata
        """
        enhanced_doc = document.copy()
        
        # Extract file information from metadata
        metadata = document.get("metadata", {})
        
        # Add format information based on Tika metadata
        file_name = metadata.get("file_name", metadata.get("resourceName", ""))
        if file_name:
            file_extension = Path(file_name).suffix.lower().lstrip('.')
            enhanced_doc["format"] = file_extension or metadata.get("Content-Type", "unknown")
        else:
            enhanced_doc["format"] = metadata.get("Content-Type", "unknown")
            
        # Add title from metadata or filename
        if "title" not in enhanced_doc:
            enhanced_doc["title"] = (
                metadata.get("title") or 
                metadata.get("dc:title") or 
                metadata.get("file_name") or 
                f"Document {enhanced_doc['id']}"
            )
            
        # Add author information
        if "author" not in enhanced_doc:
            enhanced_doc["author"] = (
                metadata.get("author") or 
                metadata.get("dc:creator") or 
                metadata.get("creator") or
                "Unknown"
            )
            
        # Add processing timestamp
        enhanced_doc["tika_processed_at"] = datetime.now().isoformat()
        
        # Preserve original Tika metadata
        enhanced_doc["tika_metadata"] = metadata
        
        return enhanced_doc
        
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single Tika document through Chenking pipeline.
        
        Args:
            document: Tika-extracted document
            
        Returns:
            Processed document with validation results and embeddings
        """
        self.logger.info(f"🔍 Processing Tika document: {document.get('id', 'unknown')}")
        
        # Validate document format
        if not self.validate_tika_document(document):
            return {
                "id": document.get("id", "unknown"),
                "status": "error",
                "error": "Document validation failed",
                "original_document": document
            }
            
        try:
            # Enhance document with metadata
            enhanced_doc = self.enhance_document_metadata(document)
            
            # Process through Chenking
            result = self.processor.process(enhanced_doc)
            
            # Add Tika-specific information to result
            result["source"] = "tika"
            result["original_metadata"] = document.get("metadata", {})
            
            self.logger.info(f"✅ Successfully processed document: {document['id']}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing document {document.get('id', 'unknown')}: {str(e)}")
            return {
                "id": document.get("id", "unknown"),
                "status": "error", 
                "error": str(e),
                "original_document": document
            }
            
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple Tika documents.
        
        Args:
            documents: List of Tika-extracted documents
            
        Returns:
            List of processed documents
        """
        self.logger.info(f"📚 Processing batch of {len(documents)} documents")
        
        results = []
        successful = 0
        failed = 0
        
        for i, document in enumerate(documents, 1):
            self.logger.info(f"Processing document {i}/{len(documents)}")
            
            result = self.process_document(document)
            results.append(result)
            
            if result.get("status") == "error":
                failed += 1
            else:
                successful += 1
                
        self.logger.info(f"📊 Batch processing complete: {successful} successful, {failed} failed")
        return results
        
    def load_json_file(self, file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load JSON document(s) from file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Document or list of documents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.logger.info(f"📄 Loaded JSON from: {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
            raise
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
            
    def save_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], output_path: str):
        """
        Save processing results to JSON file.
        
        Args:
            results: Processing results
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"💾 Results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Process Tika-extracted documents through Chenking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single document
  python tika_document_processor.py --input document.json --output result.json
  
  # Process batch of documents
  python tika_document_processor.py --input documents.json --batch --output results.json
  
  # Process from stdin
  echo '{"id":"test","content":"sample text","metadata":{}}' | python tika_document_processor.py --stdin
  
  # Process directory of JSON files
  python tika_document_processor.py --input /path/to/docs/ --batch --output results.json
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", help="Input JSON file or directory")
    input_group.add_argument("--stdin", action="store_true", help="Read from stdin")
    
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process multiple documents")
    parser.add_argument("--api-url", default="http://localhost:8002/chenking/embedding", 
                       help="Embedding API URL")
    
    # Chenking configuration
    parser.add_argument("--min-word-count", type=int, default=10,
                       help="Minimum word count for documents (default: %(default)s)")
    parser.add_argument("--max-word-count", type=int, default=50,
                       help="Maximum word count for documents (default: %(default)s)")
    parser.add_argument("--enable-page-splitting", action="store_true", default=True,
                       help="Enable page-based document splitting (default: enabled)")
    parser.add_argument("--disable-page-splitting", action="store_true",
                       help="Disable page-based document splitting")
    parser.add_argument("--max-pages-to-process", type=int, default=100,
                       help="Maximum number of pages to process per document (default: %(default)s)")
    parser.add_argument("--min-words-per-page", type=int, default=5,
                       help="Minimum words required per page (default: %(default)s)")
    parser.add_argument("--enable-chenk-numbering", action="store_true", default=True,
                       help="Enable chenk numbering in results (default: enabled)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    chenking_config = {
        "min_word_count": args.min_word_count,
        "max_word_count": args.max_word_count,
        "enable_page_splitting": args.enable_page_splitting and not args.disable_page_splitting,
        "max_pages_to_process": args.max_pages_to_process,
        "min_words_per_page": args.min_words_per_page,
        "enable_chenk_numbering": args.enable_chenk_numbering,
        "enable_detailed_logging": args.verbose
    }
    
    processor = TikaDocumentProcessor(api_url=args.api_url, **chenking_config)
    
    try:
        # Load input data
        if args.stdin:
            input_data = json.load(sys.stdin)
        elif os.path.isdir(args.input):
            # Process directory of JSON files
            input_data = []
            for file_path in Path(args.input).glob("*.json"):
                file_data = processor.load_json_file(str(file_path))
                if isinstance(file_data, list):
                    input_data.extend(file_data)
                else:
                    input_data.append(file_data)
        else:
            input_data = processor.load_json_file(args.input)
        
        # Process documents
        if args.batch or isinstance(input_data, list):
            if not isinstance(input_data, list):
                input_data = [input_data]
            results = processor.process_batch(input_data)
        else:
            results = processor.process_document(input_data)
        
        # Output results
        if args.output:
            processor.save_results(results, args.output)
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        processor.logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
