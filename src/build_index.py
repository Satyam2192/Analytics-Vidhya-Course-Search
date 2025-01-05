import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument

# cheack for logs directory ->before configuring logging
Path('../logs').mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/search_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CourseMetadata:
    """Data class for course metadata"""
    title: str
    url: str
    lesson_count: Union[str, int]
    image_url: Optional[str] = None
    description: Optional[str] = None
    last_updated: str = datetime.utcnow().isoformat()
    created_by: str = "Satyam2192"

class CourseSearchSystem:
    def __init__(
        self,
        data_path: str = '../data/courses_data.json',
        index_path: str = '../embeddings',
        embed_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the Course Search System.

        Args:
            data_path (str): Path to the course data JSON file
            index_path (str): Path to store/load the FAISS index
            embed_model (str): Name of the HuggingFace embedding model
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between text chunks
        """
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create necessary directories
        self.index_path.mkdir(parents=True, exist_ok=True)
        Path('../logs').mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embed_model,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"Successfully initialized embedding model: {embed_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

        self.vector_store = None
        self.courses_data = None
        self.metadata = {}

    def load_data(self) -> List[LangchainDocument]:
        """
        Load course data and convert to Langchain documents.
        
        Returns:
            List[LangchainDocument]: List of processed documents
        """
        logger.info(f"Loading course data from {self.data_path}")
        
        try:
            df = pd.read_json(self.data_path)
            self.courses_data = df
            
            documents = []
            for _, row in df.iterrows():
                # Create course metadata
                metadata = CourseMetadata(
                    title=row['title'],
                    url=row['url'],
                    lesson_count=row['lesson_count'],
                    image_url=row['image_url'],
                    description=row.get('description', '')
                )
                
                # Create rich text content
                content = self._create_document_content(row)
                
                doc = LangchainDocument(
                    page_content=content,
                    metadata=asdict(metadata)
                )
                documents.append(doc)
                
            logger.info(f"Successfully loaded {len(documents)} course documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_document_content(self, row: pd.Series) -> str:
        """
        Create rich text content from course data.
        
        Args:
            row (pd.Series): Course data row
            
        Returns:
            str: Formatted content string
        """
        content_parts = [
            f"Title: {row['title']}",
            f"Description: {row.get('description', 'No description available')}",
            f"Curriculum: {row.get('curriculum', 'No curriculum available')}",
            f"Number of Lessons: {row['lesson_count']}",
            f"Course URL: {row['url']}"
        ]
        return "\n".join(content_parts)

    def build_index(self) -> None:
        """Build the vector store index for course search."""
        logger.info("Starting index building process...")
        
        try:
            # Load documents
            documents = self.load_data()
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            
            # Split documents into chunks
            splits = text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} text chunks for indexing")
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                splits,
                self.embedding_model
            )
            
            # Save index
            self.save_index()
            logger.info("Index built and saved successfully")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise

    def save_index(self) -> None:
        """Save the index and metadata to disk."""
        if self.vector_store is None:
            raise ValueError("Index has not been built yet")
            
        try:
            # Save FAISS index
            self.vector_store.save_local(
                folder_path=str(self.index_path)
            )
            
            # Save metadata as JSON
            metadata_path = self.index_path / "metadata.json"
            metadata = {
                'last_updated': datetime.utcnow().isoformat(),
                'created_by': "Satyam2192",
                'num_documents': len(self.courses_data) if self.courses_data is not None else 0,
                'embedding_model': self.embedding_model.model_name
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save courses data
            if self.courses_data is not None:
                courses_path = self.index_path / "courses.json"
                self.courses_data.to_json(courses_path, orient='records', indent=2)
                
            logger.info(f"Index and metadata saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def load_index(self) -> None:
        """Load the index and metadata from disk."""
        try:
            # Load vector store
            self.vector_store = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            metadata_path = self.index_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load courses data
            courses_path = self.index_path / "courses.json"
            if courses_path.exists():
                self.courses_data = pd.read_json(courses_path)
            
            logger.info("Index and metadata loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def search_courses(
        self,
        query: str,
        k: int = 5,
        similarity_cutoff: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for courses using the query string.
        
        Args:
            query: Search query string
            k: Number of results to return
            similarity_cutoff: Minimum similarity score threshold
            
        Returns:
            List[Dict[str, Any]]: List of course information dictionaries
        """
        if self.vector_store is None:
            raise ValueError("Index not loaded. Please load or build the index first.")
            
        try:
            # Perform similarity search
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            results = []
            for doc, score in docs_and_scores:
                if score >= similarity_cutoff:
                    course_info = {
                        'title': doc.metadata['title'],
                        'url': doc.metadata['url'],
                        'lesson_count': doc.metadata['lesson_count'],
                        'image_url': doc.metadata.get('image_url'),
                        'similarity_score': score,
                        'description': doc.metadata.get('description', '')
                    }
                    results.append(course_info)
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dict[str, Any]: Dictionary containing index statistics
        """
        if self.vector_store is None:
            return {"status": "Index not loaded"}
            
        return {
            "num_documents": len(self.courses_data) if self.courses_data is not None else 0,
            "last_updated": self.metadata.get('last_updated', 'Unknown'),
            "embedding_model": self.metadata.get('embedding_model', 'Unknown'),
            "created_by": self.metadata.get('created_by', 'Unknown')
        }

def main():
    """Build and test the search index."""
    logger.info("Starting main execution...")
    
    try:
        search_system = CourseSearchSystem()
        search_system.build_index()
        
        # Test search
        query = "machine learning for beginners"
        results = search_system.search_courses(query)
        
        logger.info(f"\nSearch results for: {query}")
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. {result['title']}")
            logger.info(f"   Score: {result['similarity_score']:.3f}")
            logger.info(f"   URL: {result['url']}")
            
        # Print index stats
        stats = search_system.get_index_stats()
        logger.info(f"\nIndex Statistics:\n{json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()