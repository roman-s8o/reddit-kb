"""
PreprocessorAgent - Cleans, chunks, and embeds text using Ollama + LangChain.
Stores processed data in FAISS vector database.
"""
import re
import uuid
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger
import httpx
import tiktoken
import emoji
from bs4 import BeautifulSoup

from config import settings
from utils.faiss_manager import faiss_manager


@dataclass
class ProcessedDocument:
    """Data structure for processed documents."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class OllamaEmbeddings:
    """Ollama embeddings wrapper for LangChain compatibility."""
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize Ollama embeddings."""
        self.model = model or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query/document."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class TextCleaner:
    """Enhanced text cleaning utilities for Reddit content."""
    
    @staticmethod
    def remove_html(text: str) -> str:
        """Remove HTML tags and entities from text."""
        if not text:
            return ""
        
        # Use BeautifulSoup to properly parse and remove HTML
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Clean remaining HTML entities
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#x27;': "'",
            '&#x2F;': '/',
            '&nbsp;': ' ',
            '&ndash;': '–',
            '&mdash;': '—',
            '&hellip;': '...',
            '&copy;': '©',
            '&reg;': '®',
            '&trade;': '™'
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text
    
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis from text."""
        if not text:
            return ""
        
        # Use the emoji library to remove emojis
        return emoji.replace_emoji(text, replace='')
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        if not text:
            return ""
        
        # Enhanced URL pattern to catch more variations
        url_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:/[^\s]*)?'
        ]
        
        for pattern in url_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def clean_reddit_text(text: str) -> str:
        """Clean Reddit-specific text formatting."""
        if not text or text.strip() == "":
            return ""
        
        # Remove Reddit markdown
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'~~([^~]+)~~', r'\1', text)  # Strikethrough
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
        text = re.sub(r'```[^`]*```', '', text)  # Code blocks
        text = re.sub(r'^>', '', text, flags=re.MULTILINE)  # Quotes
        text = re.sub(r'^\s*&gt;', '', text, flags=re.MULTILINE)  # HTML quotes
        
        # Remove Reddit-specific patterns
        text = re.sub(r'/u/\w+', '', text)  # User mentions
        text = re.sub(r'/r/\w+', '', text)  # Subreddit mentions
        text = re.sub(r'u/\w+', '', text)  # User mentions without slash
        text = re.sub(r'r/\w+', '', text)  # Subreddit mentions without slash
        text = re.sub(r'EDIT\s*:?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'UPDATE\s*:?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'TL;?DR\s*:?', '', text, flags=re.IGNORECASE)
        
        # Remove special Reddit formatting
        text = re.sub(r'\^\^\^', '', text)  # Superscript
        text = re.sub(r'\\[nrt]', ' ', text)  # Escaped characters
        
        # Clean whitespace and special characters
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\{\}\"\']', '', text)  # Keep only basic punctuation
        text = text.strip()
        
        return text
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Apply all cleaning steps in the correct order."""
        if not text:
            return ""
        
        # Step 1: Remove HTML tags and entities
        text = cls.remove_html(text)
        
        # Step 2: Remove emojis
        text = cls.remove_emojis(text)
        
        # Step 3: Remove URLs
        text = cls.remove_urls(text)
        
        # Step 4: Clean Reddit-specific formatting
        text = cls.clean_reddit_text(text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class TokenBasedTextSplitter:
    """Token-based text splitter using tiktoken for accurate token counting."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"):
        """
        Initialize the token-based text splitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Tiktoken encoding name (cl100k_base for GPT-4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.encoding.encode(text))
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        if not text.strip():
            return []
        
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Calculate end position
            end = start + self.chunk_size
            
            # Get chunk tokens
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Try to break at sentence boundaries if possible
            if end < len(tokens):  # Not the last chunk
                # Look for sentence endings in the last 50 tokens
                last_part = chunk_text[-200:]  # Approximate last 50 tokens as chars
                sentence_endings = ['.', '!', '?', '\n']
                
                best_break = -1
                for ending in sentence_endings:
                    pos = last_part.rfind(ending)
                    if pos > best_break:
                        best_break = pos
                
                if best_break > 0:
                    # Adjust chunk to end at sentence boundary
                    chunk_text = chunk_text[:len(chunk_text) - len(last_part) + best_break + 1]
            
            chunks.append(chunk_text.strip())
            
            # Calculate next start position with overlap
            if end >= len(tokens):
                break
            
            # Move start position considering overlap
            start = end - self.chunk_overlap
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into token-based chunks."""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata.get('post_id', 'unknown')}_{i}",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "token_count": self.count_tokens(chunk)
                        }
                    )
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs


class PreprocessorAgent:
    """Agent responsible for preprocessing Reddit data for vector storage."""
    
    def __init__(self):
        """Initialize the preprocessor agent."""
        self.embeddings = OllamaEmbeddings()
        self.text_splitter = TokenBasedTextSplitter(
            chunk_size=500,  # 500 tokens per chunk
            chunk_overlap=50,  # 50 token overlap
            encoding_name="cl100k_base"  # GPT-4 compatible encoding
        )

        self.collection = None
        
        # Ensure data directory exists
        Path(settings.vector_db_directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize FAISS vector database."""
        try:
            # Initialize FAISS manager
            faiss_manager.initialize(dimension=768)  # nomic-embed-text dimension
            logger.info(f"Connected to FAISS vector database")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS database: {e}")
            raise
    
    def load_json_data(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load JSON data from CollectorAgent output file.
        
        Args:
            json_file_path: Path to the JSON file created by CollectorAgent
        
        Returns:
            Dictionary containing the loaded Reddit data
        """
        try:
            json_path = Path(json_file_path)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_file_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON data from: {json_file_path}")
            
            # Extract the actual Reddit data from the collector output format
            if 'data' in data:
                reddit_data = data['data']
                metadata = data.get('metadata', {})
                logger.info(f"Loaded data: {metadata.get('total_posts', 0)} posts, "
                          f"{metadata.get('total_comments', 0)} comments from "
                          f"{metadata.get('total_subreddits', 0)} subreddits")
                return reddit_data
            else:
                # Assume direct format if no 'data' key
                return data
                
        except Exception as e:
            logger.error(f"Error loading JSON data from {json_file_path}: {e}")
            raise
    
    def create_documents_from_posts(
        self,
        posts_data: Dict[str, List[Any]]
    ) -> List[Document]:
        """Convert Reddit posts data to LangChain documents."""
        documents = []
        
        for subreddit, posts in posts_data.items():
            for post in posts:
                # Handle both dict and object formats
                if hasattr(post, 'title'):
                    # Object format (RedditPost)
                    title = post.title or ""
                    selftext = post.selftext or ""
                    post_id = post.id or ""
                    author = post.author or ""
                    score = post.score or 0
                    upvote_ratio = getattr(post, 'upvote_ratio', 0.0) or 0.0
                    num_comments = getattr(post, 'num_comments', 0) or 0
                    created_utc = getattr(post, 'created_utc', 0) or 0
                    url = getattr(post, 'url', "") or ""
                    permalink = getattr(post, 'permalink', "") or ""
                    is_self = getattr(post, 'is_self', False) or False
                    link_flair_text = getattr(post, 'link_flair_text', "") or ""
                    comments = getattr(post, 'comments', []) or []
                else:
                    # Dict format
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")
                    post_id = post.get("id", "")
                    author = post.get("author", "")
                    score = post.get("score", 0)
                    upvote_ratio = post.get("upvote_ratio", 0.0)
                    num_comments = post.get("num_comments", 0)
                    created_utc = post.get("created_utc", 0)
                    url = post.get("url", "")
                    permalink = post.get("permalink", "")
                    is_self = post.get("is_self", False)
                    link_flair_text = post.get("link_flair_text", "")
                    comments = post.get("comments", [])
                
                # Create document for the main post
                post_content = f"Title: {title}\n\nContent: {selftext}"
                cleaned_content = TextCleaner.clean_text(post_content)
                
                if cleaned_content.strip() and len(cleaned_content) > 10:
                    doc = Document(
                        page_content=cleaned_content,
                        metadata={
                            "type": "post",
                            "post_id": post_id,
                            "subreddit": subreddit,
                            "author": author,
                            "score": score,
                            "upvote_ratio": upvote_ratio,
                            "num_comments": num_comments,
                            "created_utc": created_utc,
                            "url": url,
                            "permalink": permalink,
                            "is_self": is_self,
                            "link_flair_text": link_flair_text,
                            "title": title
                        }
                    )
                    documents.append(doc)
                
                # Create documents for comments
                for comment in comments:
                    if hasattr(comment, 'body'):
                        # Object format
                        comment_body = comment.body or ""
                        comment_id = comment.id or ""
                        comment_author = comment.author or ""
                        comment_score = comment.score or 0
                        comment_created_utc = getattr(comment, 'created_utc', 0) or 0
                        parent_id = getattr(comment, 'parent_id', "") or ""
                        comment_permalink = getattr(comment, 'permalink', "") or ""
                        is_submitter = getattr(comment, 'is_submitter', False) or False
                        depth = getattr(comment, 'depth', 0) or 0
                    else:
                        # Dict format
                        comment_body = comment.get("body", "")
                        comment_id = comment.get("id", "")
                        comment_author = comment.get("author", "")
                        comment_score = comment.get("score", 0)
                        comment_created_utc = comment.get("created_utc", 0)
                        parent_id = comment.get("parent_id", "")
                        comment_permalink = comment.get("permalink", "")
                        is_submitter = comment.get("is_submitter", False)
                        depth = comment.get("depth", 0)
                    
                    if (comment_body and 
                        comment_body not in ["[deleted]", "[removed]", "", None]):
                        
                        cleaned_comment = TextCleaner.clean_text(comment_body)
                        
                        if cleaned_comment.strip() and len(cleaned_comment) > 20:
                            comment_doc = Document(
                                page_content=cleaned_comment,
                                metadata={
                                    "type": "comment",
                                    "comment_id": comment_id,
                                    "post_id": post_id,
                                    "subreddit": subreddit,
                                    "author": comment_author,
                                    "score": comment_score,
                                    "created_utc": comment_created_utc,
                                    "parent_id": parent_id,
                                    "permalink": comment_permalink,
                                    "is_submitter": is_submitter,
                                    "depth": depth,
                                    "post_title": title
                                }
                            )
                            documents.append(comment_doc)
        
        logger.info(f"Created {len(documents)} documents from Reddit data")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into 500-token chunks."""
        logger.info(f"Chunking {len(documents)} documents into 500-token chunks...")
        
        # Use the token-based text splitter
        chunked_docs = self.text_splitter.split_documents(documents)
        
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} token-based chunks")
        return chunked_docs
    
    async def embed_documents(self, documents: List[Document]) -> List[ProcessedDocument]:
        """Generate embeddings for documents."""
        processed_docs = []
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                batch_embeddings = await self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Add empty embeddings as fallback
                all_embeddings.extend([[] for _ in batch_texts])
        
        # Create processed documents
        for doc, embedding in zip(documents, all_embeddings):
            processed_doc = ProcessedDocument(
                id=str(uuid.uuid4()),
                content=doc.page_content,
                metadata=doc.metadata,
                embedding=embedding
            )
            processed_docs.append(processed_doc)
        
        logger.info(f"Generated embeddings for {len(processed_docs)} documents")
        return processed_docs
    
    async def store_in_faiss(self, processed_docs: List[ProcessedDocument]) -> int:
        """Store processed documents in FAISS vector database."""
        # Prepare data for FAISS
        documents = []
        embeddings = []
        metadatas = []
        
        for doc in processed_docs:
            if doc.embedding:  # Only store documents with valid embeddings
                documents.append(doc.content)
                embeddings.append(doc.embedding)
                
                # Prepare metadata
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
                    else:
                        metadata[key] = str(value)
                
                # Add processing timestamp
                metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
                metadatas.append(metadata)
        
        if not documents:
            logger.warning("No valid documents to store in FAISS")
            return 0
        
        try:
            # Add documents to FAISS
            faiss_manager.add_documents(documents, embeddings, metadatas)
            
            # Save to disk
            faiss_manager.save()
            
            stored_count = len(documents)
            logger.info(f"Successfully stored {stored_count} documents in FAISS")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing documents in FAISS: {e}")
            raise
    
    async def process_from_json_file(
        self,
        json_file_path: str
    ) -> Dict[str, Any]:
        """
        Process Reddit data from JSON file (CollectorAgent output).
        
        Args:
            json_file_path: Path to JSON file created by CollectorAgent
        
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Load JSON data
            posts_data = self.load_json_data(json_file_path)
            
            # Process the data
            return await self.process_reddit_data(posts_data)
            
        except Exception as e:
            logger.error(f"Error processing data from JSON file {json_file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {"processed_documents": 0}
            }
    
    async def process_reddit_data(
        self,
        posts_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Process Reddit data through the complete pipeline.
        
        Args:
            posts_data: Dictionary of Reddit posts by subreddit
        
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize if not already done
            if not self.collection:
                await self.initialize()
            
            # Step 1: Convert to documents
            logger.info("Converting Reddit posts to documents...")
            documents = self.create_documents_from_posts(posts_data)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents created from Reddit data",
                    "statistics": {"processed_documents": 0}
                }
            
            # Step 2: Chunk documents
            logger.info("Chunking documents...")
            chunked_docs = self.chunk_documents(documents)
            
            # Step 3: Generate embeddings
            logger.info("Generating embeddings...")
            processed_docs = await self.embed_documents(chunked_docs)
            
            # Step 4: Store in FAISS
            logger.info("Storing in FAISS database...")
            stored_count = await self.store_in_faiss(processed_docs)
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Calculate statistics
            total_posts = sum(len(posts) for posts in posts_data.values())
            total_comments = sum(
                sum(len(getattr(post, 'comments', []) if hasattr(post, 'comments') else post.get('comments', [])) for post in posts)
                for posts in posts_data.values()
            )
            
            result = {
                "success": True,
                "statistics": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "total_posts": total_posts,
                    "total_comments": total_comments,
                    "created_documents": len(documents),
                    "chunked_documents": len(chunked_docs),
                    "processed_documents": len(processed_docs),
                    "stored_documents": stored_count,
                    "subreddits": list(posts_data.keys())
                }
            }
            
            logger.info(f"Processing completed successfully in {duration:.2f}s")
            logger.info(f"Processed {total_posts} posts and {total_comments} comments")
            logger.info(f"Stored {stored_count} document chunks in FAISS")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {
                    "start_time": start_time.isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS collection."""
        try:
            stats = faiss_manager.get_stats()
            
            if stats["total_documents"] > 0:
                # Get sample of metadata to understand data distribution
                sample_size = min(100, stats["total_documents"])
                
                # Analyze subreddit distribution from metadata
                subreddit_counts = {}
                type_counts = {}
                
                # Get sample metadata (we'll need to access it from faiss_manager)
                # For now, return basic stats
                return {
                    "total_documents": stats["total_documents"],
                    "dimension": stats["dimension"],
                    "is_trained": stats["is_trained"],
                    "sample_size": sample_size
                }
            else:
                return {"total_documents": 0}
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close connections and clean up resources."""
        if self.embeddings:
            await self.embeddings.close()
        logger.info("Preprocessor agent closed")


# LangGraph node function
async def preprocessor_node(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    LangGraph node function for the PreprocessorAgent.
    
    Args:
        state: Current workflow state
        config: Optional configuration
    
    Returns:
        Updated state with preprocessing results
    """
    try:
        # Extract parameters from state
        collection_output_path = state.get("collection_output_path")
        
        if not collection_output_path:
            error_msg = "No collection output path provided for preprocessing"
            state.update({
                "preprocessing_status": "failed",
                "preprocessing_error": error_msg,
                "preprocessing_timestamp": datetime.now(timezone.utc).isoformat()
            })
            return state
        
        # Create and run preprocessor agent
        preprocessor = PreprocessorAgent()
        
        # Process the JSON file from CollectorAgent
        result = await preprocessor.process_from_json_file(collection_output_path)
        
        # Update state with results
        if result["success"]:
            state.update({
                "preprocessing_status": "completed",
                "preprocessing_results": result,
                "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
                "processed_documents_count": result["statistics"]["stored_documents"]
            })
            logger.info(f"PreprocessorAgent node completed: {result['statistics']['stored_documents']} documents stored")
        else:
            state.update({
                "preprocessing_status": "failed",
                "preprocessing_error": result.get("error", "Unknown error"),
                "preprocessing_timestamp": datetime.now(timezone.utc).isoformat()
            })
            logger.error(f"PreprocessorAgent node failed: {result.get('error')}")
        
        # Clean up
        await preprocessor.close()
        
        return state
        
    except Exception as e:
        logger.error(f"PreprocessorAgent node error: {e}")
        state.update({
            "preprocessing_status": "failed",
            "preprocessing_error": str(e),
            "preprocessing_timestamp": datetime.now(timezone.utc).isoformat()
        })
        return state


# Example usage
async def main():
    """Example usage of the PreprocessorAgent."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize preprocessor
    preprocessor = PreprocessorAgent()
    
    # Example: Process data from a JSON file
    try:
        # Assuming you have a JSON file from CollectorAgent
        json_file_path = "data/raw/reddit_data_20231201_120000.json"
        
        if Path(json_file_path).exists():
            result = await preprocessor.process_from_json_file(json_file_path)
            
            if result["success"]:
                stats = result["statistics"]
                print(f"✅ Processing completed successfully!")
                print(f"📊 Statistics:")
                print(f"   - Processed {stats['total_posts']} posts and {stats['total_comments']} comments")
                print(f"   - Created {stats['created_documents']} documents")
                print(f"   - Split into {stats['chunked_documents']} chunks")
                print(f"   - Stored {stats['stored_documents']} embeddings in ChromaDB")
                print(f"   - Duration: {stats['duration_seconds']:.2f} seconds")
            else:
                print(f"❌ Processing failed: {result['error']}")
        else:
            print(f"❌ JSON file not found: {json_file_path}")
            print("Please run the CollectorAgent first to generate data.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await preprocessor.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
