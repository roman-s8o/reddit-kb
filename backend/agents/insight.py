"""
InsightAgent - Clusters embeddings, extracts keywords, performs sentiment analysis,
and generates summaries from Reddit data.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import chromadb
from loguru import logger
import httpx
import re

from config import settings


@dataclass
class TopicCluster:
    """Data structure for topic clusters."""
    cluster_id: int
    name: str
    description: str
    keywords: List[str]
    document_count: int
    avg_score: float
    representative_texts: List[str]
    subreddits: List[str]
    sentiment_distribution: Dict[str, float]


@dataclass
class SentimentAnalysis:
    """Data structure for sentiment analysis results."""
    text_id: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]  # detailed scores


@dataclass
class InsightSummary:
    """Data structure for insight summaries."""
    id: str
    created_at: str
    subreddits: List[str]
    total_documents: int
    clusters: List[TopicCluster]
    overall_sentiment: Dict[str, float]
    top_keywords: List[Tuple[str, float]]
    trending_topics: List[str]
    key_insights: List[str]


class OllamaLLM:
    """Ollama LLM wrapper for text generation."""
    
    def __init__(self, model: str = None, base_url: str = None):
        """Initialize Ollama LLM client."""
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Ollama."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class InsightAgent:
    """Agent responsible for analyzing embeddings and generating insights."""
    
    def __init__(self):
        """Initialize the insight agent."""
        self.chroma_client = None
        self.collection = None
        self.llm = OllamaLLM()
        self.db_path = Path(settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing insights."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    subreddits TEXT NOT NULL,
                    total_documents INTEGER NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            # Create clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id TEXT PRIMARY KEY,
                    insight_id TEXT NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    avg_score REAL NOT NULL,
                    representative_texts TEXT NOT NULL,
                    subreddits TEXT NOT NULL,
                    sentiment_distribution TEXT NOT NULL,
                    FOREIGN KEY (insight_id) REFERENCES insights (id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("SQLite database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise
    
    async def initialize(self):
        """Initialize Chroma database connection."""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory
            )
            self.collection = self.chroma_client.get_collection(
                name=settings.chroma_collection_name
            )
            logger.info("Connected to Chroma collection for insights")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma connection: {e}")
            raise
    
    async def get_embeddings_data(
        self,
        subreddits: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        """
        Retrieve embeddings and metadata from Chroma.
        
        Returns:
            Tuple of (embeddings_array, metadata_list, document_texts)
        """
        if not self.collection:
            await self.initialize()
        
        try:
            # Build query filters
            where_clause = {}
            if subreddits:
                where_clause["subreddit"] = {"$in": subreddits}
            
            # Get data from Chroma
            results = self.collection.get(
                where=where_clause if where_clause else None,
                limit=limit,
                include=["embeddings", "metadatas", "documents"]
            )
            
            if not results["embeddings"]:
                logger.warning("No embeddings found in Chroma collection")
                return np.array([]), [], []
            
            embeddings = np.array(results["embeddings"])
            metadatas = results["metadatas"]
            documents = results["documents"]
            
            logger.info(f"Retrieved {len(embeddings)} embeddings from Chroma")
            return embeddings, metadatas, documents
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings from Chroma: {e}")
            raise
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Cluster embeddings using specified method.
        
        Args:
            embeddings: Embeddings array
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for KMeans
        
        Returns:
            Array of cluster labels
        """
        if len(embeddings) == 0:
            return np.array([])
        
        try:
            if method == "kmeans":
                # Determine optimal number of clusters if not specified
                if n_clusters is None:
                    n_clusters = min(10, max(2, len(embeddings) // 50))
                
                # Ensure we don't have more clusters than samples
                n_clusters = min(n_clusters, len(embeddings))
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score if we have enough samples
                if len(embeddings) > n_clusters:
                    silhouette = silhouette_score(embeddings, cluster_labels)
                    logger.info(f"KMeans clustering completed with silhouette score: {silhouette:.3f}")
                
            elif method == "dbscan":
                # Use DBSCAN for density-based clustering
                dbscan = DBSCAN(
                    eps=0.5,
                    min_samples=5,
                    metric='cosine'
                )
                cluster_labels = dbscan.fit_predict(embeddings)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                logger.info(f"DBSCAN clustering completed with {n_clusters} clusters")
            
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise
    
    def extract_keywords(
        self,
        texts: List[str],
        max_features: int = 100,
        ngram_range: Tuple[int, int] = (1, 3)
    ) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        if not texts:
            return []
        
        try:
            # Clean texts
            cleaned_texts = [self._clean_text_for_keywords(text) for text in texts]
            cleaned_texts = [text for text in cleaned_texts if text.strip()]
            
            if not cleaned_texts:
                return []
            
            # Extract keywords using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword-score pairs
            keywords = list(zip(feature_names, mean_scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:50]  # Return top 50 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _clean_text_for_keywords(self, text: str) -> str:
        """Clean text for keyword extraction."""
        if not text:
            return ""
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove common Reddit-specific terms
        reddit_terms = {
            'reddit', 'post', 'comment', 'upvote', 'downvote', 'karma',
            'subreddit', 'thread', 'op', 'tldr', 'edit', 'update'
        }
        
        words = text.lower().split()
        words = [word for word in words if word not in reddit_terms and len(word) > 2]
        
        return ' '.join(words)
    
    async def analyze_sentiment(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[SentimentAnalysis]:
        """Analyze sentiment using Ollama."""
        if not texts:
            return []
        
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"Analyzing sentiment for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                for j, text in enumerate(batch_texts):
                    if not text.strip():
                        continue
                    
                    # Truncate text if too long
                    if len(text) > 500:
                        text = text[:500] + "..."
                    
                    prompt = f"""
Analyze the sentiment of the following text. Respond with only a JSON object containing:
- "sentiment": one of "positive", "negative", or "neutral"
- "confidence": a number between 0 and 1
- "reasoning": brief explanation

Text: "{text}"

JSON response:"""
                    
                    try:
                        response = await self.llm.generate(prompt, max_tokens=150)
                        
                        # Try to extract JSON from response
                        json_start = response.find('{')
                        json_end = response.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_str = response[json_start:json_end]
                            result = json.loads(json_str)
                            
                            sentiment_analysis = SentimentAnalysis(
                                text_id=f"text_{i + j}",
                                sentiment=result.get("sentiment", "neutral"),
                                confidence=float(result.get("confidence", 0.5)),
                                scores={
                                    "reasoning": result.get("reasoning", "")
                                }
                            )
                            results.append(sentiment_analysis)
                        else:
                            # Fallback to neutral if parsing fails
                            results.append(SentimentAnalysis(
                                text_id=f"text_{i + j}",
                                sentiment="neutral",
                                confidence=0.5,
                                scores={"reasoning": "Failed to parse sentiment"}
                            ))
                    
                    except Exception as e:
                        logger.warning(f"Error analyzing sentiment for text {i + j}: {e}")
                        results.append(SentimentAnalysis(
                            text_id=f"text_{i + j}",
                            sentiment="neutral",
                            confidence=0.5,
                            scores={"reasoning": f"Error: {str(e)}"}
                        ))
            
            logger.info(f"Completed sentiment analysis for {len(results)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return []
    
    def create_topic_clusters(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        cluster_labels: np.ndarray,
        sentiment_results: List[SentimentAnalysis]
    ) -> List[TopicCluster]:
        """Create topic clusters from clustering results."""
        if len(cluster_labels) == 0:
            return []
        
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        # Create sentiment lookup
        sentiment_lookup = {s.text_id: s for s in sentiment_results}
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            # Get documents in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_docs = [documents[i] for i in range(len(documents)) if cluster_mask[i]]
            cluster_metadata = [metadatas[i] for i in range(len(metadatas)) if cluster_mask[i]]
            
            if not cluster_docs:
                continue
            
            # Extract keywords for this cluster
            keywords = self.extract_keywords(cluster_docs, max_features=20)
            top_keywords = [kw[0] for kw in keywords[:10]]
            
            # Calculate average score
            scores = []
            subreddits = set()
            
            for meta in cluster_metadata:
                if "score" in meta:
                    try:
                        scores.append(float(meta["score"]))
                    except (ValueError, TypeError):
                        pass
                
                if "subreddit" in meta:
                    subreddits.add(meta["subreddit"])
            
            avg_score = np.mean(scores) if scores else 0.0
            
            # Analyze sentiment distribution
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for i, is_in_cluster in enumerate(cluster_mask):
                if is_in_cluster:
                    text_id = f"text_{i}"
                    if text_id in sentiment_lookup:
                        sentiment = sentiment_lookup[text_id].sentiment
                        sentiment_counts[sentiment] += 1
            
            total_sentiments = sum(sentiment_counts.values())
            sentiment_distribution = {}
            if total_sentiments > 0:
                for sentiment, count in sentiment_counts.items():
                    sentiment_distribution[sentiment] = count / total_sentiments
            else:
                sentiment_distribution = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            
            # Get representative texts (up to 3 shortest texts)
            sorted_docs = sorted(cluster_docs, key=len)
            representative_texts = sorted_docs[:3]
            
            # Generate cluster name and description
            cluster_name = self._generate_cluster_name(top_keywords)
            cluster_description = self._generate_cluster_description(
                cluster_name, top_keywords, len(cluster_docs)
            )
            
            cluster = TopicCluster(
                cluster_id=int(cluster_id),
                name=cluster_name,
                description=cluster_description,
                keywords=top_keywords,
                document_count=len(cluster_docs),
                avg_score=avg_score,
                representative_texts=representative_texts,
                subreddits=list(subreddits),
                sentiment_distribution=sentiment_distribution
            )
            
            clusters.append(cluster)
        
        # Sort clusters by document count
        clusters.sort(key=lambda x: x.document_count, reverse=True)
        
        logger.info(f"Created {len(clusters)} topic clusters")
        return clusters
    
    def _generate_cluster_name(self, keywords: List[str]) -> str:
        """Generate a name for a topic cluster."""
        if not keywords:
            return "General Discussion"
        
        # Use top 2-3 keywords to create a name
        name_words = keywords[:3]
        return " & ".join(name_words).title()
    
    def _generate_cluster_description(
        self,
        name: str,
        keywords: List[str],
        doc_count: int
    ) -> str:
        """Generate a description for a topic cluster."""
        keyword_str = ", ".join(keywords[:5])
        return f"Topic cluster '{name}' contains {doc_count} documents discussing {keyword_str}"
    
    async def generate_key_insights(
        self,
        clusters: List[TopicCluster],
        overall_sentiment: Dict[str, float],
        top_keywords: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate key insights using LLM."""
        if not clusters:
            return ["No significant topics found in the data"]
        
        try:
            # Prepare context for LLM
            cluster_info = []
            for cluster in clusters[:5]:  # Top 5 clusters
                sentiment_desc = max(cluster.sentiment_distribution.items(), key=lambda x: x[1])
                cluster_info.append(
                    f"- {cluster.name}: {cluster.document_count} posts, "
                    f"mainly {sentiment_desc[0]} sentiment, "
                    f"keywords: {', '.join(cluster.keywords[:5])}"
                )
            
            overall_sentiment_desc = max(overall_sentiment.items(), key=lambda x: x[1])
            top_keyword_names = [kw[0] for kw in top_keywords[:10]]
            
            prompt = f"""
Based on the following Reddit data analysis, provide 3-5 key insights:

Top Topics:
{chr(10).join(cluster_info)}

Overall Sentiment: {overall_sentiment_desc[0]} ({overall_sentiment_desc[1]:.1%})

Top Keywords: {', '.join(top_keyword_names)}

Please provide concise, actionable insights about trends, sentiment patterns, and notable topics. Format as a simple list of insights, one per line starting with "- ".
"""
            
            response = await self.llm.generate(prompt, max_tokens=300)
            
            # Parse insights from response
            insights = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    insights.append(line[2:].strip())
                elif line and not line.startswith(('Based on', 'Here are', 'The analysis')):
                    insights.append(line)
            
            # Filter and limit insights
            insights = [insight for insight in insights if len(insight) > 10][:5]
            
            if not insights:
                insights = [
                    f"Most discussed topic: {clusters[0].name} with {clusters[0].document_count} posts",
                    f"Overall sentiment is {overall_sentiment_desc[0]} ({overall_sentiment_desc[1]:.1%})",
                    f"Top trending keywords: {', '.join(top_keyword_names[:5])}"
                ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return [
                f"Most discussed topic: {clusters[0].name} with {clusters[0].document_count} posts",
                f"Found {len(clusters)} distinct topic clusters",
                f"Top keywords: {', '.join([kw[0] for kw in top_keywords[:5]])}"
            ]
    
    async def run_analysis(
        self,
        subreddits: Optional[List[str]] = None,
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None
    ) -> InsightSummary:
        """
        Run the complete insight analysis pipeline.
        
        Args:
            subreddits: List of subreddits to analyze
            clustering_method: Clustering method to use
            n_clusters: Number of clusters for KMeans
        
        Returns:
            InsightSummary object with all analysis results
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize if needed
            if not self.collection:
                await self.initialize()
            
            # Step 1: Get embeddings data
            logger.info("Retrieving embeddings from Chroma...")
            embeddings, metadatas, documents = await self.get_embeddings_data(
                subreddits=subreddits
            )
            
            if len(embeddings) == 0:
                logger.warning("No embeddings found for analysis")
                return InsightSummary(
                    id=f"insight_{start_time.strftime('%Y%m%d_%H%M%S')}",
                    created_at=start_time.isoformat(),
                    subreddits=subreddits or [],
                    total_documents=0,
                    clusters=[],
                    overall_sentiment={"neutral": 1.0},
                    top_keywords=[],
                    trending_topics=[],
                    key_insights=["No data available for analysis"]
                )
            
            # Step 2: Cluster embeddings
            logger.info("Clustering embeddings...")
            cluster_labels = self.cluster_embeddings(
                embeddings,
                method=clustering_method,
                n_clusters=n_clusters
            )
            
            # Step 3: Extract keywords
            logger.info("Extracting keywords...")
            top_keywords = self.extract_keywords(documents)
            
            # Step 4: Analyze sentiment
            logger.info("Analyzing sentiment...")
            sentiment_results = await self.analyze_sentiment(documents[:100])  # Limit for speed
            
            # Step 5: Create topic clusters
            logger.info("Creating topic clusters...")
            clusters = self.create_topic_clusters(
                embeddings, metadatas, documents, cluster_labels, sentiment_results
            )
            
            # Step 6: Calculate overall sentiment
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for sentiment_result in sentiment_results:
                sentiment_counts[sentiment_result.sentiment] += 1
            
            total_sentiments = sum(sentiment_counts.values())
            overall_sentiment = {}
            if total_sentiments > 0:
                for sentiment, count in sentiment_counts.items():
                    overall_sentiment[sentiment] = count / total_sentiments
            else:
                overall_sentiment = {"neutral": 1.0}
            
            # Step 7: Generate insights
            logger.info("Generating key insights...")
            key_insights = await self.generate_key_insights(
                clusters, overall_sentiment, top_keywords
            )
            
            # Step 8: Create summary
            trending_topics = [cluster.name for cluster in clusters[:5]]
            
            summary = InsightSummary(
                id=f"insight_{start_time.strftime('%Y%m%d_%H%M%S')}",
                created_at=start_time.isoformat(),
                subreddits=subreddits or list(set(meta.get("subreddit", "") for meta in metadatas)),
                total_documents=len(documents),
                clusters=clusters,
                overall_sentiment=overall_sentiment,
                top_keywords=top_keywords[:20],
                trending_topics=trending_topics,
                key_insights=key_insights
            )
            
            # Step 9: Save to database
            await self.save_insights(summary)
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Analysis completed successfully in {duration:.2f}s")
            logger.info(f"Found {len(clusters)} topic clusters from {len(documents)} documents")
            
            return summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def save_insights(self, summary: InsightSummary):
        """Save insights to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert main insight record
            cursor.execute("""
                INSERT OR REPLACE INTO insights 
                (id, created_at, subreddits, total_documents, data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                summary.id,
                summary.created_at,
                json.dumps(summary.subreddits),
                summary.total_documents,
                json.dumps(asdict(summary))
            ))
            
            # Insert cluster records
            for cluster in summary.clusters:
                cursor.execute("""
                    INSERT OR REPLACE INTO clusters
                    (id, insight_id, cluster_id, name, description, keywords,
                     document_count, avg_score, representative_texts, subreddits,
                     sentiment_distribution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{summary.id}_cluster_{cluster.cluster_id}",
                    summary.id,
                    cluster.cluster_id,
                    cluster.name,
                    cluster.description,
                    json.dumps(cluster.keywords),
                    cluster.document_count,
                    cluster.avg_score,
                    json.dumps(cluster.representative_texts),
                    json.dumps(cluster.subreddits),
                    json.dumps(cluster.sentiment_distribution)
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved insights {summary.id} to database")
            
        except Exception as e:
            logger.error(f"Error saving insights to database: {e}")
            raise
    
    def get_latest_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest insights from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, created_at, subreddits, total_documents, data
                FROM insights
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                insight_data = json.loads(row[4])
                results.append(insight_data)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving insights from database: {e}")
            return []
    
    async def close(self):
        """Close connections and clean up resources."""
        if self.llm:
            await self.llm.close()
        logger.info("Insight agent closed")
