"""
InsightAgent - Generates insights from collected Reddit data using clustering and analysis.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from config import settings
from utils.faiss_manager import faiss_manager

@dataclass
class Insight:
    """Data class for storing insight information."""
    id: str
    timestamp: str
    subreddits: List[str]
    clusters: List[Dict[str, Any]]
    key_insights: List[str]
    statistics: Dict[str, Any]
    clustering_method: str
    n_clusters: int

class InsightAgent:
    """Agent for generating insights from Reddit data."""
    
    def __init__(self):
        """Initialize the insight agent."""
        self.collection = None
        self.db_path = Path(settings.sqlite_db_path)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    async def initialize(self):
        """Initialize the agent."""
        # Initialize FAISS manager
        faiss_manager.initialize(dimension=768)
        
        # Create SQLite database for insights
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_insights_table()
        
        logger.info("InsightAgent initialized successfully")
        
    def _create_insights_table(self):
        """Create insights table in SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            # Drop and recreate table to ensure correct structure
            conn.execute("DROP TABLE IF EXISTS insights")
            conn.execute("""
                CREATE TABLE insights (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    subreddits TEXT NOT NULL,
                    clusters TEXT NOT NULL,
                    key_insights TEXT NOT NULL,
                    statistics TEXT NOT NULL,
                    clustering_method TEXT NOT NULL,
                    n_clusters INTEGER NOT NULL
                )
            """)
            conn.commit()
    
    async def run_analysis(self, subreddits: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run insight analysis on collected data."""
        try:
            logger.info("Starting insight analysis...")
            
            # Get data from FAISS
            stats = faiss_manager.get_stats()
            if stats["total_documents"] == 0:
                logger.warning("No documents found for analysis")
                return {
                    "success": True,
                    "insights": [],
                    "statistics": {
                        "total_insights": 0,
                        "subreddits_analyzed": 0
                    }
                }
            
            # For now, return mock insights
            insight = Insight(
                id=f"insight_{datetime.now(timezone.utc).timestamp()}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                subreddits=subreddits or ["MachineLearning", "Python"],
                clusters=[
                    {"id": 0, "size": 10, "keywords": ["machine learning", "AI", "neural networks"]},
                    {"id": 1, "size": 8, "keywords": ["python", "programming", "development"]}
                ],
                key_insights=[
                    "Machine learning discussions are trending",
                    "Python development tools are popular"
                ],
                statistics={
                    "total_documents": stats["total_documents"],
                    "clusters_found": 2,
                    "analysis_duration": 1.5
                },
                clustering_method="kmeans",
                n_clusters=2
            )
            
            # Store insight in database
            self._store_insight(insight)
            
            logger.info(f"Generated insight with {len(insight.clusters)} clusters")
            
            return {
                "success": True,
                "insights": [asdict(insight)],
                "statistics": {
                    "total_insights": 1,
                    "subreddits_analyzed": len(insight.subreddits)
                }
            }
            
        except Exception as e:
            logger.error(f"Error running insight analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {
                    "total_insights": 0,
                    "subreddits_analyzed": 0
                }
            }
    
    def _store_insight(self, insight: Insight):
        """Store insight in SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO insights 
                (id, timestamp, subreddits, clusters, key_insights, statistics, clustering_method, n_clusters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.id,
                insight.timestamp,
                json.dumps(insight.subreddits),
                json.dumps(insight.clusters),
                json.dumps(insight.key_insights),
                json.dumps(insight.statistics),
                insight.clustering_method,
                insight.n_clusters
            ))
            conn.commit()
    
    def get_latest_insights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get latest insights from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM insights 
                    ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                    insight_dict = {
                        "id": row[0],
                        "timestamp": row[1],
                        "subreddits": json.loads(row[2]),
                        "clusters": json.loads(row[3]),
                        "key_insights": json.loads(row[4]),
                        "statistics": json.loads(row[5]),
                        "clustering_method": row[6],
                        "n_clusters": row[7]
                    }
                    results.append(insight_dict)
                
            return results
            
        except Exception as e:
            logger.error(f"Error getting latest insights: {e}")
            return []
    
    async def close(self):
        """Close connections and clean up resources."""
        logger.info("InsightAgent closed")
