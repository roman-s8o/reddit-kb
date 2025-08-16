"""
Configuration management for the Reddit Knowledge Base application.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Reddit API Configuration
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "rediit-kb-bot/1.0"
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2:7b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Chroma Database
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "reddit_posts"
    
    # SQLite Database
    sqlite_db_path: str = "./data/insights.db"
    
    # Application Configuration
    subreddits: str = "MachineLearning,Python,programming,datascience"
    max_posts_per_subreddit: int = 100
    collection_interval_hours: int = 24
    insight_generation_interval_hours: int = 6
    
    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    @property
    def subreddit_list(self) -> List[str]:
        """Get subreddits as a list."""
        return [s.strip() for s in self.subreddits.split(",") if s.strip()]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
