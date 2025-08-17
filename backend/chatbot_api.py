"""
FastAPI backend specifically for chatbot functionality.

This module provides a focused API for chat interactions with the Reddit Knowledge Base,
implementing the exact requirements:
1. Accept user query + optional subreddit filter
2. Trigger ChatbotAgent in LangGraph
3. Retrieve top-5 relevant posts from Chroma
4. Combine context + insights and generate response with Ollama
5. Return JSON: { "answer": string, "references": [post_ids], "insights": [topics] }
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config import settings
from workflow.reddit_workflow import reddit_workflow, ask_reddit_question
from agents.chatbot import ChatbotAgent
from agents.insight import InsightAgent


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User's question", min_length=1, max_length=1000)
    subreddits: Optional[List[str]] = Field(
        default=None, 
        description="Optional subreddit filter (e.g., ['Python', 'MachineLearning'])",
        max_items=10
    )


class PostReference(BaseModel):
    """Model for post references in chat response."""
    post_id: str = Field(..., description="Unique identifier for the post/comment")
    subreddit: str = Field(..., description="Source subreddit")
    title: Optional[str] = Field(None, description="Post title (if available)")
    url: Optional[str] = Field(None, description="Reddit URL (if available)")
    similarity_score: float = Field(..., description="Relevance score (0-1)")
    source_type: str = Field(..., description="Type: 'post' or 'comment'")


class TopicInsight(BaseModel):
    """Model for topic insights in chat response."""
    topic: str = Field(..., description="Topic name")
    keywords: List[str] = Field(..., description="Related keywords")
    relevance: float = Field(..., description="Relevance to query (0-1)")
    document_count: int = Field(..., description="Number of documents in topic")


class ChatResponse(BaseModel):
    """Response model for chat queries."""
    answer: str = Field(..., description="Generated response to user query")
    references: List[PostReference] = Field(..., description="Top 5 relevant Reddit posts/comments")
    insights: List[TopicInsight] = Field(..., description="Relevant topic insights")
    timestamp: str = Field(..., description="Response timestamp")
    confidence: float = Field(..., description="Response confidence (0-1)")
    query: str = Field(..., description="Original user query")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    query: Optional[str] = Field(None, description="Original query if available")


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("🚀 Starting Reddit Chatbot API...")
    try:
        await reddit_workflow.initialize_all_agents()
        logger.info("✅ All agents initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Reddit Chatbot API...")
    try:
        await reddit_workflow.close_all_agents()
        logger.info("✅ All agents closed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Reddit Knowledge Base Chatbot API",
    description="Focused API for chat interactions with Reddit data using LangGraph and Ollama",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Reddit Knowledge Base Chatbot API",
        "version": "1.0.0",
        "description": "Chat with Reddit discussions using AI",
        "endpoints": {
            "POST /chat": "Main chat endpoint",
            "GET /health": "Health check",
            "GET /status": "System status",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/status")
async def get_system_status():
    """Get system status and agent readiness."""
    try:
        # Check agent status
        status = await reddit_workflow.get_workflow_status()
        
        # Get collection stats
        collection_stats = {}
        if reddit_workflow.preprocessor.collection:
            collection_stats = reddit_workflow.preprocessor.get_collection_stats()
        
        # Get latest insights count
        insights_count = len(reddit_workflow.insight_agent.get_latest_insights(limit=1))
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "agents": status,
            "knowledge_base": collection_stats,
            "insights_available": insights_count > 0,
            "ollama_configured": {
                "base_url": settings.ollama_base_url,
                "model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def chat_with_reddit_data(request: ChatRequest):
    """
    Main chat endpoint that processes user queries against Reddit data.
    
    This endpoint:
    1. Accepts user query + optional subreddit filter
    2. Triggers ChatbotAgent through LangGraph workflow
    3. Retrieves top-5 relevant posts from Chroma
    4. Combines context + insights and generates response with Ollama
    5. Returns structured JSON with answer, references, and insights
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"🔍 Processing chat query: '{request.query[:100]}...' with subreddits: {request.subreddits}")
        
        # Validate request
        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Process query through LangGraph workflow
        workflow_result = await ask_reddit_question(
            question=request.query,
            subreddits=request.subreddits
        )
        
        if not workflow_result["success"]:
            error_msg = workflow_result.get("error", "Unknown workflow error")
            logger.error(f"❌ Workflow failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Chat processing failed: {error_msg}"
            )
        
        # Extract chat response data
        chat_data = workflow_result["chat_response"]
        if not chat_data:
            raise HTTPException(
                status_code=500,
                detail="No chat response data received from workflow"
            )
        
        # Format references (top 5 most relevant)
        references = []
        sources = chat_data.get("sources", [])[:5]  # Limit to top 5
        
        for i, source in enumerate(sources):
            metadata = source.get("metadata", {})
            
            # Extract post ID (use chunk_id or generate from metadata)
            post_id = metadata.get("chunk_id", f"post_{i}_{int(start_time.timestamp())}")
            
            # Extract URL if available
            url = None
            if "post_id" in metadata:
                url = f"https://reddit.com/comments/{metadata['post_id']}"
            elif "url" in metadata:
                url = metadata["url"]
            
            reference = PostReference(
                post_id=post_id,
                subreddit=metadata.get("subreddit", "unknown"),
                title=metadata.get("title", None),
                url=url,
                similarity_score=source.get("similarity_score", 0.0),
                source_type=source.get("source_type", "post")
            )
            references.append(reference)
        
        # Get relevant insights
        insights = await _get_relevant_insights_for_query(request.query, request.subreddits)
        
        # Create response
        response = ChatResponse(
            answer=chat_data["response"],
            references=references,
            insights=insights,
            timestamp=start_time.isoformat(),
            confidence=chat_data.get("confidence", 0.8),
            query=request.query
        )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"✅ Chat query processed successfully in {duration:.2f}s with confidence {response.confidence:.2f}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


async def _get_relevant_insights_for_query(
    query: str, 
    subreddits: Optional[List[str]] = None
) -> List[TopicInsight]:
    """
    Get relevant topic insights for the given query.
    
    Args:
        query: User query
        subreddits: Optional subreddit filter
        
    Returns:
        List of relevant topic insights
    """
    try:
        # Get latest insights from database
        insights_data = reddit_workflow.insight_agent.get_latest_insights(limit=3)
        
        if not insights_data:
            return []
        
        relevant_insights = []
        
        for insight_record in insights_data:
            clusters = insight_record.get("clusters", [])
            
            for cluster in clusters[:5]:  # Top 5 clusters per insight
                # Filter by subreddits if specified
                if subreddits:
                    cluster_subreddits = cluster.get("subreddits", [])
                    if not any(sub in cluster_subreddits for sub in subreddits):
                        continue
                
                # Calculate relevance based on keyword matching
                keywords = cluster.get("keywords", [])
                query_words = set(query.lower().split())
                keyword_words = set(" ".join(keywords).lower().split())
                
                # Simple relevance calculation
                common_words = query_words.intersection(keyword_words)
                relevance = len(common_words) / max(len(query_words), 1) if query_words else 0.0
                
                # Only include if somewhat relevant
                if relevance > 0.1 or not relevant_insights:  # Include at least one
                    insight = TopicInsight(
                        topic=cluster.get("name", "General Discussion"),
                        keywords=keywords[:5],  # Top 5 keywords
                        relevance=min(relevance, 1.0),
                        document_count=cluster.get("document_count", 0)
                    )
                    relevant_insights.append(insight)
        
        # Sort by relevance and return top 3
        relevant_insights.sort(key=lambda x: x.relevance, reverse=True)
        return relevant_insights[:3]
        
    except Exception as e:
        logger.error(f"❌ Error getting relevant insights: {e}")
        return []


@app.get("/insights/topics")
async def get_available_topics(subreddits: Optional[List[str]] = None):
    """Get available topics for suggestions."""
    try:
        insights = reddit_workflow.insight_agent.get_latest_insights(limit=5)
        
        topics = []
        for insight_record in insights:
            clusters = insight_record.get("clusters", [])
            for cluster in clusters:
                if subreddits:
                    cluster_subreddits = cluster.get("subreddits", [])
                    if not any(sub in cluster_subreddits for sub in subreddits):
                        continue
                
                topics.append({
                    "name": cluster.get("name", ""),
                    "keywords": cluster.get("keywords", [])[:3],
                    "document_count": cluster.get("document_count", 0)
                })
        
        # Sort by document count and return top 10
        topics.sort(key=lambda x: x["document_count"], reverse=True)
        return {"topics": topics[:10]}
        
    except Exception as e:
        logger.error(f"❌ Error getting available topics: {e}")
        return {"topics": []}


@app.get("/suggestions")
async def get_query_suggestions(subreddits: Optional[List[str]] = None):
    """Get suggested queries based on available data."""
    try:
        # Get recent insights to generate suggestions
        insights = reddit_workflow.insight_agent.get_latest_insights(limit=3)
        
        suggestions = []
        
        if insights:
            for insight_record in insights:
                clusters = insight_record.get("clusters", [])[:3]  # Top 3 clusters
                
                for cluster in clusters:
                    if subreddits:
                        cluster_subreddits = cluster.get("subreddits", [])
                        if not any(sub in cluster_subreddits for sub in subreddits):
                            continue
                    
                    cluster_name = cluster.get("name", "")
                    keywords = cluster.get("keywords", [])
                    
                    if cluster_name and keywords:
                        suggestions.extend([
                            f"What are people saying about {cluster_name.lower()}?",
                            f"Tell me about {keywords[0]} discussions",
                            f"What's the sentiment around {cluster_name.lower()}?"
                        ])
        
        # Add default suggestions if none generated
        if not suggestions:
            suggestions = [
                "What are the trending topics?",
                "Show me recent programming discussions",
                "What's the community sentiment about Python?",
                "Tell me about machine learning trends",
                "What are people discussing in data science?"
            ]
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:8]
        
        return {"suggestions": unique_suggestions}
        
    except Exception as e:
        logger.error(f"❌ Error getting query suggestions: {e}")
        return {
            "suggestions": [
                "What are the trending topics?",
                "Show me recent discussions",
                "What's the community sentiment?"
            ]
        }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error format."""
    return ErrorResponse(
        error=exc.detail,
        timestamp=datetime.utcnow().isoformat(),
        query=getattr(request, 'query', None) if hasattr(request, 'query') else None
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"❌ Unhandled exception in chatbot API: {exc}")
    return ErrorResponse(
        error="Internal server error occurred",
        timestamp=datetime.utcnow().isoformat(),
        query=None
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Reddit Chatbot API server...")
    
    uvicorn.run(
        "chatbot_api:app",
        host="0.0.0.0",
        port=8001,  # Different port from main API
        reload=True,
        log_level="info"
    )
