"""
FastAPI backend for Reddit Knowledge Base application.
Exposes LangGraph workflows as REST endpoints.
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from config import settings
from workflow.reddit_workflow import (
    reddit_workflow,
    run_data_collection_and_analysis,
    ask_reddit_question,
    get_system_status
)


# Pydantic models for request/response
class CollectionRequest(BaseModel):
    """Request model for data collection."""
    subreddits: Optional[List[str]] = Field(default=None, description="List of subreddits to collect from")
    max_posts_per_subreddit: Optional[int] = Field(default=None, description="Maximum posts per subreddit")
    time_filter: str = Field(default="day", description="Time filter for posts")
    sort_by: str = Field(default="hot", description="Sort method for posts")


class ChatRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User's question")
    subreddits: Optional[List[str]] = Field(default=None, description="Filter by specific subreddits")
    include_insights: bool = Field(default=True, description="Whether to include analysis insights")
    max_results: int = Field(default=10, description="Maximum number of results to retrieve")


class InsightRequest(BaseModel):
    """Request model for insight generation."""
    subreddits: Optional[List[str]] = Field(default=None, description="Filter by specific subreddits")
    clustering_method: str = Field(default="kmeans", description="Clustering method to use")
    n_clusters: Optional[int] = Field(default=None, description="Number of clusters for KMeans")


class TopicSummaryRequest(BaseModel):
    """Request model for topic summaries."""
    topic: str = Field(..., description="Topic to summarize")
    subreddits: Optional[List[str]] = Field(default=None, description="Filter by specific subreddits")
    max_documents: int = Field(default=20, description="Maximum documents to analyze")


# Response models
class APIResponse(BaseModel):
    """Base API response model."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Reddit Knowledge Base API...")
    try:
        await reddit_workflow.initialize_all_agents()
        logger.info("All agents initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Reddit Knowledge Base API...")
    try:
        await reddit_workflow.close_all_agents()
        logger.info("All agents closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Reddit Knowledge Base API",
    description="Multi-agent system for collecting, processing, and analyzing Reddit data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/info", response_model=APIResponse)
async def get_info():
    """Get current model information (alias for /model/info)."""
    return await get_model_info()

@app.get("/model/info", response_model=APIResponse)
async def get_model_info():
    """Get current model information."""
    try:
        return APIResponse(
            success=True,
            message="Model information retrieved successfully",
            data={
                "current_model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "ollama_base_url": settings.ollama_base_url
            }
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System status endpoint
@app.get("/status", response_model=APIResponse)
async def get_status():
    """Get system status and statistics."""
    try:
        status = await get_system_status()
        return APIResponse(
            success=True,
            message="System status retrieved successfully",
            data=status
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data collection endpoints
@app.post("/collect", response_model=APIResponse)
async def collect_data(request: CollectionRequest, background_tasks: BackgroundTasks):
    """Start data collection process."""
    try:
        # Start collection in background
        background_tasks.add_task(
            _run_collection_task,
            request.subreddits,
            request.max_posts_per_subreddit,
            request.time_filter,
            request.sort_by
        )
        
        return APIResponse(
            success=True,
            message="Data collection started in background",
            data={
                "subreddits": request.subreddits or settings.subreddit_list,
                "status": "started"
            }
        )
    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collect/sync", response_model=APIResponse)
async def collect_data_sync(request: CollectionRequest):
    """Run data collection synchronously (for testing/small datasets)."""
    try:
        result = await run_data_collection_and_analysis(
            subreddits=request.subreddits
        )
        
        if result["success"]:
            return APIResponse(
                success=True,
                message="Data collection completed successfully",
                data=result
            )
        else:
            return APIResponse(
                success=False,
                message="Data collection failed",
                error=result.get("error"),
                data=result
            )
    except Exception as e:
        logger.error(f"Error in synchronous data collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Chat endpoints
@app.post("/chat", response_model=APIResponse)
async def chat_query(request: ChatRequest):
    """Process a chat query."""
    try:
        result = await ask_reddit_question(
            question=request.query,
            subreddits=request.subreddits
        )
        
        if result["success"]:
            return APIResponse(
                success=True,
                message="Query processed successfully",
                data=result["chat_response"]
            )
        else:
            return APIResponse(
                success=False,
                message="Query processing failed",
                error=result.get("error")
            )
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggestions", response_model=APIResponse)
async def get_suggestions(query: Optional[str] = None):
    """Get chat topic suggestions."""
    try:
        if query:
            suggestions = await reddit_workflow.chatbot.suggest_related_topics(query)
        else:
            # Get general suggestions based on recent insights
            insights = reddit_workflow.insight_agent.get_latest_insights(limit=3)
            suggestions = []
            for insight in insights:
                clusters = insight.get("clusters", [])[:3]
                for cluster in clusters:
                    suggestions.append(cluster.get("name", ""))
            
            if not suggestions:
                suggestions = [
                    "What are the trending topics?",
                    "Show me recent discussions",
                    "What's the community sentiment?",
                    "Popular programming topics",
                    "Machine learning discussions"
                ]
        
        return APIResponse(
            success=True,
            message="Suggestions retrieved successfully",
            data={"suggestions": suggestions[:5]}
        )
    except Exception as e:
        logger.error(f"Error getting chat suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Topic analysis endpoints
@app.post("/topic/summary", response_model=APIResponse)
async def get_topic_summary(request: TopicSummaryRequest):
    """Get a comprehensive summary of a specific topic."""
    try:
        summary = await reddit_workflow.chatbot.get_topic_summary(
            topic=request.topic,
            subreddits=request.subreddits,
            max_documents=request.max_documents
        )
        
        return APIResponse(
            success=True,
            message="Topic summary generated successfully",
            data=summary
        )
    except Exception as e:
        logger.error(f"Error generating topic summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Insights endpoints
@app.post("/insights/generate", response_model=APIResponse)
async def generate_insights(request: InsightRequest, background_tasks: BackgroundTasks):
    """Generate insights from stored data."""
    try:
        # Start insight generation in background
        background_tasks.add_task(
            _run_insight_task,
            request.subreddits,
            request.clustering_method,
            request.n_clusters
        )
        
        return APIResponse(
            success=True,
            message="Insight generation started in background",
            data={
                "subreddits": request.subreddits,
                "clustering_method": request.clustering_method,
                "status": "started"
            }
        )
    except Exception as e:
        logger.error(f"Error starting insight generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights/latest", response_model=APIResponse)
async def get_latest_insights(limit: int = 10):
    """Get the latest generated insights."""
    try:
        insights = reddit_workflow.insight_agent.get_latest_insights(limit=limit)
        
        return APIResponse(
            success=True,
            message="Latest insights retrieved successfully",
            data={"insights": insights}
        )
    except Exception as e:
        logger.error(f"Error getting latest insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights/dashboard", response_model=APIResponse)
async def get_insights_dashboard():
    """Get dashboard data with insights overview."""
    try:
        insights = reddit_workflow.insight_agent.get_latest_insights(limit=5)
        chat_stats = reddit_workflow.chatbot.get_chat_statistics()
        
        dashboard_data = {
            "recent_insights": insights,
            "knowledge_base_stats": chat_stats,
            "available_subreddits": settings.subreddit_list,
            "last_updated": insights[0].get("created_at") if insights else None
        }
        
        return APIResponse(
            success=True,
            message="Dashboard data retrieved successfully",
            data=dashboard_data
        )
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Workflow management endpoints
@app.post("/workflow/batch", response_model=APIResponse)
async def run_batch_workflow(
    subreddits: Optional[List[str]] = None,
    background: bool = True
):
    """Run the complete batch workflow (collect → process → analyze)."""
    try:
        if background:
            # Run in background
            asyncio.create_task(_run_batch_workflow_task(subreddits))
            return APIResponse(
                success=True,
                message="Batch workflow started in background",
                data={"subreddits": subreddits or settings.subreddit_list}
            )
        else:
            # Run synchronously
            result = await run_data_collection_and_analysis(subreddits=subreddits)
            return APIResponse(
                success=result["success"],
                message="Batch workflow completed" if result["success"] else "Batch workflow failed",
                data=result,
                error=result.get("error")
            )
    except Exception as e:
        logger.error(f"Error running batch workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.get("/config", response_model=APIResponse)
async def get_configuration():
    """Get current configuration."""
    try:
        config_data = {
            "subreddits": settings.subreddit_list,
            "max_posts_per_subreddit": settings.max_posts_per_subreddit,
            "collection_interval_hours": settings.collection_interval_hours,
            "insight_generation_interval_hours": settings.insight_generation_interval_hours,
            "ollama_model": settings.ollama_model,
            "ollama_embedding_model": settings.ollama_embedding_model
        }
        
        return APIResponse(
            success=True,
            message="Configuration retrieved successfully",
            data=config_data
        )
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _run_collection_task(
    subreddits: Optional[List[str]],
    max_posts: Optional[int],
    time_filter: str,
    sort_by: str
):
    """Background task for data collection."""
    try:
        logger.info("Starting background data collection task...")
        result = await run_data_collection_and_analysis(subreddits=subreddits)
        
        if result["success"]:
            logger.info("Background data collection completed successfully")
        else:
            logger.error(f"Background data collection failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"Background data collection task error: {e}")


async def _run_insight_task(
    subreddits: Optional[List[str]],
    clustering_method: str,
    n_clusters: Optional[int]
):
    """Background task for insight generation."""
    try:
        logger.info("Starting background insight generation task...")
        insights = await reddit_workflow.insight_agent.run_analysis(
            subreddits=subreddits,
            clustering_method=clustering_method,
            n_clusters=n_clusters
        )
        logger.info(f"Background insight generation completed: {insights.id}")
    except Exception as e:
        logger.error(f"Background insight generation task error: {e}")


async def _run_batch_workflow_task(subreddits: Optional[List[str]]):
    """Background task for complete batch workflow."""
    try:
        logger.info("Starting background batch workflow task...")
        result = await run_data_collection_and_analysis(subreddits=subreddits)
        
        if result["success"]:
            logger.info("Background batch workflow completed successfully")
        else:
            logger.error(f"Background batch workflow failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"Background batch workflow task error: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
