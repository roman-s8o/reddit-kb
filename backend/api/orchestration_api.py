"""
Orchestration API for Reddit Knowledge Base Workflow Management.

This module provides FastAPI endpoints for managing and monitoring
the LangGraph workflow orchestration and scheduling system.

Phase 4 - Orchestration implementation.
"""
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config import settings
from workflow.scheduler import (
    workflow_scheduler,
    start_scheduler,
    stop_scheduler,
    get_scheduler_status
)
from workflow.reddit_workflow import (
    reddit_workflow,
    run_data_collection_and_analysis,
    ask_reddit_question,
    get_system_status
)


# Request/Response Models
class JobTriggerRequest(BaseModel):
    """Request model for manually triggering jobs."""
    job_id: str = Field(..., description="ID of the job to trigger")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Optional job parameters")


class JobManagementRequest(BaseModel):
    """Request model for job management operations."""
    job_id: str = Field(..., description="ID of the job to manage")
    action: str = Field(..., description="Action to perform: enable, disable, remove")


class WorkflowRequest(BaseModel):
    """Request model for workflow execution."""
    workflow_type: str = Field(..., description="Type of workflow: batch or chat")
    subreddits: Optional[List[str]] = Field(default=None, description="Subreddits to process")
    query: Optional[str] = Field(default=None, description="Query for chat workflow")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters")


class SchedulerConfigRequest(BaseModel):
    """Request model for scheduler configuration updates."""
    collection_interval_hours: Optional[int] = Field(default=None, description="Collection interval in hours")
    insight_generation_interval_hours: Optional[int] = Field(default=None, description="Insight generation interval")
    enable_hourly_collection: Optional[bool] = Field(default=None, description="Enable hourly collection")
    enable_health_checks: Optional[bool] = Field(default=None, description="Enable health checks")


class APIResponse(BaseModel):
    """Standard API response model."""
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
    logger.info("🚀 Starting Orchestration API with Scheduler...")
    try:
        await start_scheduler()
        logger.info("✅ Orchestration API initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestration API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Orchestration API...")
    try:
        await stop_scheduler()
        logger.info("✅ Orchestration API shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during orchestration shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Reddit Knowledge Base Orchestration API",
    description="Workflow orchestration, scheduling, and monitoring for the Reddit Knowledge Base system",
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
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Reddit Knowledge Base Orchestration API",
        "version": "1.0.0",
        "description": "Workflow orchestration and scheduling management",
        "features": [
            "Workflow scheduling and management",
            "Job monitoring and control",
            "System health monitoring",
            "Manual workflow execution",
            "Configuration management"
        ],
        "endpoints": {
            "GET /scheduler/status": "Get scheduler status and job information",
            "POST /scheduler/jobs/trigger": "Manually trigger a scheduled job",
            "POST /scheduler/jobs/manage": "Enable, disable, or remove jobs",
            "POST /workflow/execute": "Execute workflows manually",
            "GET /system/health": "Comprehensive system health check",
            "GET /monitoring/metrics": "Get system metrics and statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "scheduler_running": workflow_scheduler.scheduler.running,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Scheduler Management Endpoints

@app.get("/scheduler/status", response_model=APIResponse)
async def get_scheduler_status_endpoint():
    """Get current scheduler status and job information."""
    try:
        status = await get_scheduler_status()
        
        return APIResponse(
            success=True,
            message="Scheduler status retrieved successfully",
            data=status
        )
    except Exception as e:
        logger.error(f"❌ Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/jobs/trigger", response_model=APIResponse)
async def trigger_job(request: JobTriggerRequest, background_tasks: BackgroundTasks):
    """Manually trigger a scheduled job."""
    try:
        job_id = request.job_id
        
        # Validate job exists
        if job_id not in workflow_scheduler.jobs_registry:
            raise HTTPException(
                status_code=404,
                detail=f"Job '{job_id}' not found"
            )
        
        # Get job function
        job = workflow_scheduler.jobs_registry[job_id]
        job_func = getattr(workflow_scheduler, job.function)
        
        # Execute job in background
        background_tasks.add_task(_execute_job_task, job_func, job_id)
        
        return APIResponse(
            success=True,
            message=f"Job '{job.name}' triggered successfully",
            data={
                "job_id": job_id,
                "job_name": job.name,
                "status": "started"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error triggering job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/jobs/manage", response_model=APIResponse)
async def manage_job(request: JobManagementRequest):
    """Enable, disable, or remove a scheduled job."""
    try:
        job_id = request.job_id
        action = request.action.lower()
        
        if job_id not in workflow_scheduler.jobs_registry:
            raise HTTPException(
                status_code=404,
                detail=f"Job '{job_id}' not found"
            )
        
        job = workflow_scheduler.jobs_registry[job_id]
        
        if action == "enable":
            await workflow_scheduler.enable_job(job_id)
            message = f"Job '{job.name}' enabled successfully"
        elif action == "disable":
            await workflow_scheduler.disable_job(job_id)
            message = f"Job '{job.name}' disabled successfully"
        elif action == "remove":
            await workflow_scheduler.remove_job(job_id)
            message = f"Job '{job.name}' removed successfully"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action '{action}'. Use: enable, disable, or remove"
            )
        
        return APIResponse(
            success=True,
            message=message,
            data={
                "job_id": job_id,
                "job_name": job.name,
                "action": action
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error managing job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scheduler/jobs", response_model=APIResponse)
async def list_jobs():
    """List all registered jobs with their status."""
    try:
        jobs_info = []
        
        for job_id, job in workflow_scheduler.jobs_registry.items():
            scheduler_job = workflow_scheduler.scheduler.get_job(job_id)
            
            job_info = {
                "id": job.id,
                "name": job.name,
                "enabled": job.enabled,
                "trigger_type": job.trigger_type,
                "trigger_config": job.trigger_config,
                "success_count": job.success_count,
                "error_count": job.error_count,
                "last_run": job.last_run,
                "next_run": scheduler_job.next_run_time.isoformat() if scheduler_job and scheduler_job.next_run_time else None,
                "last_result": job.last_result
            }
            jobs_info.append(job_info)
        
        return APIResponse(
            success=True,
            message="Jobs list retrieved successfully",
            data={
                "jobs": jobs_info,
                "total_jobs": len(jobs_info),
                "active_jobs": len([j for j in jobs_info if j["enabled"]])
            }
        )
    except Exception as e:
        logger.error(f"❌ Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Workflow Execution Endpoints

@app.post("/workflow/execute", response_model=APIResponse)
async def execute_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Execute a workflow manually."""
    try:
        workflow_type = request.workflow_type.lower()
        
        if workflow_type == "batch":
            # Execute batch workflow
            background_tasks.add_task(
                _execute_batch_workflow_task,
                request.subreddits,
                request.parameters
            )
            
            return APIResponse(
                success=True,
                message="Batch workflow execution started",
                data={
                    "workflow_type": "batch",
                    "subreddits": request.subreddits,
                    "status": "started"
                }
            )
            
        elif workflow_type == "chat":
            if not request.query:
                raise HTTPException(
                    status_code=400,
                    detail="Query is required for chat workflow"
                )
            
            # Execute chat workflow
            background_tasks.add_task(
                _execute_chat_workflow_task,
                request.query,
                request.subreddits
            )
            
            return APIResponse(
                success=True,
                message="Chat workflow execution started",
                data={
                    "workflow_type": "chat",
                    "query": request.query,
                    "subreddits": request.subreddits,
                    "status": "started"
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow type '{workflow_type}'. Use 'batch' or 'chat'"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/history", response_model=APIResponse)
async def get_workflow_history(limit: int = 10):
    """Get recent workflow execution history."""
    try:
        # Get execution history from job results
        history = []
        
        for job_id, job in workflow_scheduler.jobs_registry.items():
            if job.last_result:
                history.append({
                    "job_id": job_id,
                    "job_name": job.name,
                    "last_run": job.last_run,
                    "result": job.last_result,
                    "success_count": job.success_count,
                    "error_count": job.error_count
                })
        
        # Sort by last run time
        history.sort(key=lambda x: x["last_run"] or "", reverse=True)
        
        return APIResponse(
            success=True,
            message="Workflow history retrieved successfully",
            data={
                "history": history[:limit],
                "total_entries": len(history)
            }
        )
    except Exception as e:
        logger.error(f"❌ Error getting workflow history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System Monitoring Endpoints

@app.get("/system/health", response_model=APIResponse)
async def comprehensive_health_check():
    """Perform comprehensive system health check."""
    try:
        # Get system status
        system_status = await get_system_status()
        
        # Get scheduler status
        scheduler_status = await get_scheduler_status()
        
        # Perform additional health checks
        health_results = {
            "system_status": system_status,
            "scheduler_status": {
                "running": scheduler_status["running"],
                "total_jobs": scheduler_status["total_jobs"],
                "active_jobs": scheduler_status["active_jobs"],
                "uptime_seconds": scheduler_status["uptime_seconds"]
            },
            "health_checks": {
                "agents_ready": _check_agents_health(system_status),
                "databases_connected": _check_databases_health(system_status),
                "scheduler_healthy": scheduler_status["running"],
                "recent_errors": _check_recent_errors()
            }
        }
        
        # Determine overall health
        all_checks = health_results["health_checks"]
        overall_healthy = all(all_checks.values())
        
        return APIResponse(
            success=overall_healthy,
            message="System health check completed" if overall_healthy else "System health issues detected",
            data=health_results
        )
    except Exception as e:
        logger.error(f"❌ Error performing health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/metrics", response_model=APIResponse)
async def get_system_metrics():
    """Get system metrics and statistics."""
    try:
        scheduler_status = await get_scheduler_status()
        system_status = await get_system_status()
        
        # Calculate metrics
        total_successes = sum(job.get("success_count", 0) for job in scheduler_status["jobs"])
        total_errors = sum(job.get("error_count", 0) for job in scheduler_status["jobs"])
        
        metrics = {
            "scheduler_metrics": {
                "uptime_seconds": scheduler_status["uptime_seconds"],
                "total_jobs": scheduler_status["total_jobs"],
                "active_jobs": scheduler_status["active_jobs"],
                "total_executions": total_successes + total_errors,
                "success_rate": total_successes / max(total_successes + total_errors, 1) * 100
            },
            "system_metrics": {
                "knowledge_base_stats": system_status.get("agents", {}).get("preprocessor", {}).get("collection_stats", {}),
                "latest_insights": system_status.get("agents", {}).get("insight_agent", {}).get("latest_insights", 0),
                "chat_statistics": system_status.get("agents", {}).get("chatbot", {}).get("knowledge_base", {})
            },
            "job_metrics": [
                {
                    "job_id": job["id"],
                    "job_name": job["name"],
                    "success_count": job["success_count"],
                    "error_count": job["error_count"],
                    "success_rate": job["success_count"] / max(job["success_count"] + job["error_count"], 1) * 100
                }
                for job in scheduler_status["jobs"]
            ]
        }
        
        return APIResponse(
            success=True,
            message="System metrics retrieved successfully",
            data=metrics
        )
    except Exception as e:
        logger.error(f"❌ Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/config", response_model=APIResponse)
async def update_scheduler_config(request: SchedulerConfigRequest):
    """Update scheduler configuration."""
    try:
        changes = []
        
        # Note: This is a simplified implementation
        # In a production system, you'd want to persist these changes
        
        if request.enable_hourly_collection is not None:
            if request.enable_hourly_collection:
                if "hourly_data_collection" not in workflow_scheduler.jobs_registry:
                    await workflow_scheduler.add_job(
                        job_id="hourly_data_collection",
                        name="Hourly Data Collection",
                        func=workflow_scheduler._run_hourly_collection,
                        trigger_type="interval",
                        trigger_config={"hours": 1},
                        enabled=True
                    )
                    changes.append("Enabled hourly data collection")
                else:
                    await workflow_scheduler.enable_job("hourly_data_collection")
                    changes.append("Enabled hourly data collection")
            else:
                await workflow_scheduler.disable_job("hourly_data_collection")
                changes.append("Disabled hourly data collection")
        
        if request.enable_health_checks is not None:
            if request.enable_health_checks:
                await workflow_scheduler.enable_job("health_check")
                changes.append("Enabled health checks")
            else:
                await workflow_scheduler.disable_job("health_check")
                changes.append("Disabled health checks")
        
        return APIResponse(
            success=True,
            message="Scheduler configuration updated successfully",
            data={
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"❌ Error updating scheduler config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _execute_job_task(job_func, job_id: str):
    """Background task for job execution."""
    try:
        logger.info(f"🔄 Executing job manually: {job_id}")
        result = await job_func()
        logger.info(f"✅ Manual job execution completed: {job_id}")
        return result
    except Exception as e:
        logger.error(f"❌ Manual job execution failed: {job_id} - {e}")


async def _execute_batch_workflow_task(subreddits: Optional[List[str]], parameters: Optional[Dict[str, Any]]):
    """Background task for batch workflow execution."""
    try:
        logger.info("🔄 Executing batch workflow manually...")
        result = await run_data_collection_and_analysis(subreddits=subreddits)
        logger.info("✅ Manual batch workflow completed")
        return result
    except Exception as e:
        logger.error(f"❌ Manual batch workflow failed: {e}")


async def _execute_chat_workflow_task(query: str, subreddits: Optional[List[str]]):
    """Background task for chat workflow execution."""
    try:
        logger.info(f"🔄 Executing chat workflow manually: {query[:50]}...")
        result = await ask_reddit_question(question=query, subreddits=subreddits)
        logger.info("✅ Manual chat workflow completed")
        return result
    except Exception as e:
        logger.error(f"❌ Manual chat workflow failed: {e}")


# Helper functions
def _check_agents_health(system_status: Dict[str, Any]) -> bool:
    """Check if all agents are healthy."""
    agents = system_status.get("agents", {})
    
    return all([
        agents.get("collector", {}).get("status") == "ready",
        agents.get("preprocessor", {}).get("status") == "ready",
        agents.get("insight_agent", {}).get("status") == "ready",
        agents.get("chatbot", {}).get("status") == "ready"
    ])


def _check_databases_health(system_status: Dict[str, Any]) -> bool:
    """Check if databases are connected."""
    agents = system_status.get("agents", {})
    
    return all([
        agents.get("preprocessor", {}).get("chroma_connected", False),
        agents.get("insight_agent", {}).get("database_connected", False)
    ])


def _check_recent_errors() -> int:
    """Check for recent errors in jobs."""
    recent_errors = 0
    
    for job in workflow_scheduler.jobs_registry.values():
        if job.last_result and not job.last_result.get("success", True):
            recent_errors += 1
    
    return recent_errors


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return APIResponse(
        success=False,
        message=exc.detail,
        timestamp=datetime.utcnow().isoformat()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"❌ Unhandled exception in orchestration API: {exc}")
    return APIResponse(
        success=False,
        message="Internal server error occurred",
        error=str(exc),
        timestamp=datetime.utcnow().isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Reddit Knowledge Base Orchestration API...")
    
    uvicorn.run(
        "orchestration_api:app",
        host="0.0.0.0",
        port=8002,  # Different port from main API and chatbot API
        reload=True,
        log_level="info"
    )
