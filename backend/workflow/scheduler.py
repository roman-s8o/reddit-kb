"""
Workflow Scheduler for Reddit Knowledge Base.

This module implements periodic scheduling for:
- Data collection and processing (daily/hourly)
- Insight generation (every 6 hours)
- System maintenance tasks
- Workflow monitoring and health checks

Phase 4 - Orchestration implementation.
"""
import asyncio
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from loguru import logger

from config import settings
from workflow.reddit_workflow import (
    reddit_workflow,
    run_data_collection_and_analysis,
    get_system_status
)


@dataclass
class ScheduledJob:
    """Data structure for scheduled jobs."""
    id: str
    name: str
    function: str
    trigger_type: str  # 'cron' or 'interval'
    trigger_config: Dict[str, Any]
    enabled: bool
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    success_count: int = 0
    error_count: int = 0
    last_result: Optional[Dict[str, Any]] = None


@dataclass
class SchedulerStatus:
    """Data structure for scheduler status."""
    running: bool
    jobs: List[ScheduledJob]
    total_jobs: int
    active_jobs: int
    last_health_check: str
    uptime_seconds: float
    system_status: Dict[str, Any]


class WorkflowScheduler:
    """Scheduler for Reddit Knowledge Base workflows."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.scheduler = AsyncIOScheduler(timezone='UTC')
        self.start_time = datetime.now(timezone.utc)
        self.jobs_registry: Dict[str, ScheduledJob] = {}
        self.status_file = Path("./data/scheduler_status.json")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        self.scheduler.add_listener(self._job_missed, EVENT_JOB_MISSED)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down scheduler...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize the scheduler and register jobs."""
        logger.info("🔧 Initializing Workflow Scheduler...")
        
        try:
            # Initialize workflow agents
            await reddit_workflow.initialize_all_agents()
            
            # Register scheduled jobs
            await self._register_jobs()
            
            # Load previous status if available
            await self._load_status()
            
            logger.info("✅ Workflow Scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize scheduler: {e}")
            raise
    
    async def _register_jobs(self):
        """Register all scheduled jobs."""
        logger.info("📋 Registering scheduled jobs...")
        
        # Job 1: Daily data collection and analysis
        await self.add_job(
            job_id="daily_data_collection",
            name="Daily Data Collection & Analysis",
            func=self._run_daily_collection,
            trigger_type="cron",
            trigger_config={"hour": 2, "minute": 0},  # 2:00 AM UTC daily
            enabled=True
        )
        
        # Job 2: Hourly data collection (if configured)
        if settings.collection_interval_hours <= 1:
            await self.add_job(
                job_id="hourly_data_collection",
                name="Hourly Data Collection",
                func=self._run_hourly_collection,
                trigger_type="interval",
                trigger_config={"hours": 1},
                enabled=True
            )
        
        # Job 3: Insight generation every 6 hours
        await self.add_job(
            job_id="insight_generation",
            name="Insight Generation",
            func=self._run_insight_generation,
            trigger_type="interval",
            trigger_config={"hours": settings.insight_generation_interval_hours},
            enabled=True
        )
        
        # Job 4: System health check every 15 minutes
        await self.add_job(
            job_id="health_check",
            name="System Health Check",
            func=self._run_health_check,
            trigger_type="interval",
            trigger_config={"minutes": 15},
            enabled=True
        )
        
        # Job 5: Status persistence every 5 minutes
        await self.add_job(
            job_id="status_persistence",
            name="Status Persistence",
            func=self._persist_status,
            trigger_type="interval",
            trigger_config={"minutes": 5},
            enabled=True
        )
        
        # Job 6: Weekly cleanup (optional)
        await self.add_job(
            job_id="weekly_cleanup",
            name="Weekly System Cleanup",
            func=self._run_weekly_cleanup,
            trigger_type="cron",
            trigger_config={"day_of_week": 0, "hour": 3, "minute": 0},  # Sunday 3:00 AM
            enabled=False  # Disabled by default
        )
        
        logger.info(f"✅ Registered {len(self.jobs_registry)} scheduled jobs")
    
    async def add_job(
        self,
        job_id: str,
        name: str,
        func: Callable,
        trigger_type: str,
        trigger_config: Dict[str, Any],
        enabled: bool = True
    ):
        """Add a scheduled job."""
        try:
            # Create job registry entry
            scheduled_job = ScheduledJob(
                id=job_id,
                name=name,
                function=func.__name__,
                trigger_type=trigger_type,
                trigger_config=trigger_config,
                enabled=enabled
            )
            
            self.jobs_registry[job_id] = scheduled_job
            
            if enabled:
                # Add to scheduler
                if trigger_type == "cron":
                    trigger = CronTrigger(**trigger_config, timezone='UTC')
                elif trigger_type == "interval":
                    trigger = IntervalTrigger(**trigger_config, timezone='UTC')
                else:
                    raise ValueError(f"Unknown trigger type: {trigger_type}")
                
                self.scheduler.add_job(
                    func=func,
                    trigger=trigger,
                    id=job_id,
                    name=name,
                    replace_existing=True
                )
                
                logger.info(f"✅ Added job: {name} ({job_id})")
            else:
                logger.info(f"📝 Registered job (disabled): {name} ({job_id})")
                
        except Exception as e:
            logger.error(f"❌ Failed to add job {job_id}: {e}")
            raise
    
    async def remove_job(self, job_id: str):
        """Remove a scheduled job."""
        try:
            if job_id in self.jobs_registry:
                del self.jobs_registry[job_id]
            
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
            
            logger.info(f"✅ Removed job: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to remove job {job_id}: {e}")
    
    async def enable_job(self, job_id: str):
        """Enable a scheduled job."""
        if job_id not in self.jobs_registry:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs_registry[job_id]
        job.enabled = True
        
        # Add to scheduler if not already there
        if not self.scheduler.get_job(job_id):
            func = getattr(self, job.function)
            
            if job.trigger_type == "cron":
                trigger = CronTrigger(**job.trigger_config, timezone='UTC')
            else:
                trigger = IntervalTrigger(**job.trigger_config, timezone='UTC')
            
            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                name=job.name,
                replace_existing=True
            )
        
        logger.info(f"✅ Enabled job: {job.name}")
    
    async def disable_job(self, job_id: str):
        """Disable a scheduled job."""
        if job_id not in self.jobs_registry:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs_registry[job_id]
        job.enabled = False
        
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        logger.info(f"✅ Disabled job: {job.name}")
    
    def start(self):
        """Start the scheduler."""
        logger.info("🚀 Starting Workflow Scheduler...")
        self.scheduler.start()
        logger.info("✅ Scheduler started successfully")
    
    async def shutdown(self):
        """Shutdown the scheduler gracefully."""
        logger.info("🔄 Shutting down Workflow Scheduler...")
        
        try:
            # Persist final status
            await self._persist_status()
            
            # Shutdown scheduler
            self.scheduler.shutdown(wait=True)
            
            # Close workflow agents
            await reddit_workflow.close_all_agents()
            
            logger.info("✅ Scheduler shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during scheduler shutdown: {e}")
    
    async def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        system_status = await get_system_status()
        
        # Update job info from scheduler
        for job_id, job in self.jobs_registry.items():
            scheduler_job = self.scheduler.get_job(job_id)
            if scheduler_job:
                job.next_run = scheduler_job.next_run_time.isoformat() if scheduler_job.next_run_time else None
        
        return SchedulerStatus(
            running=self.scheduler.running,
            jobs=list(self.jobs_registry.values()),
            total_jobs=len(self.jobs_registry),
            active_jobs=len([j for j in self.jobs_registry.values() if j.enabled]),
            last_health_check=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime,
            system_status=system_status
        )
    
    # Job execution methods
    async def _run_daily_collection(self):
        """Run daily data collection and analysis."""
        logger.info("📊 Starting daily data collection and analysis...")
        
        try:
            result = await run_data_collection_and_analysis()
            
            if result["success"]:
                logger.info("✅ Daily data collection completed successfully")
                return {"success": True, "message": "Daily collection completed"}
            else:
                logger.error(f"❌ Daily data collection failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            logger.error(f"❌ Daily data collection error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_hourly_collection(self):
        """Run hourly data collection."""
        logger.info("⏱️ Starting hourly data collection...")
        
        try:
            # Run collection only (no full analysis)
            result = await reddit_workflow.run_batch_workflow()
            
            if result["success"]:
                logger.info("✅ Hourly data collection completed")
                return {"success": True, "message": "Hourly collection completed"}
            else:
                logger.error(f"❌ Hourly collection failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            logger.error(f"❌ Hourly collection error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_insight_generation(self):
        """Run insight generation."""
        logger.info("💡 Starting insight generation...")
        
        try:
            
            logger.info(f"✅ Insight generation completed: {insights.id}")
            return {
                "success": True,
                "message": f"Generated insights: {insights.id}",
                "insights_id": insights.id,
                "clusters": len(insights.clusters),
                "documents": insights.total_documents
            }
            
        except Exception as e:
            logger.error(f"❌ Insight generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_health_check(self):
        """Run system health check."""
        try:
            status = await get_system_status()
            
            # Check critical components
            issues = []
            
            if not status.get("agents", {}).get("collector", {}).get("status") == "ready":
                issues.append("Collector agent not ready")
            
            if not status.get("agents", {}).get("preprocessor", {}).get("vector_db_connected"):
                issues.append("Vector database not connected")
            
                issues.append("SQLite database not connected")
            
            if issues:
                logger.warning(f"⚠️ Health check issues: {', '.join(issues)}")
                return {"success": False, "issues": issues}
            else:
                return {"success": True, "message": "All systems healthy"}
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _persist_status(self):
        """Persist scheduler status to file."""
        try:
            status = await self.get_status()
            status_dict = asdict(status)
            
            with open(self.status_file, 'w') as f:
                json.dump(status_dict, f, indent=2, default=str)
            
            return {"success": True, "message": "Status persisted"}
            
        except Exception as e:
            logger.error(f"❌ Status persistence error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_weekly_cleanup(self):
        """Run weekly system cleanup."""
        logger.info("🧹 Starting weekly system cleanup...")
        
        try:
            cleanup_results = []
            
            # Clean old log files (older than 30 days)
            log_dir = Path("./logs")
            if log_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for log_file in log_dir.glob("*.log.*"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        cleanup_results.append(f"Removed old log: {log_file.name}")
            
            # Clean old status files
            status_files = Path("./data").glob("scheduler_status_*.json")
            for status_file in status_files:
                if status_file.stat().st_mtime < (datetime.now() - timedelta(days=7)).timestamp():
                    status_file.unlink()
                    cleanup_results.append(f"Removed old status: {status_file.name}")
            
            logger.info(f"✅ Weekly cleanup completed: {len(cleanup_results)} items cleaned")
            return {"success": True, "cleaned_items": cleanup_results}
            
        except Exception as e:
            logger.error(f"❌ Weekly cleanup error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _load_status(self):
        """Load previous scheduler status."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                
                # Restore job statistics
                for job_data in status_data.get("jobs", []):
                    job_id = job_data["id"]
                    if job_id in self.jobs_registry:
                        job = self.jobs_registry[job_id]
                        job.success_count = job_data.get("success_count", 0)
                        job.error_count = job_data.get("error_count", 0)
                        job.last_run = job_data.get("last_run")
                        job.last_result = job_data.get("last_result")
                
                logger.info("✅ Previous scheduler status loaded")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load previous status: {e}")
    
    # Event handlers
    def _job_executed(self, event):
        """Handle job execution events."""
        job_id = event.job_id
        
        if job_id in self.jobs_registry:
            job = self.jobs_registry[job_id]
            job.success_count += 1
            job.last_run = datetime.now(timezone.utc).isoformat()
            job.last_result = {"success": True, "executed_at": job.last_run}
            
            logger.info(f"✅ Job executed successfully: {job.name}")
    
    def _job_error(self, event):
        """Handle job error events."""
        job_id = event.job_id
        exception = event.exception
        
        if job_id in self.jobs_registry:
            job = self.jobs_registry[job_id]
            job.error_count += 1
            job.last_run = datetime.now(timezone.utc).isoformat()
            job.last_result = {
                "success": False,
                "error": str(exception),
                "executed_at": job.last_run
            }
            
            logger.error(f"❌ Job failed: {job.name} - {exception}")
    
    def _job_missed(self, event):
        """Handle missed job events."""
        job_id = event.job_id
        
        if job_id in self.jobs_registry:
            job = self.jobs_registry[job_id]
            logger.warning(f"⚠️ Job missed: {job.name}")


# Global scheduler instance
workflow_scheduler = WorkflowScheduler()


# Convenience functions
async def start_scheduler():
    """Start the workflow scheduler."""
    await workflow_scheduler.initialize()
    workflow_scheduler.start()
    return workflow_scheduler


async def stop_scheduler():
    """Stop the workflow scheduler."""
    await workflow_scheduler.shutdown()


async def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status."""
    status = await workflow_scheduler.get_status()
    return asdict(status)


# Main function for running as standalone service
async def main():
    """Main function for running scheduler as standalone service."""
    logger.info("🚀 Starting Reddit Knowledge Base Scheduler Service")
    
    try:
        # Start scheduler
        scheduler = await start_scheduler()
        
        logger.info("✅ Scheduler service started successfully")
        logger.info(f"📊 Monitoring {len(scheduler.jobs_registry)} scheduled jobs")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("📝 Received shutdown signal")
        
    except Exception as e:
        logger.error(f"❌ Scheduler service failed: {e}")
        sys.exit(1)
    finally:
        await stop_scheduler()


if __name__ == "__main__":
    asyncio.run(main())


This module implements periodic scheduling for:
- Data collection and processing (daily/hourly)
- Insight generation (every 6 hours)
- System maintenance tasks
- Workflow monitoring and health checks

Phase 4 - Orchestration implementation.
"""
import asyncio
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from loguru import logger

from config import settings
from workflow.reddit_workflow import (
    reddit_workflow,
    run_data_collection_and_analysis,
    get_system_status
)


@dataclass
class ScheduledJob:
    """Data structure for scheduled jobs."""
    id: str
    name: str
    function: str
    trigger_type: str  # 'cron' or 'interval'
    trigger_config: Dict[str, Any]
    enabled: bool
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    success_count: int = 0
    error_count: int = 0
    last_result: Optional[Dict[str, Any]] = None


@dataclass
class SchedulerStatus:
    """Data structure for scheduler status."""
    running: bool
    jobs: List[ScheduledJob]
    total_jobs: int
    active_jobs: int
    last_health_check: str
    uptime_seconds: float
    system_status: Dict[str, Any]


class WorkflowScheduler:
    """Scheduler for Reddit Knowledge Base workflows."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.scheduler = AsyncIOScheduler(timezone='UTC')
        self.start_time = datetime.now(timezone.utc)
        self.jobs_registry: Dict[str, ScheduledJob] = {}
        self.status_file = Path("./data/scheduler_status.json")
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        self.scheduler.add_listener(self._job_missed, EVENT_JOB_MISSED)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down scheduler...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize the scheduler and register jobs."""
        logger.info("🔧 Initializing Workflow Scheduler...")
        
        try:
            # Initialize workflow agents
            await reddit_workflow.initialize_all_agents()
            
            # Register scheduled jobs
            await self._register_jobs()
            
            # Load previous status if available
            await self._load_status()
            
            logger.info("✅ Workflow Scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize scheduler: {e}")
            raise
    
    async def _register_jobs(self):
        """Register all scheduled jobs."""
        logger.info("📋 Registering scheduled jobs...")
        
        # Job 1: Daily data collection and analysis
        await self.add_job(
            job_id="daily_data_collection",
            name="Daily Data Collection & Analysis",
            func=self._run_daily_collection,
            trigger_type="cron",
            trigger_config={"hour": 2, "minute": 0},  # 2:00 AM UTC daily
            enabled=True
        )
        
        # Job 2: Hourly data collection (if configured)
        if settings.collection_interval_hours <= 1:
            await self.add_job(
                job_id="hourly_data_collection",
                name="Hourly Data Collection",
                func=self._run_hourly_collection,
                trigger_type="interval",
                trigger_config={"hours": 1},
                enabled=True
            )
        
        # Job 3: Insight generation every 6 hours
        await self.add_job(
            job_id="insight_generation",
            name="Insight Generation",
            func=self._run_insight_generation,
            trigger_type="interval",
            trigger_config={"hours": settings.insight_generation_interval_hours},
            enabled=True
        )
        
        # Job 4: System health check every 15 minutes
        await self.add_job(
            job_id="health_check",
            name="System Health Check",
            func=self._run_health_check,
            trigger_type="interval",
            trigger_config={"minutes": 15},
            enabled=True
        )
        
        # Job 5: Status persistence every 5 minutes
        await self.add_job(
            job_id="status_persistence",
            name="Status Persistence",
            func=self._persist_status,
            trigger_type="interval",
            trigger_config={"minutes": 5},
            enabled=True
        )
        
        # Job 6: Weekly cleanup (optional)
        await self.add_job(
            job_id="weekly_cleanup",
            name="Weekly System Cleanup",
            func=self._run_weekly_cleanup,
            trigger_type="cron",
            trigger_config={"day_of_week": 0, "hour": 3, "minute": 0},  # Sunday 3:00 AM
            enabled=False  # Disabled by default
        )
        
        logger.info(f"✅ Registered {len(self.jobs_registry)} scheduled jobs")
    
    async def add_job(
        self,
        job_id: str,
        name: str,
        func: Callable,
        trigger_type: str,
        trigger_config: Dict[str, Any],
        enabled: bool = True
    ):
        """Add a scheduled job."""
        try:
            # Create job registry entry
            scheduled_job = ScheduledJob(
                id=job_id,
                name=name,
                function=func.__name__,
                trigger_type=trigger_type,
                trigger_config=trigger_config,
                enabled=enabled
            )
            
            self.jobs_registry[job_id] = scheduled_job
            
            if enabled:
                # Add to scheduler
                if trigger_type == "cron":
                    trigger = CronTrigger(**trigger_config, timezone='UTC')
                elif trigger_type == "interval":
                    trigger = IntervalTrigger(**trigger_config, timezone='UTC')
                else:
                    raise ValueError(f"Unknown trigger type: {trigger_type}")
                
                self.scheduler.add_job(
                    func=func,
                    trigger=trigger,
                    id=job_id,
                    name=name,
                    replace_existing=True
                )
                
                logger.info(f"✅ Added job: {name} ({job_id})")
            else:
                logger.info(f"📝 Registered job (disabled): {name} ({job_id})")
                
        except Exception as e:
            logger.error(f"❌ Failed to add job {job_id}: {e}")
            raise
    
    async def remove_job(self, job_id: str):
        """Remove a scheduled job."""
        try:
            if job_id in self.jobs_registry:
                del self.jobs_registry[job_id]
            
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
            
            logger.info(f"✅ Removed job: {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to remove job {job_id}: {e}")
    
    async def enable_job(self, job_id: str):
        """Enable a scheduled job."""
        if job_id not in self.jobs_registry:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs_registry[job_id]
        job.enabled = True
        
        # Add to scheduler if not already there
        if not self.scheduler.get_job(job_id):
            func = getattr(self, job.function)
            
            if job.trigger_type == "cron":
                trigger = CronTrigger(**job.trigger_config, timezone='UTC')
            else:
                trigger = IntervalTrigger(**job.trigger_config, timezone='UTC')
            
            self.scheduler.add_job(
                func=func,
                trigger=trigger,
                id=job_id,
                name=job.name,
                replace_existing=True
            )
        
        logger.info(f"✅ Enabled job: {job.name}")
    
    async def disable_job(self, job_id: str):
        """Disable a scheduled job."""
        if job_id not in self.jobs_registry:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs_registry[job_id]
        job.enabled = False
        
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
        
        logger.info(f"✅ Disabled job: {job.name}")
    
    def start(self):
        """Start the scheduler."""
        logger.info("🚀 Starting Workflow Scheduler...")
        self.scheduler.start()
        logger.info("✅ Scheduler started successfully")
    
    async def shutdown(self):
        """Shutdown the scheduler gracefully."""
        logger.info("🔄 Shutting down Workflow Scheduler...")
        
        try:
            # Persist final status
            await self._persist_status()
            
            # Shutdown scheduler
            self.scheduler.shutdown(wait=True)
            
            # Close workflow agents
            await reddit_workflow.close_all_agents()
            
            logger.info("✅ Scheduler shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during scheduler shutdown: {e}")
    
    async def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        system_status = await get_system_status()
        
        # Update job info from scheduler
        for job_id, job in self.jobs_registry.items():
            scheduler_job = self.scheduler.get_job(job_id)
            if scheduler_job:
                job.next_run = scheduler_job.next_run_time.isoformat() if scheduler_job.next_run_time else None
        
        return SchedulerStatus(
            running=self.scheduler.running,
            jobs=list(self.jobs_registry.values()),
            total_jobs=len(self.jobs_registry),
            active_jobs=len([j for j in self.jobs_registry.values() if j.enabled]),
            last_health_check=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime,
            system_status=system_status
        )
    
    # Job execution methods
    async def _run_daily_collection(self):
        """Run daily data collection and analysis."""
        logger.info("📊 Starting daily data collection and analysis...")
        
        try:
            result = await run_data_collection_and_analysis()
            
            if result["success"]:
                logger.info("✅ Daily data collection completed successfully")
                return {"success": True, "message": "Daily collection completed"}
            else:
                logger.error(f"❌ Daily data collection failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            logger.error(f"❌ Daily data collection error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_hourly_collection(self):
        """Run hourly data collection."""
        logger.info("⏱️ Starting hourly data collection...")
        
        try:
            # Run collection only (no full analysis)
            result = await reddit_workflow.run_batch_workflow()
            
            if result["success"]:
                logger.info("✅ Hourly data collection completed")
                return {"success": True, "message": "Hourly collection completed"}
            else:
                logger.error(f"❌ Hourly collection failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            logger.error(f"❌ Hourly collection error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_insight_generation(self):
        """Run insight generation."""
        logger.info("💡 Starting insight generation...")
        
        try:
            
            logger.info(f"✅ Insight generation completed: {insights.id}")
            return {
                "success": True,
                "message": f"Generated insights: {insights.id}",
                "insights_id": insights.id,
                "clusters": len(insights.clusters),
                "documents": insights.total_documents
            }
            
        except Exception as e:
            logger.error(f"❌ Insight generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_health_check(self):
        """Run system health check."""
        try:
            status = await get_system_status()
            
            # Check critical components
            issues = []
            
            if not status.get("agents", {}).get("collector", {}).get("status") == "ready":
                issues.append("Collector agent not ready")
            
            if not status.get("agents", {}).get("preprocessor", {}).get("vector_db_connected"):
                issues.append("Vector database not connected")
            
                issues.append("SQLite database not connected")
            
            if issues:
                logger.warning(f"⚠️ Health check issues: {', '.join(issues)}")
                return {"success": False, "issues": issues}
            else:
                return {"success": True, "message": "All systems healthy"}
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _persist_status(self):
        """Persist scheduler status to file."""
        try:
            status = await self.get_status()
            status_dict = asdict(status)
            
            with open(self.status_file, 'w') as f:
                json.dump(status_dict, f, indent=2, default=str)
            
            return {"success": True, "message": "Status persisted"}
            
        except Exception as e:
            logger.error(f"❌ Status persistence error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _run_weekly_cleanup(self):
        """Run weekly system cleanup."""
        logger.info("🧹 Starting weekly system cleanup...")
        
        try:
            cleanup_results = []
            
            # Clean old log files (older than 30 days)
            log_dir = Path("./logs")
            if log_dir.exists():
                cutoff_date = datetime.now() - timedelta(days=30)
                for log_file in log_dir.glob("*.log.*"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        cleanup_results.append(f"Removed old log: {log_file.name}")
            
            # Clean old status files
            status_files = Path("./data").glob("scheduler_status_*.json")
            for status_file in status_files:
                if status_file.stat().st_mtime < (datetime.now() - timedelta(days=7)).timestamp():
                    status_file.unlink()
                    cleanup_results.append(f"Removed old status: {status_file.name}")
            
            logger.info(f"✅ Weekly cleanup completed: {len(cleanup_results)} items cleaned")
            return {"success": True, "cleaned_items": cleanup_results}
            
        except Exception as e:
            logger.error(f"❌ Weekly cleanup error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _load_status(self):
        """Load previous scheduler status."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
                
                # Restore job statistics
                for job_data in status_data.get("jobs", []):
                    job_id = job_data["id"]
                    if job_id in self.jobs_registry:
                        job = self.jobs_registry[job_id]
                        job.success_count = job_data.get("success_count", 0)
                        job.error_count = job_data.get("error_count", 0)
                        job.last_run = job_data.get("last_run")
                        job.last_result = job_data.get("last_result")
                
                logger.info("✅ Previous scheduler status loaded")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load previous status: {e}")
    
    # Event handlers
    def _job_executed(self, event):
        """Handle job execution events."""
        job_id = event.job_id
        
        if job_id in self.jobs_registry:
            job = self.jobs_registry[job_id]
            job.success_count += 1
            job.last_run = datetime.now(timezone.utc).isoformat()
            job.last_result = {"success": True, "executed_at": job.last_run}
            
            logger.info(f"✅ Job executed successfully: {job.name}")
    
    def _job_error(self, event):
        """Handle job error events."""
        job_id = event.job_id
        exception = event.exception
        
        if job_id in self.jobs_registry:
            job = self.jobs_registry[job_id]
            job.error_count += 1
            job.last_run = datetime.now(timezone.utc).isoformat()
            job.last_result = {
                "success": False,
                "error": str(exception),
                "executed_at": job.last_run
            }
            
            logger.error(f"❌ Job failed: {job.name} - {exception}")
    
    def _job_missed(self, event):
        """Handle missed job events."""
        job_id = event.job_id
        
        if job_id in self.jobs_registry:
            job = self.jobs_registry[job_id]
            logger.warning(f"⚠️ Job missed: {job.name}")


# Global scheduler instance
workflow_scheduler = WorkflowScheduler()


# Convenience functions
async def start_scheduler():
    """Start the workflow scheduler."""
    await workflow_scheduler.initialize()
    workflow_scheduler.start()
    return workflow_scheduler


async def stop_scheduler():
    """Stop the workflow scheduler."""
    await workflow_scheduler.shutdown()


async def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status."""
    status = await workflow_scheduler.get_status()
    return asdict(status)


# Main function for running as standalone service
async def main():
    """Main function for running scheduler as standalone service."""
    logger.info("🚀 Starting Reddit Knowledge Base Scheduler Service")
    
    try:
        # Start scheduler
        scheduler = await start_scheduler()
        
        logger.info("✅ Scheduler service started successfully")
        logger.info(f"📊 Monitoring {len(scheduler.jobs_registry)} scheduled jobs")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("📝 Received shutdown signal")
        
    except Exception as e:
        logger.error(f"❌ Scheduler service failed: {e}")
        sys.exit(1)
    finally:
        await stop_scheduler()


if __name__ == "__main__":
    asyncio.run(main())

