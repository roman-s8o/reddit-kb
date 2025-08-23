"""
Workflow State Persistence and Recovery System.

This module provides state persistence and recovery capabilities for the
Reddit Knowledge Base workflow orchestration system.

Phase 4 - Orchestration implementation.
"""
import asyncio
import json
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3

from loguru import logger

from config import settings


@dataclass
class WorkflowCheckpoint:
    """Data structure for workflow checkpoints."""
    checkpoint_id: str
    workflow_type: str
    workflow_state: Dict[str, Any]
    timestamp: str
    step_completed: str
    next_step: Optional[str]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Data structure for workflow execution tracking."""
    execution_id: str
    workflow_type: str
    started_at: str
    completed_at: Optional[str]
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    checkpoints: List[WorkflowCheckpoint]
    final_result: Optional[Dict[str, Any]]
    parameters: Dict[str, Any]


class WorkflowPersistence:
    """Persistence manager for workflow states and executions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the persistence manager."""
        self.db_path = Path(db_path or "./data/workflow_persistence.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # State storage directories
        self.state_dir = Path("./data/workflow_states")
        self.checkpoint_dir = Path("./data/checkpoints")
        self.recovery_dir = Path("./data/recovery")
        
        # Create directories
        for dir_path in [self.state_dir, self.checkpoint_dir, self.recovery_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for workflow tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create workflow executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    final_result TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    step_completed TEXT NOT NULL,
                    next_step TEXT,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error TEXT,
                    metadata TEXT,
                    state_file TEXT,
                    FOREIGN KEY (execution_id) REFERENCES workflow_executions (execution_id)
                )
            """)
            
            # Create recovery log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    recovery_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    details TEXT,
                    FOREIGN KEY (execution_id) REFERENCES workflow_executions (execution_id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Workflow persistence database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize persistence database: {e}")
            raise
    
    async def start_workflow_execution(
        self,
        execution_id: str,
        workflow_type: str,
        parameters: Dict[str, Any]
    ) -> WorkflowExecution:
        """Start tracking a new workflow execution."""
        try:
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_type=workflow_type,
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=None,
                status="running",
                checkpoints=[],
                final_result=None,
                parameters=parameters
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workflow_executions 
                (execution_id, workflow_type, started_at, status, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                workflow_type,
                execution.started_at,
                "running",
                json.dumps(parameters),
                datetime.now(timezone.utc).isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Started workflow execution: {execution_id}")
            return execution
            
        except Exception as e:
            logger.error(f"❌ Failed to start workflow execution: {e}")
            raise
    
    async def create_checkpoint(
        self,
        execution_id: str,
        workflow_type: str,
        workflow_state: Dict[str, Any],
        step_completed: str,
        next_step: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowCheckpoint:
        """Create a workflow checkpoint."""
        try:
            checkpoint_id = f"{execution_id}_{step_completed}_{int(datetime.now().timestamp())}"
            timestamp = datetime.now(timezone.utc).isoformat()
            
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=checkpoint_id,
                workflow_type=workflow_type,
                workflow_state=workflow_state,
                timestamp=timestamp,
                step_completed=step_completed,
                next_step=next_step,
                metadata=metadata or {},
                success=success,
                error=error
            )
            
            # Save state to file
            state_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(state_file, 'w') as f:
                json.dump({
                    "checkpoint": asdict(checkpoint),
                    "workflow_state": workflow_state
                }, f, indent=2, default=str)
            
            # Store checkpoint in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workflow_checkpoints 
                (checkpoint_id, execution_id, workflow_type, step_completed, 
                 next_step, timestamp, success, error, metadata, state_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                execution_id,
                workflow_type,
                step_completed,
                next_step,
                timestamp,
                success,
                error,
                json.dumps(metadata or {}),
                str(state_file)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Created checkpoint: {checkpoint_id} at step {step_completed}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ Failed to create checkpoint: {e}")
            raise
    
    async def complete_workflow_execution(
        self,
        execution_id: str,
        status: str,
        final_result: Optional[Dict[str, Any]] = None
    ):
        """Mark workflow execution as completed."""
        try:
            completed_at = datetime.now(timezone.utc).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE workflow_executions 
                SET completed_at = ?, status = ?, final_result = ?
                WHERE execution_id = ?
            """, (
                completed_at,
                status,
                json.dumps(final_result) if final_result else None,
                execution_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Completed workflow execution: {execution_id} with status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to complete workflow execution: {e}")
            raise
    
    async def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get execution record
            cursor.execute("""
                SELECT execution_id, workflow_type, started_at, completed_at, 
                       status, parameters, final_result
                FROM workflow_executions 
                WHERE execution_id = ?
            """, (execution_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            # Get checkpoints
            cursor.execute("""
                SELECT checkpoint_id, step_completed, next_step, timestamp, 
                       success, error, metadata, state_file
                FROM workflow_checkpoints 
                WHERE execution_id = ?
                ORDER BY timestamp
            """, (execution_id,))
            
            checkpoint_rows = cursor.fetchall()
            conn.close()
            
            # Build checkpoints list
            checkpoints = []
            for cp_row in checkpoint_rows:
                checkpoint = WorkflowCheckpoint(
                    checkpoint_id=cp_row[0],
                    workflow_type=row[1],
                    workflow_state={},  # Will be loaded from file if needed
                    timestamp=cp_row[3],
                    step_completed=cp_row[1],
                    next_step=cp_row[2],
                    metadata=json.loads(cp_row[6]) if cp_row[6] else {},
                    success=bool(cp_row[4]),
                    error=cp_row[5]
                )
                checkpoints.append(checkpoint)
            
            # Build execution object
            execution = WorkflowExecution(
                execution_id=row[0],
                workflow_type=row[1],
                started_at=row[2],
                completed_at=row[3],
                status=row[4],
                checkpoints=checkpoints,
                final_result=json.loads(row[6]) if row[6] else None,
                parameters=json.loads(row[5])
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"❌ Failed to get workflow execution: {e}")
            return None
    
    async def get_latest_checkpoint(
        self,
        execution_id: str
    ) -> Optional[Tuple[WorkflowCheckpoint, Dict[str, Any]]]:
        """Get the latest checkpoint for a workflow execution."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT checkpoint_id, workflow_type, step_completed, next_step, 
                       timestamp, success, error, metadata, state_file
                FROM workflow_checkpoints 
                WHERE execution_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (execution_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Load state from file
            state_file = Path(row[8])
            workflow_state = {}
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    workflow_state = data.get("workflow_state", {})
            
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=row[0],
                workflow_type=row[1],
                workflow_state=workflow_state,
                timestamp=row[4],
                step_completed=row[2],
                next_step=row[3],
                metadata=json.loads(row[7]) if row[7] else {},
                success=bool(row[5]),
                error=row[6]
            )
            
            return checkpoint, workflow_state
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest checkpoint: {e}")
            return None
    
    async def get_failed_executions(
        self,
        limit: int = 10
    ) -> List[WorkflowExecution]:
        """Get failed workflow executions for recovery."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT execution_id, workflow_type, started_at, completed_at, 
                       status, parameters, final_result
                FROM workflow_executions 
                WHERE status = 'failed'
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            executions = []
            for row in rows:
                execution = WorkflowExecution(
                    execution_id=row[0],
                    workflow_type=row[1],
                    started_at=row[2],
                    completed_at=row[3],
                    status=row[4],
                    checkpoints=[],  # Not loading checkpoints for list view
                    final_result=json.loads(row[6]) if row[6] else None,
                    parameters=json.loads(row[5])
                )
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"❌ Failed to get failed executions: {e}")
            return []
    
    async def recover_workflow_execution(
        self,
        execution_id: str,
        recovery_strategy: str = "from_last_checkpoint"
    ) -> bool:
        """Recover a failed workflow execution."""
        try:
            logger.info(f"🔄 Starting workflow recovery: {execution_id}")
            
            # Get execution details
            execution = await self.get_workflow_execution(execution_id)
            if not execution:
                logger.error(f"❌ Execution not found: {execution_id}")
                return False
            
            # Get latest checkpoint
            checkpoint_data = await self.get_latest_checkpoint(execution_id)
            if not checkpoint_data:
                logger.error(f"❌ No checkpoints found for execution: {execution_id}")
                return False
            
            checkpoint, workflow_state = checkpoint_data
            
            if recovery_strategy == "from_last_checkpoint":
                # Log recovery attempt
                await self._log_recovery_attempt(
                    execution_id,
                    "from_last_checkpoint",
                    f"Recovering from step: {checkpoint.step_completed}"
                )
                
                # Update execution status
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE workflow_executions 
                    SET status = 'running', completed_at = NULL
                    WHERE execution_id = ?
                """, (execution_id,))
                
                conn.commit()
                conn.close()
                
                logger.info(f"✅ Workflow recovery prepared: {execution_id}")
                logger.info(f"   Recovery point: {checkpoint.step_completed}")
                logger.info(f"   Next step: {checkpoint.next_step}")
                
                return True
            
            else:
                logger.error(f"❌ Unknown recovery strategy: {recovery_strategy}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to recover workflow execution: {e}")
            await self._log_recovery_attempt(
                execution_id,
                recovery_strategy,
                f"Recovery failed: {str(e)}",
                success=False
            )
            return False
    
    async def _log_recovery_attempt(
        self,
        execution_id: str,
        recovery_type: str,
        details: str,
        success: bool = True
    ):
        """Log recovery attempt."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO recovery_log 
                (execution_id, recovery_type, timestamp, success, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                execution_id,
                recovery_type,
                datetime.now(timezone.utc).isoformat(),
                success,
                details
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to log recovery attempt: {e}")
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old workflow data."""
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old executions to clean up files
            cursor.execute("""
                SELECT execution_id FROM workflow_executions 
                WHERE started_at < ? AND status IN ('completed', 'failed')
            """, (cutoff_str,))
            
            old_executions = [row[0] for row in cursor.fetchall()]
            
            # Delete old checkpoints and files
            cursor.execute("""
                SELECT state_file FROM workflow_checkpoints 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            state_files = [row[0] for row in cursor.fetchall()]
            
            # Delete database records
            cursor.execute("""
                DELETE FROM workflow_checkpoints 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            cursor.execute("""
                DELETE FROM recovery_log 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            cursor.execute("""
                DELETE FROM workflow_executions 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            conn.commit()
            conn.close()
            
            # Delete state files
            files_deleted = 0
            for state_file_path in state_files:
                try:
                    state_file = Path(state_file_path)
                    if state_file.exists():
                        state_file.unlink()
                        files_deleted += 1
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete state file {state_file_path}: {e}")
            
            logger.info(f"🧹 Cleaned up {len(old_executions)} old executions and {files_deleted} files")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
    
    async def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get execution stats
            cursor.execute("""
                SELECT status, COUNT(*) FROM workflow_executions 
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Get total checkpoints
            cursor.execute("SELECT COUNT(*) FROM workflow_checkpoints")
            total_checkpoints = cursor.fetchone()[0]
            
            # Get recovery attempts
            cursor.execute("SELECT COUNT(*) FROM recovery_log")
            total_recoveries = cursor.fetchone()[0]
            
            # Get recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM workflow_executions 
                WHERE started_at > datetime('now', '-24 hours')
            """)
            recent_executions = cursor.fetchone()[0]
            
            conn.close()
            
            # Get file system stats
            checkpoint_files = len(list(self.checkpoint_dir.glob("*.json")))
            
            return {
                "database_path": str(self.db_path),
                "execution_counts": status_counts,
                "total_checkpoints": total_checkpoints,
                "total_recoveries": total_recoveries,
                "recent_executions_24h": recent_executions,
                "checkpoint_files": checkpoint_files,
                "storage_directories": {
                    "states": str(self.state_dir),
                    "checkpoints": str(self.checkpoint_dir),
                    "recovery": str(self.recovery_dir)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get persistence stats: {e}")
            return {}


# Global persistence manager
workflow_persistence = WorkflowPersistence()


# Convenience functions
async def start_workflow_tracking(execution_id: str, workflow_type: str, parameters: Dict[str, Any]) -> WorkflowExecution:
    """Start tracking a workflow execution."""
    return await workflow_persistence.start_workflow_execution(execution_id, workflow_type, parameters)


async def create_workflow_checkpoint(
    execution_id: str,
    workflow_type: str,
    workflow_state: Dict[str, Any],
    step_completed: str,
    next_step: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WorkflowCheckpoint:
    """Create a workflow checkpoint."""
    return await workflow_persistence.create_checkpoint(
        execution_id, workflow_type, workflow_state, step_completed,
        next_step, success, error, metadata
    )


async def complete_workflow_tracking(execution_id: str, status: str, final_result: Optional[Dict[str, Any]] = None):
    """Complete workflow tracking."""
    await workflow_persistence.complete_workflow_execution(execution_id, status, final_result)


async def recover_failed_workflow(execution_id: str) -> bool:
    """Recover a failed workflow."""
    return await workflow_persistence.recover_workflow_execution(execution_id)


async def get_persistence_statistics() -> Dict[str, Any]:
    """Get persistence system statistics."""
    return await workflow_persistence.get_persistence_stats()


if __name__ == "__main__":
    # Example usage
    async def demo():
        logger.info("🔄 Workflow Persistence Demo")
        
        # Start workflow
        execution = await start_workflow_tracking(
            "demo_workflow_001",
            "batch",
            {"subreddits": ["Python"]}
        )
        
        # Create checkpoints
        await create_workflow_checkpoint(
            execution.execution_id,
            "batch",
            {"step": "collection", "progress": 0.3},
            "collection_completed",
            "preprocessing_start"
        )
        
        await create_workflow_checkpoint(
            execution.execution_id,
            "batch",
            {"step": "preprocessing", "progress": 0.7},
            "preprocessing_completed",
            "insight_generation_start"
        )
        
        # Complete workflow
        await complete_workflow_tracking(
            execution.execution_id,
            "completed",
            {"total_documents": 150, "insights_generated": 5}
        )
        
        # Get stats
        stats = await get_persistence_statistics()
        logger.info(f"📊 Persistence stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(demo())


This module provides state persistence and recovery capabilities for the
Reddit Knowledge Base workflow orchestration system.

Phase 4 - Orchestration implementation.
"""
import asyncio
import json
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3

from loguru import logger

from config import settings


@dataclass
class WorkflowCheckpoint:
    """Data structure for workflow checkpoints."""
    checkpoint_id: str
    workflow_type: str
    workflow_state: Dict[str, Any]
    timestamp: str
    step_completed: str
    next_step: Optional[str]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Data structure for workflow execution tracking."""
    execution_id: str
    workflow_type: str
    started_at: str
    completed_at: Optional[str]
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    checkpoints: List[WorkflowCheckpoint]
    final_result: Optional[Dict[str, Any]]
    parameters: Dict[str, Any]


class WorkflowPersistence:
    """Persistence manager for workflow states and executions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the persistence manager."""
        self.db_path = Path(db_path or "./data/workflow_persistence.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # State storage directories
        self.state_dir = Path("./data/workflow_states")
        self.checkpoint_dir = Path("./data/checkpoints")
        self.recovery_dir = Path("./data/recovery")
        
        # Create directories
        for dir_path in [self.state_dir, self.checkpoint_dir, self.recovery_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for workflow tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create workflow executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    final_result TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    step_completed TEXT NOT NULL,
                    next_step TEXT,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error TEXT,
                    metadata TEXT,
                    state_file TEXT,
                    FOREIGN KEY (execution_id) REFERENCES workflow_executions (execution_id)
                )
            """)
            
            # Create recovery log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    recovery_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    details TEXT,
                    FOREIGN KEY (execution_id) REFERENCES workflow_executions (execution_id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Workflow persistence database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize persistence database: {e}")
            raise
    
    async def start_workflow_execution(
        self,
        execution_id: str,
        workflow_type: str,
        parameters: Dict[str, Any]
    ) -> WorkflowExecution:
        """Start tracking a new workflow execution."""
        try:
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_type=workflow_type,
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=None,
                status="running",
                checkpoints=[],
                final_result=None,
                parameters=parameters
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workflow_executions 
                (execution_id, workflow_type, started_at, status, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                workflow_type,
                execution.started_at,
                "running",
                json.dumps(parameters),
                datetime.now(timezone.utc).isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Started workflow execution: {execution_id}")
            return execution
            
        except Exception as e:
            logger.error(f"❌ Failed to start workflow execution: {e}")
            raise
    
    async def create_checkpoint(
        self,
        execution_id: str,
        workflow_type: str,
        workflow_state: Dict[str, Any],
        step_completed: str,
        next_step: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowCheckpoint:
        """Create a workflow checkpoint."""
        try:
            checkpoint_id = f"{execution_id}_{step_completed}_{int(datetime.now().timestamp())}"
            timestamp = datetime.now(timezone.utc).isoformat()
            
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=checkpoint_id,
                workflow_type=workflow_type,
                workflow_state=workflow_state,
                timestamp=timestamp,
                step_completed=step_completed,
                next_step=next_step,
                metadata=metadata or {},
                success=success,
                error=error
            )
            
            # Save state to file
            state_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(state_file, 'w') as f:
                json.dump({
                    "checkpoint": asdict(checkpoint),
                    "workflow_state": workflow_state
                }, f, indent=2, default=str)
            
            # Store checkpoint in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workflow_checkpoints 
                (checkpoint_id, execution_id, workflow_type, step_completed, 
                 next_step, timestamp, success, error, metadata, state_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                execution_id,
                workflow_type,
                step_completed,
                next_step,
                timestamp,
                success,
                error,
                json.dumps(metadata or {}),
                str(state_file)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Created checkpoint: {checkpoint_id} at step {step_completed}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ Failed to create checkpoint: {e}")
            raise
    
    async def complete_workflow_execution(
        self,
        execution_id: str,
        status: str,
        final_result: Optional[Dict[str, Any]] = None
    ):
        """Mark workflow execution as completed."""
        try:
            completed_at = datetime.now(timezone.utc).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE workflow_executions 
                SET completed_at = ?, status = ?, final_result = ?
                WHERE execution_id = ?
            """, (
                completed_at,
                status,
                json.dumps(final_result) if final_result else None,
                execution_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Completed workflow execution: {execution_id} with status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to complete workflow execution: {e}")
            raise
    
    async def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get execution record
            cursor.execute("""
                SELECT execution_id, workflow_type, started_at, completed_at, 
                       status, parameters, final_result
                FROM workflow_executions 
                WHERE execution_id = ?
            """, (execution_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            # Get checkpoints
            cursor.execute("""
                SELECT checkpoint_id, step_completed, next_step, timestamp, 
                       success, error, metadata, state_file
                FROM workflow_checkpoints 
                WHERE execution_id = ?
                ORDER BY timestamp
            """, (execution_id,))
            
            checkpoint_rows = cursor.fetchall()
            conn.close()
            
            # Build checkpoints list
            checkpoints = []
            for cp_row in checkpoint_rows:
                checkpoint = WorkflowCheckpoint(
                    checkpoint_id=cp_row[0],
                    workflow_type=row[1],
                    workflow_state={},  # Will be loaded from file if needed
                    timestamp=cp_row[3],
                    step_completed=cp_row[1],
                    next_step=cp_row[2],
                    metadata=json.loads(cp_row[6]) if cp_row[6] else {},
                    success=bool(cp_row[4]),
                    error=cp_row[5]
                )
                checkpoints.append(checkpoint)
            
            # Build execution object
            execution = WorkflowExecution(
                execution_id=row[0],
                workflow_type=row[1],
                started_at=row[2],
                completed_at=row[3],
                status=row[4],
                checkpoints=checkpoints,
                final_result=json.loads(row[6]) if row[6] else None,
                parameters=json.loads(row[5])
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"❌ Failed to get workflow execution: {e}")
            return None
    
    async def get_latest_checkpoint(
        self,
        execution_id: str
    ) -> Optional[Tuple[WorkflowCheckpoint, Dict[str, Any]]]:
        """Get the latest checkpoint for a workflow execution."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT checkpoint_id, workflow_type, step_completed, next_step, 
                       timestamp, success, error, metadata, state_file
                FROM workflow_checkpoints 
                WHERE execution_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (execution_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Load state from file
            state_file = Path(row[8])
            workflow_state = {}
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    workflow_state = data.get("workflow_state", {})
            
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=row[0],
                workflow_type=row[1],
                workflow_state=workflow_state,
                timestamp=row[4],
                step_completed=row[2],
                next_step=row[3],
                metadata=json.loads(row[7]) if row[7] else {},
                success=bool(row[5]),
                error=row[6]
            )
            
            return checkpoint, workflow_state
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest checkpoint: {e}")
            return None
    
    async def get_failed_executions(
        self,
        limit: int = 10
    ) -> List[WorkflowExecution]:
        """Get failed workflow executions for recovery."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT execution_id, workflow_type, started_at, completed_at, 
                       status, parameters, final_result
                FROM workflow_executions 
                WHERE status = 'failed'
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            executions = []
            for row in rows:
                execution = WorkflowExecution(
                    execution_id=row[0],
                    workflow_type=row[1],
                    started_at=row[2],
                    completed_at=row[3],
                    status=row[4],
                    checkpoints=[],  # Not loading checkpoints for list view
                    final_result=json.loads(row[6]) if row[6] else None,
                    parameters=json.loads(row[5])
                )
                executions.append(execution)
            
            return executions
            
        except Exception as e:
            logger.error(f"❌ Failed to get failed executions: {e}")
            return []
    
    async def recover_workflow_execution(
        self,
        execution_id: str,
        recovery_strategy: str = "from_last_checkpoint"
    ) -> bool:
        """Recover a failed workflow execution."""
        try:
            logger.info(f"🔄 Starting workflow recovery: {execution_id}")
            
            # Get execution details
            execution = await self.get_workflow_execution(execution_id)
            if not execution:
                logger.error(f"❌ Execution not found: {execution_id}")
                return False
            
            # Get latest checkpoint
            checkpoint_data = await self.get_latest_checkpoint(execution_id)
            if not checkpoint_data:
                logger.error(f"❌ No checkpoints found for execution: {execution_id}")
                return False
            
            checkpoint, workflow_state = checkpoint_data
            
            if recovery_strategy == "from_last_checkpoint":
                # Log recovery attempt
                await self._log_recovery_attempt(
                    execution_id,
                    "from_last_checkpoint",
                    f"Recovering from step: {checkpoint.step_completed}"
                )
                
                # Update execution status
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE workflow_executions 
                    SET status = 'running', completed_at = NULL
                    WHERE execution_id = ?
                """, (execution_id,))
                
                conn.commit()
                conn.close()
                
                logger.info(f"✅ Workflow recovery prepared: {execution_id}")
                logger.info(f"   Recovery point: {checkpoint.step_completed}")
                logger.info(f"   Next step: {checkpoint.next_step}")
                
                return True
            
            else:
                logger.error(f"❌ Unknown recovery strategy: {recovery_strategy}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to recover workflow execution: {e}")
            await self._log_recovery_attempt(
                execution_id,
                recovery_strategy,
                f"Recovery failed: {str(e)}",
                success=False
            )
            return False
    
    async def _log_recovery_attempt(
        self,
        execution_id: str,
        recovery_type: str,
        details: str,
        success: bool = True
    ):
        """Log recovery attempt."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO recovery_log 
                (execution_id, recovery_type, timestamp, success, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                execution_id,
                recovery_type,
                datetime.now(timezone.utc).isoformat(),
                success,
                details
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to log recovery attempt: {e}")
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old workflow data."""
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get old executions to clean up files
            cursor.execute("""
                SELECT execution_id FROM workflow_executions 
                WHERE started_at < ? AND status IN ('completed', 'failed')
            """, (cutoff_str,))
            
            old_executions = [row[0] for row in cursor.fetchall()]
            
            # Delete old checkpoints and files
            cursor.execute("""
                SELECT state_file FROM workflow_checkpoints 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            state_files = [row[0] for row in cursor.fetchall()]
            
            # Delete database records
            cursor.execute("""
                DELETE FROM workflow_checkpoints 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            cursor.execute("""
                DELETE FROM recovery_log 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            cursor.execute("""
                DELETE FROM workflow_executions 
                WHERE execution_id IN ({})
            """.format(','.join('?' * len(old_executions))), old_executions)
            
            conn.commit()
            conn.close()
            
            # Delete state files
            files_deleted = 0
            for state_file_path in state_files:
                try:
                    state_file = Path(state_file_path)
                    if state_file.exists():
                        state_file.unlink()
                        files_deleted += 1
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete state file {state_file_path}: {e}")
            
            logger.info(f"🧹 Cleaned up {len(old_executions)} old executions and {files_deleted} files")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
    
    async def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get execution stats
            cursor.execute("""
                SELECT status, COUNT(*) FROM workflow_executions 
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Get total checkpoints
            cursor.execute("SELECT COUNT(*) FROM workflow_checkpoints")
            total_checkpoints = cursor.fetchone()[0]
            
            # Get recovery attempts
            cursor.execute("SELECT COUNT(*) FROM recovery_log")
            total_recoveries = cursor.fetchone()[0]
            
            # Get recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM workflow_executions 
                WHERE started_at > datetime('now', '-24 hours')
            """)
            recent_executions = cursor.fetchone()[0]
            
            conn.close()
            
            # Get file system stats
            checkpoint_files = len(list(self.checkpoint_dir.glob("*.json")))
            
            return {
                "database_path": str(self.db_path),
                "execution_counts": status_counts,
                "total_checkpoints": total_checkpoints,
                "total_recoveries": total_recoveries,
                "recent_executions_24h": recent_executions,
                "checkpoint_files": checkpoint_files,
                "storage_directories": {
                    "states": str(self.state_dir),
                    "checkpoints": str(self.checkpoint_dir),
                    "recovery": str(self.recovery_dir)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get persistence stats: {e}")
            return {}


# Global persistence manager
workflow_persistence = WorkflowPersistence()


# Convenience functions
async def start_workflow_tracking(execution_id: str, workflow_type: str, parameters: Dict[str, Any]) -> WorkflowExecution:
    """Start tracking a workflow execution."""
    return await workflow_persistence.start_workflow_execution(execution_id, workflow_type, parameters)


async def create_workflow_checkpoint(
    execution_id: str,
    workflow_type: str,
    workflow_state: Dict[str, Any],
    step_completed: str,
    next_step: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> WorkflowCheckpoint:
    """Create a workflow checkpoint."""
    return await workflow_persistence.create_checkpoint(
        execution_id, workflow_type, workflow_state, step_completed,
        next_step, success, error, metadata
    )


async def complete_workflow_tracking(execution_id: str, status: str, final_result: Optional[Dict[str, Any]] = None):
    """Complete workflow tracking."""
    await workflow_persistence.complete_workflow_execution(execution_id, status, final_result)


async def recover_failed_workflow(execution_id: str) -> bool:
    """Recover a failed workflow."""
    return await workflow_persistence.recover_workflow_execution(execution_id)


async def get_persistence_statistics() -> Dict[str, Any]:
    """Get persistence system statistics."""
    return await workflow_persistence.get_persistence_stats()


if __name__ == "__main__":
    # Example usage
    async def demo():
        logger.info("🔄 Workflow Persistence Demo")
        
        # Start workflow
        execution = await start_workflow_tracking(
            "demo_workflow_001",
            "batch",
            {"subreddits": ["Python"]}
        )
        
        # Create checkpoints
        await create_workflow_checkpoint(
            execution.execution_id,
            "batch",
            {"step": "collection", "progress": 0.3},
            "collection_completed",
            "preprocessing_start"
        )
        
        await create_workflow_checkpoint(
            execution.execution_id,
            "batch",
            {"step": "preprocessing", "progress": 0.7},
            "preprocessing_completed",
            "insight_generation_start"
        )
        
        # Complete workflow
        await complete_workflow_tracking(
            execution.execution_id,
            "completed",
            {"total_documents": 150, "insights_generated": 5}
        )
        
        # Get stats
        stats = await get_persistence_statistics()
        logger.info(f"📊 Persistence stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(demo())

