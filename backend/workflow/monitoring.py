"""
Workflow Monitoring and Metrics Collection.

This module provides comprehensive monitoring capabilities for the
Reddit Knowledge Base workflow orchestration system.

Phase 4 - Orchestration implementation.
"""
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque

from loguru import logger

from config import settings
from workflow.reddit_workflow import get_system_status


@dataclass
class SystemMetrics:
    """Data structure for system metrics."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    response_time: float


@dataclass
class WorkflowMetrics:
    """Data structure for workflow metrics."""
    workflow_type: str
    execution_count: int
    success_count: int
    error_count: int
    avg_duration: float
    last_execution: Optional[str]
    success_rate: float


@dataclass
class AgentMetrics:
    """Data structure for agent metrics."""
    agent_name: str
    status: str
    last_activity: str
    total_operations: int
    success_operations: int
    error_operations: int
    avg_response_time: float


class WorkflowMonitor:
    """Monitor for workflow orchestration system."""
    
    def __init__(self, metrics_history_size: int = 1000):
        """Initialize the workflow monitor."""
        self.metrics_history_size = metrics_history_size
        self.metrics_history: deque = deque(maxlen=metrics_history_size)
        self.workflow_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "execution_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_duration": 0.0,
            "last_execution": None
        })
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_operations": 0,
            "success_operations": 0,
            "error_operations": 0,
            "total_response_time": 0.0,
            "last_activity": None,
            "status": "unknown"
        })
        
        self.alerts: List[Dict[str, Any]] = []
        self.metrics_file = Path("./data/workflow_metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 10.0,  # seconds
            "response_time_critical": 30.0,
            "error_rate_warning": 0.1,  # 10%
            "error_rate_critical": 0.25,  # 25%
            "memory_usage_warning": 0.8,  # 80%
            "memory_usage_critical": 0.95,  # 95%
        }
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network connections (approximate)
            connections = len(psutil.net_connections())
            
            # Measure response time to system status
            start_time = time.time()
            await get_system_status()
            response_time = time.time() - start_time
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                active_connections=connections,
                response_time=response_time
            )
            
            # Store in history
            self.metrics_history.append(asdict(metrics))
            
            # Check for alerts
            await self._check_system_alerts(metrics)
            
            return metrics
            
        except ImportError:
            logger.warning("psutil not available, using mock metrics")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                active_connections=0,
                response_time=1.0
            )
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                active_connections=0,
                response_time=999.0
            )
    
    def record_workflow_execution(
        self,
        workflow_type: str,
        success: bool,
        duration: float,
        error: Optional[str] = None
    ):
        """Record workflow execution metrics."""
        stats = self.workflow_stats[workflow_type]
        
        stats["execution_count"] += 1
        stats["total_duration"] += duration
        stats["last_execution"] = datetime.now(timezone.utc).isoformat()
        
        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1
            logger.warning(f"Workflow {workflow_type} failed: {error}")
        
        # Check for workflow alerts
        self._check_workflow_alerts(workflow_type, stats)
    
    def record_agent_activity(
        self,
        agent_name: str,
        operation: str,
        success: bool,
        response_time: float,
        status: str = "active"
    ):
        """Record agent activity metrics."""
        stats = self.agent_stats[agent_name]
        
        stats["total_operations"] += 1
        stats["total_response_time"] += response_time
        stats["last_activity"] = datetime.now(timezone.utc).isoformat()
        stats["status"] = status
        
        if success:
            stats["success_operations"] += 1
        else:
            stats["error_operations"] += 1
            logger.warning(f"Agent {agent_name} operation {operation} failed")
        
        # Check for agent alerts
        self._check_agent_alerts(agent_name, stats)
    
    def get_workflow_metrics(self) -> List[WorkflowMetrics]:
        """Get workflow metrics summary."""
        metrics = []
        
        for workflow_type, stats in self.workflow_stats.items():
            total_executions = stats["execution_count"]
            success_rate = (stats["success_count"] / max(total_executions, 1)) * 100
            avg_duration = stats["total_duration"] / max(total_executions, 1)
            
            metrics.append(WorkflowMetrics(
                workflow_type=workflow_type,
                execution_count=total_executions,
                success_count=stats["success_count"],
                error_count=stats["error_count"],
                avg_duration=avg_duration,
                last_execution=stats["last_execution"],
                success_rate=success_rate
            ))
        
        return metrics
    
    def get_agent_metrics(self) -> List[AgentMetrics]:
        """Get agent metrics summary."""
        metrics = []
        
        for agent_name, stats in self.agent_stats.items():
            total_operations = stats["total_operations"]
            avg_response_time = stats["total_response_time"] / max(total_operations, 1)
            
            metrics.append(AgentMetrics(
                agent_name=agent_name,
                status=stats["status"],
                last_activity=stats["last_activity"] or "never",
                total_operations=total_operations,
                success_operations=stats["success_operations"],
                error_operations=stats["error_operations"],
                avg_response_time=avg_response_time
            ))
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        # Calculate overall metrics
        total_executions = sum(stats["execution_count"] for stats in self.workflow_stats.values())
        total_successes = sum(stats["success_count"] for stats in self.workflow_stats.values())
        total_errors = sum(stats["error_count"] for stats in self.workflow_stats.values())
        
        overall_success_rate = (total_successes / max(total_executions, 1)) * 100
        
        # Get recent system metrics
        recent_metrics = list(self.metrics_history)[-10:] if self.metrics_history else []
        
        avg_response_time = 0.0
        avg_cpu_usage = 0.0
        avg_memory_usage = 0.0
        
        if recent_metrics:
            avg_response_time = sum(m["response_time"] for m in recent_metrics) / len(recent_metrics)
            avg_cpu_usage = sum(m["cpu_usage"] for m in recent_metrics) / len(recent_metrics)
            avg_memory_usage = sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics)
        
        return {
            "overall_metrics": {
                "total_executions": total_executions,
                "success_rate": overall_success_rate,
                "avg_response_time": avg_response_time,
                "avg_cpu_usage": avg_cpu_usage,
                "avg_memory_usage": avg_memory_usage
            },
            "workflow_metrics": [asdict(m) for m in self.get_workflow_metrics()],
            "agent_metrics": [asdict(m) for m in self.get_agent_metrics()],
            "active_alerts": self.get_active_alerts(),
            "metrics_collected": len(self.metrics_history),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        # Filter alerts from last 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        active_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert["timestamp"].replace('Z', '+00:00'))
            if alert_time > cutoff_time:
                active_alerts.append(alert)
        
        return sorted(active_alerts, key=lambda x: x["timestamp"], reverse=True)
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-level alerts."""
        alerts = []
        
        # Check response time
        if metrics.response_time > self.thresholds["response_time_critical"]:
            alerts.append({
                "type": "system",
                "level": "critical",
                "message": f"System response time is {metrics.response_time:.2f}s (threshold: {self.thresholds['response_time_critical']}s)",
                "metric": "response_time",
                "value": metrics.response_time
            })
        elif metrics.response_time > self.thresholds["response_time_warning"]:
            alerts.append({
                "type": "system",
                "level": "warning",
                "message": f"System response time is {metrics.response_time:.2f}s (threshold: {self.thresholds['response_time_warning']}s)",
                "metric": "response_time",
                "value": metrics.response_time
            })
        
        # Check memory usage
        if metrics.memory_usage > self.thresholds["memory_usage_critical"] * 100:
            alerts.append({
                "type": "system",
                "level": "critical",
                "message": f"Memory usage is {metrics.memory_usage:.1f}% (threshold: {self.thresholds['memory_usage_critical'] * 100:.1f}%)",
                "metric": "memory_usage",
                "value": metrics.memory_usage
            })
        elif metrics.memory_usage > self.thresholds["memory_usage_warning"] * 100:
            alerts.append({
                "type": "system",
                "level": "warning",
                "message": f"Memory usage is {metrics.memory_usage:.1f}% (threshold: {self.thresholds['memory_usage_warning'] * 100:.1f}%)",
                "metric": "memory_usage",
                "value": metrics.memory_usage
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = datetime.now(timezone.utc).isoformat()
            self.alerts.append(alert)
            
            if alert["level"] == "critical":
                logger.error(f"🚨 CRITICAL ALERT: {alert['message']}")
            else:
                logger.warning(f"⚠️ WARNING: {alert['message']}")
    
    def _check_workflow_alerts(self, workflow_type: str, stats: Dict[str, Any]):
        """Check for workflow-level alerts."""
        total_executions = stats["execution_count"]
        error_rate = stats["error_count"] / max(total_executions, 1)
        
        alerts = []
        
        # Check error rate
        if error_rate > self.thresholds["error_rate_critical"]:
            alerts.append({
                "type": "workflow",
                "level": "critical",
                "message": f"Workflow {workflow_type} error rate is {error_rate:.1%} (threshold: {self.thresholds['error_rate_critical']:.1%})",
                "workflow_type": workflow_type,
                "metric": "error_rate",
                "value": error_rate
            })
        elif error_rate > self.thresholds["error_rate_warning"]:
            alerts.append({
                "type": "workflow",
                "level": "warning",
                "message": f"Workflow {workflow_type} error rate is {error_rate:.1%} (threshold: {self.thresholds['error_rate_warning']:.1%})",
                "workflow_type": workflow_type,
                "metric": "error_rate",
                "value": error_rate
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = datetime.now(timezone.utc).isoformat()
            self.alerts.append(alert)
            
            if alert["level"] == "critical":
                logger.error(f"🚨 CRITICAL ALERT: {alert['message']}")
            else:
                logger.warning(f"⚠️ WARNING: {alert['message']}")
    
    def _check_agent_alerts(self, agent_name: str, stats: Dict[str, Any]):
        """Check for agent-level alerts."""
        total_operations = stats["total_operations"]
        
        if total_operations > 0:
            error_rate = stats["error_operations"] / total_operations
            avg_response_time = stats["total_response_time"] / total_operations
            
            alerts = []
            
            # Check agent error rate
            if error_rate > self.thresholds["error_rate_critical"]:
                alerts.append({
                    "type": "agent",
                    "level": "critical",
                    "message": f"Agent {agent_name} error rate is {error_rate:.1%}",
                    "agent_name": agent_name,
                    "metric": "error_rate",
                    "value": error_rate
                })
            
            # Check agent response time
            if avg_response_time > self.thresholds["response_time_warning"]:
                alerts.append({
                    "type": "agent",
                    "level": "warning",
                    "message": f"Agent {agent_name} avg response time is {avg_response_time:.2f}s",
                    "agent_name": agent_name,
                    "metric": "response_time",
                    "value": avg_response_time
                })
            
            # Add alerts with timestamp
            for alert in alerts:
                alert["timestamp"] = datetime.now(timezone.utc).isoformat()
                self.alerts.append(alert)
                
                if alert["level"] == "critical":
                    logger.error(f"🚨 CRITICAL ALERT: {alert['message']}")
                else:
                    logger.warning(f"⚠️ WARNING: {alert['message']}")
    
    async def persist_metrics(self):
        """Persist metrics to file."""
        try:
            metrics_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "performance_summary": self.get_performance_summary(),
                "metrics_history": list(self.metrics_history),
                "workflow_stats": dict(self.workflow_stats),
                "agent_stats": dict(self.agent_stats),
                "alerts": self.alerts[-100:]  # Keep last 100 alerts
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"📊 Metrics persisted to {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"❌ Error persisting metrics: {e}")
    
    async def load_metrics(self):
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Restore data
                if "metrics_history" in metrics_data:
                    self.metrics_history.extend(metrics_data["metrics_history"])
                
                if "workflow_stats" in metrics_data:
                    self.workflow_stats.update(metrics_data["workflow_stats"])
                
                if "agent_stats" in metrics_data:
                    self.agent_stats.update(metrics_data["agent_stats"])
                
                if "alerts" in metrics_data:
                    self.alerts = metrics_data["alerts"]
                
                logger.info("📊 Previous metrics loaded successfully")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load previous metrics: {e}")
    
    def generate_report(self, hours: int = 24) -> str:
        """Generate a text report for the specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Filter metrics for time period
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"].replace('Z', '+00:00')) > cutoff_time
        ]
        
        # Generate report
        report = []
        report.append(f"📊 Workflow Monitoring Report - Last {hours} Hours")
        report.append("=" * 60)
        report.append("")
        
        # System metrics
        if recent_metrics:
            avg_cpu = sum(m["cpu_usage"] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m["response_time"] for m in recent_metrics) / len(recent_metrics)
            
            report.append("🖥️ System Performance:")
            report.append(f"   Average CPU Usage: {avg_cpu:.1f}%")
            report.append(f"   Average Memory Usage: {avg_memory:.1f}%")
            report.append(f"   Average Response Time: {avg_response_time:.2f}s")
            report.append("")
        
        # Workflow metrics
        report.append("🔄 Workflow Performance:")
        for workflow_metrics in self.get_workflow_metrics():
            report.append(f"   {workflow_metrics.workflow_type}:")
            report.append(f"     Executions: {workflow_metrics.execution_count}")
            report.append(f"     Success Rate: {workflow_metrics.success_rate:.1f}%")
            report.append(f"     Avg Duration: {workflow_metrics.avg_duration:.2f}s")
        report.append("")
        
        # Active alerts
        active_alerts = self.get_active_alerts()
        if active_alerts:
            report.append("🚨 Active Alerts:")
            for alert in active_alerts[:5]:  # Top 5 alerts
                report.append(f"   [{alert['level'].upper()}] {alert['message']}")
        else:
            report.append("✅ No Active Alerts")
        
        return "\n".join(report)


# Global monitor instance
workflow_monitor = WorkflowMonitor()


# Convenience functions
async def collect_metrics() -> SystemMetrics:
    """Collect current system metrics."""
    return await workflow_monitor.collect_system_metrics()


def record_workflow_execution(workflow_type: str, success: bool, duration: float, error: Optional[str] = None):
    """Record workflow execution."""
    workflow_monitor.record_workflow_execution(workflow_type, success, duration, error)


def record_agent_activity(agent_name: str, operation: str, success: bool, response_time: float, status: str = "active"):
    """Record agent activity."""
    workflow_monitor.record_agent_activity(agent_name, operation, success, response_time, status)


async def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    return workflow_monitor.get_performance_summary()


async def generate_monitoring_report(hours: int = 24) -> str:
    """Generate monitoring report."""
    return workflow_monitor.generate_report(hours)


# Monitoring service
async def monitoring_service(interval_seconds: int = 300):
    """Run continuous monitoring service."""
    logger.info(f"🔍 Starting workflow monitoring service (interval: {interval_seconds}s)")
    
    # Load previous metrics
    await workflow_monitor.load_metrics()
    
    try:
        while True:
            # Collect metrics
            await workflow_monitor.collect_system_metrics()
            
            # Persist metrics every 10 collections
            if len(workflow_monitor.metrics_history) % 10 == 0:
                await workflow_monitor.persist_metrics()
            
            await asyncio.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("📝 Monitoring service stopped")
    except Exception as e:
        logger.error(f"❌ Monitoring service error: {e}")
    finally:
        # Persist final metrics
        await workflow_monitor.persist_metrics()


if __name__ == "__main__":
    asyncio.run(monitoring_service())
