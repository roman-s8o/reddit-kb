"""
Example script demonstrating Phase 4 - Orchestration functionality.

This script shows how to:
1. Start and manage the workflow scheduler
2. Monitor system performance and metrics
3. Use workflow persistence and recovery
4. Interact with the orchestration API
5. Run complete orchestrated workflows
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any
import httpx
from loguru import logger


async def demo_scheduler_management():
    """Demonstrate scheduler management capabilities."""
    logger.info("🔧 Demo 1: Scheduler Management")
    logger.info("=" * 50)
    
    try:
        from workflow.scheduler import workflow_scheduler, start_scheduler, stop_scheduler
        
        # Start scheduler
        logger.info("🚀 Starting workflow scheduler...")
        scheduler = await start_scheduler()
        
        logger.info(f"✅ Scheduler started with {len(scheduler.jobs_registry)} jobs")
        
        # List all jobs
        logger.info("\n📋 Registered Jobs:")
        for job_id, job in scheduler.jobs_registry.items():
            status = "✅ Enabled" if job.enabled else "❌ Disabled"
            logger.info(f"   {job.name} ({job_id}): {status}")
            logger.info(f"      Trigger: {job.trigger_type} - {job.trigger_config}")
            logger.info(f"      Stats: {job.success_count} successes, {job.error_count} errors")
        
        # Demonstrate job management
        test_job_id = "health_check"
        if test_job_id in scheduler.jobs_registry:
            logger.info(f"\n🔄 Testing job management with '{test_job_id}'...")
            
            # Disable job
            await scheduler.disable_job(test_job_id)
            logger.info(f"   Disabled job: {test_job_id}")
            
            # Re-enable job
            await scheduler.enable_job(test_job_id)
            logger.info(f"   Re-enabled job: {test_job_id}")
        
        # Get scheduler status
        status = await scheduler.get_status()
        logger.info(f"\n📊 Scheduler Status:")
        logger.info(f"   Running: {status.running}")
        logger.info(f"   Total Jobs: {status.total_jobs}")
        logger.info(f"   Active Jobs: {status.active_jobs}")
        logger.info(f"   Uptime: {status.uptime_seconds:.1f} seconds")
        
        # Stop scheduler
        await stop_scheduler()
        logger.info("✅ Scheduler stopped gracefully")
        
    except Exception as e:
        logger.error(f"❌ Scheduler demo failed: {e}")


async def demo_monitoring_system():
    """Demonstrate monitoring and metrics collection."""
    logger.info("📊 Demo 2: Monitoring System")
    logger.info("=" * 50)
    
    try:
        from workflow.monitoring import (
            workflow_monitor, collect_metrics, record_workflow_execution,
            record_agent_activity, get_performance_summary
        )
        
        # Collect system metrics
        logger.info("📈 Collecting system metrics...")
        system_metrics = await collect_metrics()
        
        logger.info(f"   CPU Usage: {system_metrics.cpu_usage:.1f}%")
        logger.info(f"   Memory Usage: {system_metrics.memory_usage:.1f}%")
        logger.info(f"   Response Time: {system_metrics.response_time:.3f}s")
        logger.info(f"   Active Connections: {system_metrics.active_connections}")
        
        # Record some workflow executions
        logger.info("\n🔄 Recording workflow executions...")
        record_workflow_execution("batch_workflow", True, 45.2)
        record_workflow_execution("chat_workflow", True, 2.1)
        record_workflow_execution("batch_workflow", False, 12.5, "Network timeout")
        record_workflow_execution("insight_generation", True, 38.7)
        
        # Record agent activities
        logger.info("🤖 Recording agent activities...")
        record_agent_activity("collector", "fetch_posts", True, 5.2, "active")
        record_agent_activity("preprocessor", "embed_text", True, 2.8, "active")
        record_agent_activity("insight_agent", "cluster_analysis", True, 15.3, "active")
        record_agent_activity("chatbot", "generate_response", True, 3.1, "active")
        
        # Get performance summary
        logger.info("\n📋 Performance Summary:")
        summary = await get_performance_summary()
        
        overall = summary["overall_metrics"]
        logger.info(f"   Total Executions: {overall['total_executions']}")
        logger.info(f"   Success Rate: {overall['success_rate']:.1f}%")
        logger.info(f"   Avg Response Time: {overall['avg_response_time']:.3f}s")
        
        # Show workflow metrics
        logger.info("\n🔄 Workflow Metrics:")
        for workflow in summary["workflow_metrics"]:
            logger.info(f"   {workflow['workflow_type']}:")
            logger.info(f"     Executions: {workflow['execution_count']}")
            logger.info(f"     Success Rate: {workflow['success_rate']:.1f}%")
            logger.info(f"     Avg Duration: {workflow['avg_duration']:.2f}s")
        
        # Show active alerts
        alerts = summary["active_alerts"]
        if alerts:
            logger.info(f"\n🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts[:3]:
                logger.info(f"   [{alert['level'].upper()}] {alert['message']}")
        else:
            logger.info("\n✅ No active alerts")
        
        # Generate report
        logger.info("\n📄 Generating monitoring report...")
        report = workflow_monitor.generate_report(hours=1)
        logger.info("Report generated (truncated):")
        logger.info(report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        logger.error(f"❌ Monitoring demo failed: {e}")


async def demo_persistence_system():
    """Demonstrate workflow persistence and recovery."""
    logger.info("💾 Demo 3: Persistence System")
    logger.info("=" * 50)
    
    try:
        from workflow.persistence import (
            workflow_persistence, start_workflow_tracking, create_workflow_checkpoint,
            complete_workflow_tracking, get_persistence_statistics
        )
        
        # Start a workflow execution
        execution_id = f"demo_workflow_{int(datetime.now().timestamp())}"
        logger.info(f"🚀 Starting workflow execution: {execution_id}")
        
        execution = await start_workflow_tracking(
            execution_id,
            "batch",
            {"subreddits": ["Python", "MachineLearning"], "max_posts": 50}
        )
        
        logger.info(f"   Execution ID: {execution.execution_id}")
        logger.info(f"   Workflow Type: {execution.workflow_type}")
        logger.info(f"   Started At: {execution.started_at}")
        
        # Create checkpoints for each step
        steps = [
            ("collection_started", "preprocessing_start", {"collected_posts": 0}),
            ("collection_completed", "preprocessing_start", {"collected_posts": 75}),
            ("preprocessing_completed", "insight_generation_start", {"processed_chunks": 150}),
            ("insight_generation_completed", None, {"insights_generated": 5})
        ]
        
        logger.info("\n📍 Creating workflow checkpoints...")
        for i, (step_completed, next_step, state_data) in enumerate(steps, 1):
            checkpoint = await create_workflow_checkpoint(
                execution_id,
                "batch",
                {"step": i, "progress": i/len(steps), **state_data},
                step_completed,
                next_step,
                success=True,
                metadata={"step_number": i, "total_steps": len(steps)}
            )
            logger.info(f"   ✅ Checkpoint {i}: {step_completed}")
        
        # Complete the workflow
        logger.info("\n✅ Completing workflow execution...")
        await complete_workflow_tracking(
            execution_id,
            "completed",
            {"total_documents": 150, "insights_count": 5, "duration": 67.3}
        )
        
        # Get execution details
        logger.info("\n📋 Workflow Execution Details:")
        retrieved_execution = await workflow_persistence.get_workflow_execution(execution_id)
        if retrieved_execution:
            logger.info(f"   Status: {retrieved_execution.status}")
            logger.info(f"   Checkpoints: {len(retrieved_execution.checkpoints)}")
            logger.info(f"   Final Result: {retrieved_execution.final_result}")
        
        # Demonstrate recovery (simulate failure)
        failed_execution_id = f"failed_workflow_{int(datetime.now().timestamp())}"
        logger.info(f"\n🔄 Demonstrating recovery with: {failed_execution_id}")
        
        # Start failed workflow
        await start_workflow_tracking(failed_execution_id, "batch", {"test": True})
        await create_workflow_checkpoint(
            failed_execution_id, "batch", {"step": 1}, "collection_completed", "preprocessing_start"
        )
        await complete_workflow_tracking(failed_execution_id, "failed")
        
        # Attempt recovery
        recovery_success = await workflow_persistence.recover_workflow_execution(failed_execution_id)
        logger.info(f"   Recovery attempt: {'✅ Success' if recovery_success else '❌ Failed'}")
        
        # Get persistence statistics
        logger.info("\n📊 Persistence Statistics:")
        stats = await get_persistence_statistics()
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"   {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"     {sub_key}: {sub_value}")
            else:
                logger.info(f"   {key}: {value}")
        
    except Exception as e:
        logger.error(f"❌ Persistence demo failed: {e}")


async def demo_orchestration_api():
    """Demonstrate orchestration API functionality."""
    logger.info("🌐 Demo 4: Orchestration API")
    logger.info("=" * 50)
    
    api_url = "http://localhost:8002"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test API health
            logger.info("🔍 Testing API health...")
            response = await client.get(f"{api_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"   API Status: {health_data['status']}")
                logger.info(f"   Scheduler Running: {health_data['scheduler_running']}")
            else:
                logger.warning(f"   API not available (status: {response.status_code})")
                return
            
            # Get scheduler status
            logger.info("\n📊 Getting scheduler status...")
            response = await client.get(f"{api_url}/scheduler/status")
            if response.status_code == 200:
                data = response.json()
                status = data["data"]
                logger.info(f"   Total Jobs: {status['total_jobs']}")
                logger.info(f"   Active Jobs: {status['active_jobs']}")
                logger.info(f"   Uptime: {status['uptime_seconds']:.1f}s")
            
            # List jobs
            logger.info("\n📋 Listing jobs...")
            response = await client.get(f"{api_url}/scheduler/jobs")
            if response.status_code == 200:
                data = response.json()
                jobs = data["data"]["jobs"]
                for job in jobs[:3]:  # Show first 3 jobs
                    logger.info(f"   {job['name']}: {'✅' if job['enabled'] else '❌'}")
                    logger.info(f"     Success: {job['success_count']}, Errors: {job['error_count']}")
            
            # Get system health
            logger.info("\n🏥 System health check...")
            response = await client.get(f"{api_url}/system/health")
            if response.status_code == 200:
                data = response.json()
                health_checks = data["data"]["health_checks"]
                logger.info("   Health Checks:")
                for check, status in health_checks.items():
                    logger.info(f"     {check}: {'✅' if status else '❌'}")
            
            # Get metrics
            logger.info("\n📈 System metrics...")
            response = await client.get(f"{api_url}/monitoring/metrics")
            if response.status_code == 200:
                data = response.json()
                scheduler_metrics = data["data"]["scheduler_metrics"]
                logger.info(f"   Success Rate: {scheduler_metrics['success_rate']:.1f}%")
                logger.info(f"   Total Executions: {scheduler_metrics['total_executions']}")
                logger.info(f"   Uptime: {scheduler_metrics['uptime_seconds']:.1f}s")
            
            # Trigger a job manually (if jobs exist)
            response = await client.get(f"{api_url}/scheduler/jobs")
            if response.status_code == 200:
                jobs_data = response.json()
                jobs = jobs_data["data"]["jobs"]
                if jobs:
                    test_job = jobs[0]
                    logger.info(f"\n🔄 Triggering job manually: {test_job['name']}")
                    
                    trigger_payload = {"job_id": test_job["id"]}
                    response = await client.post(
                        f"{api_url}/scheduler/jobs/trigger",
                        json=trigger_payload
                    )
                    if response.status_code == 200:
                        logger.info("   ✅ Job triggered successfully")
                    else:
                        logger.warning(f"   ⚠️ Job trigger failed: {response.status_code}")
            
            logger.info("\n✅ API demo completed successfully")
        
    except httpx.ConnectError:
        logger.warning("⚠️ Orchestration API not available. Start it with:")
        logger.warning("   python api/orchestration_api.py")
    except Exception as e:
        logger.error(f"❌ API demo failed: {e}")


async def demo_complete_orchestration():
    """Demonstrate complete orchestration workflow."""
    logger.info("🎯 Demo 5: Complete Orchestration")
    logger.info("=" * 50)
    
    try:
        from workflow.reddit_workflow import run_data_collection_and_analysis, ask_reddit_question
        from workflow.monitoring import record_workflow_execution
        from workflow.persistence import start_workflow_tracking, complete_workflow_tracking
        
        # Demo 1: Batch workflow with monitoring and persistence
        logger.info("🔄 Running monitored batch workflow...")
        
        execution_id = f"orchestrated_batch_{int(datetime.now().timestamp())}"
        await start_workflow_tracking(execution_id, "batch", {"subreddits": ["Python"]})
        
        start_time = datetime.now()
        
        try:
            # Run batch workflow
            result = await run_data_collection_and_analysis(subreddits=["Python"])
            
            duration = (datetime.now() - start_time).total_seconds()
            success = result.get("success", False)
            
            # Record metrics
            record_workflow_execution("batch_orchestrated", success, duration)
            
            # Complete tracking
            await complete_workflow_tracking(
                execution_id,
                "completed" if success else "failed",
                result
            )
            
            if success:
                logger.info(f"   ✅ Batch workflow completed in {duration:.2f}s")
                logger.info(f"   📊 Processed data: {result.get('processed_data', {}).get('statistics', {})}")
                logger.info(f"   💡 Generated insights: {result.get('insights_data', {}).get('clusters', 0)} clusters")
            else:
                logger.error(f"   ❌ Batch workflow failed: {result.get('error')}")
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            record_workflow_execution("batch_orchestrated", False, duration, str(e))
            await complete_workflow_tracking(execution_id, "failed", {"error": str(e)})
            logger.error(f"   ❌ Batch workflow error: {e}")
        
        # Demo 2: Chat workflow with monitoring
        logger.info("\n💬 Running monitored chat workflow...")
        
        start_time = datetime.now()
        
        try:
            # Run chat workflow
            result = await ask_reddit_question(
                "What are the latest trends in Python programming?",
                subreddits=["Python"]
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            success = result.get("success", False)
            
            # Record metrics
            record_workflow_execution("chat_orchestrated", success, duration)
            
            if success:
                chat_response = result.get("chat_response", {})
                logger.info(f"   ✅ Chat workflow completed in {duration:.2f}s")
                logger.info(f"   🤖 Response length: {len(chat_response.get('response', ''))} chars")
                logger.info(f"   📚 Sources found: {len(chat_response.get('sources', []))}")
                logger.info(f"   🎯 Confidence: {chat_response.get('confidence', 0):.2f}")
            else:
                logger.error(f"   ❌ Chat workflow failed: {result.get('error')}")
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            record_workflow_execution("chat_orchestrated", False, duration, str(e))
            logger.error(f"   ❌ Chat workflow error: {e}")
        
        logger.info("\n✅ Complete orchestration demo finished")
        
    except Exception as e:
        logger.error(f"❌ Complete orchestration demo failed: {e}")


async def main():
    """Run all orchestration demos."""
    logger.info("🚀 Phase 4 - Orchestration Examples")
    logger.info("🎯 Demonstrating complete workflow orchestration capabilities")
    
    demos = [
        ("Scheduler Management", demo_scheduler_management),
        ("Monitoring System", demo_monitoring_system),
        ("Persistence System", demo_persistence_system),
        ("Orchestration API", demo_orchestration_api),
        ("Complete Orchestration", demo_complete_orchestration),
    ]
    
    for demo_name, demo_func in demos:
        try:
            await demo_func()
            logger.info("\n" + "=" * 60 + "\n")
        except Exception as e:
            logger.error(f"❌ Demo '{demo_name}' failed: {e}")
            logger.info("\n" + "=" * 60 + "\n")
    
    logger.info("🎉 All orchestration demos completed!")
    logger.info("\n📝 Phase 4 - Orchestration Features Demonstrated:")
    logger.info("   ✅ Workflow scheduling and job management")
    logger.info("   ✅ System monitoring and metrics collection")
    logger.info("   ✅ Workflow state persistence and recovery")
    logger.info("   ✅ Orchestration API management")
    logger.info("   ✅ Complete workflow orchestration")
    
    logger.info("\n🎯 Phase 4 is complete! Ready for Phase 5 - Frontend development.")
    
    logger.info("\n📚 Available Services:")
    logger.info("   🔧 Orchestration API: python api/orchestration_api.py (port 8002)")
    logger.info("   💬 Chatbot API: python chatbot_api.py (port 8001)")
    logger.info("   🌐 Main API: python api/main.py (port 8000)")
    logger.info("   ⏰ Standalone Scheduler: python workflow/scheduler.py")
    logger.info("   📊 Standalone Monitoring: python workflow/monitoring.py")


if __name__ == "__main__":
    asyncio.run(main())
