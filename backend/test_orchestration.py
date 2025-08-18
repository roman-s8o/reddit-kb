"""
Test suite for Phase 4 - Orchestration functionality.

This script tests:
- Workflow scheduler functionality
- Job management and execution
- Monitoring and metrics collection
- API endpoints for orchestration
- System health checks
"""
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any
import httpx
from loguru import logger


# Test configuration
ORCHESTRATION_API_URL = "http://localhost:8002"
TEST_TIMEOUT = 120.0  # seconds


class OrchestrationTester:
    """Test class for orchestration functionality."""
    
    def __init__(self, api_url: str = ORCHESTRATION_API_URL):
        """Initialize the tester."""
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=TEST_TIMEOUT)
        self.test_results = {}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def test_scheduler_initialization(self) -> bool:
        """Test scheduler initialization and basic functionality."""
        logger.info("🔍 Testing scheduler initialization...")
        
        try:
            # Test importing scheduler
            from workflow.scheduler import workflow_scheduler, start_scheduler, stop_scheduler
            
            # Test scheduler creation
            assert workflow_scheduler is not None, "Scheduler instance not created"
            
            # Test scheduler initialization (without starting)
            await workflow_scheduler.initialize()
            
            # Check jobs registry
            assert len(workflow_scheduler.jobs_registry) > 0, "No jobs registered"
            
            logger.info(f"✅ Scheduler initialized with {len(workflow_scheduler.jobs_registry)} jobs")
            
            # Test job management
            job_ids = list(workflow_scheduler.jobs_registry.keys())
            test_job_id = job_ids[0] if job_ids else None
            
            if test_job_id:
                # Test disable/enable job
                await workflow_scheduler.disable_job(test_job_id)
                assert not workflow_scheduler.jobs_registry[test_job_id].enabled, "Job not disabled"
                
                await workflow_scheduler.enable_job(test_job_id)
                assert workflow_scheduler.jobs_registry[test_job_id].enabled, "Job not enabled"
            
            logger.info("✅ Scheduler initialization test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Scheduler initialization test failed: {e}")
            return False
    
    async def test_monitoring_system(self) -> bool:
        """Test monitoring and metrics collection."""
        logger.info("🔍 Testing monitoring system...")
        
        try:
            from workflow.monitoring import workflow_monitor, collect_metrics, record_workflow_execution
            
            # Test metrics collection
            system_metrics = await collect_metrics()
            assert system_metrics is not None, "System metrics not collected"
            assert hasattr(system_metrics, 'timestamp'), "Metrics missing timestamp"
            assert hasattr(system_metrics, 'response_time'), "Metrics missing response_time"
            
            # Test workflow recording
            record_workflow_execution("test_workflow", True, 2.5)
            workflow_metrics = workflow_monitor.get_workflow_metrics()
            assert len(workflow_metrics) > 0, "No workflow metrics recorded"
            
            # Test performance summary
            summary = workflow_monitor.get_performance_summary()
            assert "overall_metrics" in summary, "Performance summary missing overall_metrics"
            assert "workflow_metrics" in summary, "Performance summary missing workflow_metrics"
            
            logger.info("✅ Monitoring system test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring system test failed: {e}")
            return False
    
    async def test_orchestration_api_health(self) -> bool:
        """Test orchestration API health endpoints."""
        logger.info("🔍 Testing orchestration API health...")
        
        try:
            # Test health endpoint
            response = await self.client.get(f"{self.api_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data, "Health response missing status"
                assert data["status"] == "healthy", "API not healthy"
                
                logger.info("✅ Orchestration API health test passed")
                return True
            else:
                logger.error(f"❌ Health endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Orchestration API health test failed: {e}")
            return False
    
    async def test_scheduler_status_api(self) -> bool:
        """Test scheduler status API endpoint."""
        logger.info("🔍 Testing scheduler status API...")
        
        try:
            response = await self.client.get(f"{self.api_url}/scheduler/status")
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"], "API response not successful"
                assert "data" in data, "Response missing data"
                
                status_data = data["data"]
                assert "running" in status_data, "Status missing running field"
                assert "jobs" in status_data, "Status missing jobs field"
                assert "total_jobs" in status_data, "Status missing total_jobs field"
                
                logger.info(f"✅ Scheduler status API test passed - {status_data['total_jobs']} jobs found")
                return True
            else:
                logger.error(f"❌ Scheduler status API returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Scheduler status API test failed: {e}")
            return False
    
    async def test_job_management_api(self) -> bool:
        """Test job management API endpoints."""
        logger.info("🔍 Testing job management API...")
        
        try:
            # First, get list of jobs
            response = await self.client.get(f"{self.api_url}/scheduler/jobs")
            
            if response.status_code != 200:
                logger.error(f"❌ Jobs list API returned status {response.status_code}")
                return False
            
            jobs_data = response.json()
            assert jobs_data["success"], "Jobs list API not successful"
            
            jobs = jobs_data["data"]["jobs"]
            if not jobs:
                logger.warning("⚠️ No jobs available for management testing")
                return True
            
            # Test job management with first job
            test_job = jobs[0]
            job_id = test_job["id"]
            
            # Test disable job
            disable_payload = {"job_id": job_id, "action": "disable"}
            response = await self.client.post(
                f"{self.api_url}/scheduler/jobs/manage",
                json=disable_payload
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully disabled job: {job_id}")
            else:
                logger.error(f"❌ Failed to disable job: {response.status_code}")
                return False
            
            # Test enable job
            enable_payload = {"job_id": job_id, "action": "enable"}
            response = await self.client.post(
                f"{self.api_url}/scheduler/jobs/manage",
                json=enable_payload
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully enabled job: {job_id}")
            else:
                logger.error(f"❌ Failed to enable job: {response.status_code}")
                return False
            
            logger.info("✅ Job management API test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Job management API test failed: {e}")
            return False
    
    async def test_workflow_execution_api(self) -> bool:
        """Test workflow execution API."""
        logger.info("🔍 Testing workflow execution API...")
        
        try:
            # Test batch workflow execution
            batch_payload = {
                "workflow_type": "batch",
                "subreddits": ["Python"],
                "parameters": {"test": True}
            }
            
            response = await self.client.post(
                f"{self.api_url}/workflow/execute",
                json=batch_payload
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"], "Batch workflow execution not successful"
                logger.info("✅ Batch workflow execution started successfully")
            else:
                logger.error(f"❌ Batch workflow execution failed: {response.status_code}")
                return False
            
            # Test chat workflow execution
            chat_payload = {
                "workflow_type": "chat",
                "query": "What are people saying about Python?",
                "subreddits": ["Python"]
            }
            
            response = await self.client.post(
                f"{self.api_url}/workflow/execute",
                json=chat_payload
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"], "Chat workflow execution not successful"
                logger.info("✅ Chat workflow execution started successfully")
            else:
                logger.error(f"❌ Chat workflow execution failed: {response.status_code}")
                return False
            
            logger.info("✅ Workflow execution API test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow execution API test failed: {e}")
            return False
    
    async def test_monitoring_api(self) -> bool:
        """Test monitoring API endpoints."""
        logger.info("🔍 Testing monitoring API...")
        
        try:
            # Test system health check
            response = await self.client.get(f"{self.api_url}/system/health")
            
            if response.status_code == 200:
                data = response.json()
                assert "data" in data, "Health check response missing data"
                
                health_data = data["data"]
                assert "system_status" in health_data, "Health data missing system_status"
                assert "scheduler_status" in health_data, "Health data missing scheduler_status"
                assert "health_checks" in health_data, "Health data missing health_checks"
                
                logger.info("✅ System health check API passed")
            else:
                logger.error(f"❌ System health check failed: {response.status_code}")
                return False
            
            # Test metrics endpoint
            response = await self.client.get(f"{self.api_url}/monitoring/metrics")
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"], "Metrics API not successful"
                
                metrics_data = data["data"]
                assert "scheduler_metrics" in metrics_data, "Metrics missing scheduler_metrics"
                assert "system_metrics" in metrics_data, "Metrics missing system_metrics"
                
                logger.info("✅ Monitoring metrics API passed")
            else:
                logger.error(f"❌ Monitoring metrics failed: {response.status_code}")
                return False
            
            logger.info("✅ Monitoring API test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring API test failed: {e}")
            return False
    
    async def test_workflow_history_api(self) -> bool:
        """Test workflow history API."""
        logger.info("🔍 Testing workflow history API...")
        
        try:
            response = await self.client.get(f"{self.api_url}/workflow/history?limit=5")
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"], "Workflow history API not successful"
                assert "data" in data, "Response missing data"
                
                history_data = data["data"]
                assert "history" in history_data, "History data missing history field"
                assert "total_entries" in history_data, "History data missing total_entries"
                
                logger.info(f"✅ Workflow history API passed - {history_data['total_entries']} entries")
                return True
            else:
                logger.error(f"❌ Workflow history API failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Workflow history API test failed: {e}")
            return False
    
    async def test_scheduler_configuration_api(self) -> bool:
        """Test scheduler configuration API."""
        logger.info("🔍 Testing scheduler configuration API...")
        
        try:
            # Test configuration update
            config_payload = {
                "enable_hourly_collection": False,
                "enable_health_checks": True
            }
            
            response = await self.client.post(
                f"{self.api_url}/scheduler/config",
                json=config_payload
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"], "Configuration update not successful"
                
                changes = data["data"]["changes"]
                logger.info(f"✅ Configuration updated: {changes}")
                return True
            else:
                logger.error(f"❌ Configuration update failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Scheduler configuration API test failed: {e}")
            return False
    
    async def test_performance_under_load(self) -> bool:
        """Test system performance under concurrent requests."""
        logger.info("🔍 Testing performance under load...")
        
        try:
            start_time = time.time()
            
            # Send multiple concurrent requests
            tasks = []
            for i in range(5):
                tasks.append(self.client.get(f"{self.api_url}/scheduler/status"))
                tasks.append(self.client.get(f"{self.api_url}/scheduler/jobs"))
                tasks.append(self.client.get(f"{self.api_url}/system/health"))
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # Check results
            successful_requests = 0
            for response in responses:
                if isinstance(response, httpx.Response) and response.status_code == 200:
                    successful_requests += 1
                elif isinstance(response, Exception):
                    logger.warning(f"Request failed: {response}")
            
            success_rate = successful_requests / len(tasks) * 100
            
            logger.info(f"✅ Performance test completed:")
            logger.info(f"   Total requests: {len(tasks)}")
            logger.info(f"   Successful: {successful_requests}")
            logger.info(f"   Success rate: {success_rate:.1f}%")
            logger.info(f"   Total duration: {duration:.2f}s")
            logger.info(f"   Avg response time: {duration/len(tasks):.3f}s")
            
            # Consider test passed if success rate > 80%
            return success_rate > 80.0
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all orchestration tests."""
        logger.info("🧪 Starting Orchestration Tests (Phase 4)")
        logger.info("=" * 60)
        
        tests = [
            ("Scheduler Initialization", self.test_scheduler_initialization),
            ("Monitoring System", self.test_monitoring_system),
            ("API Health Check", self.test_orchestration_api_health),
            ("Scheduler Status API", self.test_scheduler_status_api),
            ("Job Management API", self.test_job_management_api),
            ("Workflow Execution API", self.test_workflow_execution_api),
            ("Monitoring API", self.test_monitoring_api),
            ("Workflow History API", self.test_workflow_history_api),
            ("Configuration API", self.test_scheduler_configuration_api),
            ("Performance Under Load", self.test_performance_under_load),
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"❌ Test {test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 Test Results Summary:")
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {test_name}: {status}")
        
        logger.info(f"\n📈 Overall: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            logger.info("🎉 All orchestration tests passed! Phase 4 is working correctly.")
        else:
            logger.error(f"❌ {len(tests) - passed} tests failed. Please check the errors above.")
        
        return results


async def main():
    """Run the orchestration test suite."""
    logger.info("🚀 Phase 4 - Orchestration Test Suite")
    
    # Check if orchestration API is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ORCHESTRATION_API_URL}/health")
            if response.status_code != 200:
                logger.error("❌ Orchestration API is not responding correctly.")
                logger.error("Please start the API first:")
                logger.error("   cd backend")
                logger.error("   python api/orchestration_api.py")
                return
    except Exception as e:
        logger.error(f"❌ Cannot connect to Orchestration API at {ORCHESTRATION_API_URL}")
        logger.error("Please start the API first:")
        logger.error("   cd backend")
        logger.error("   python api/orchestration_api.py")
        return
    
    # Run tests
    tester = OrchestrationTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Additional information
        logger.info("\n📝 Additional Information:")
        logger.info("   Orchestration API URL: http://localhost:8002")
        logger.info("   Documentation: http://localhost:8002/docs")
        logger.info("   Scheduler Service: Can be run standalone with 'python workflow/scheduler.py'")
        logger.info("   Monitoring Service: Can be run standalone with 'python workflow/monitoring.py'")
        
        if all(results.values()):
            logger.info("\n🎯 Phase 4 - Orchestration is complete and ready!")
            logger.info("   ✅ Workflow scheduling implemented")
            logger.info("   ✅ Job management working")
            logger.info("   ✅ Monitoring and metrics active")
            logger.info("   ✅ API endpoints functional")
        else:
            logger.info("\n🔧 Please fix the failing tests before proceeding to Phase 5.")
            
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
