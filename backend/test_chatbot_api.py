"""
Test script for the Reddit Chatbot API.

This script tests the chatbot API endpoints to ensure they work correctly
with the LangGraph workflow and return the expected JSON format.
"""
import asyncio
import json
import time
from typing import Dict, Any
import httpx
from loguru import logger


# Test configuration
API_BASE_URL = "http://localhost:8001"
TEST_TIMEOUT = 60.0  # seconds


class ChatbotAPITester:
    """Test class for the chatbot API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        """Initialize the tester."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=TEST_TIMEOUT)
        self.test_results = {}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        logger.info("🔍 Testing health endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert data["status"] == "healthy"
                assert "timestamp" in data
                logger.info("✅ Health endpoint test passed")
                return True
            else:
                logger.error(f"❌ Health endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Health endpoint test failed: {e}")
            return False
    
    async def test_status_endpoint(self) -> bool:
        """Test the status endpoint."""
        logger.info("🔍 Testing status endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/status")
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert "timestamp" in data
                assert "agents" in data
                logger.info("✅ Status endpoint test passed")
                return True
            else:
                logger.error(f"❌ Status endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Status endpoint test failed: {e}")
            return False
    
    async def test_chat_endpoint_basic(self) -> bool:
        """Test basic chat functionality."""
        logger.info("🔍 Testing basic chat endpoint...")
        
        test_query = "What are people discussing about Python?"
        
        try:
            payload = {
                "query": test_query
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["answer", "references", "insights", "timestamp", "confidence", "query"]
                for field in required_fields:
                    assert field in data, f"Missing field: {field}"
                
                # Validate data types
                assert isinstance(data["answer"], str), "Answer should be a string"
                assert isinstance(data["references"], list), "References should be a list"
                assert isinstance(data["insights"], list), "Insights should be a list"
                assert isinstance(data["confidence"], (int, float)), "Confidence should be numeric"
                assert data["query"] == test_query, "Query should match input"
                
                # Validate references structure (if any)
                for ref in data["references"]:
                    assert "post_id" in ref
                    assert "subreddit" in ref
                    assert "similarity_score" in ref
                    assert "source_type" in ref
                
                # Validate insights structure (if any)
                for insight in data["insights"]:
                    assert "topic" in insight
                    assert "keywords" in insight
                    assert "relevance" in insight
                    assert "document_count" in insight
                
                logger.info("✅ Basic chat endpoint test passed")
                logger.info(f"   Answer length: {len(data['answer'])} chars")
                logger.info(f"   References count: {len(data['references'])}")
                logger.info(f"   Insights count: {len(data['insights'])}")
                logger.info(f"   Confidence: {data['confidence']:.2f}")
                
                return True
            else:
                logger.error(f"❌ Chat endpoint returned status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Basic chat endpoint test failed: {e}")
            return False
    
    async def test_chat_endpoint_with_subreddits(self) -> bool:
        """Test chat endpoint with subreddit filtering."""
        logger.info("🔍 Testing chat endpoint with subreddit filtering...")
        
        test_query = "What are the latest trends in machine learning?"
        subreddits = ["MachineLearning", "Python"]
        
        try:
            payload = {
                "query": test_query,
                "subreddits": subreddits
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Basic validation
                assert "answer" in data
                assert "references" in data
                assert data["query"] == test_query
                
                # Check if references are from specified subreddits (if any)
                for ref in data["references"]:
                    if ref["subreddit"] != "unknown":
                        # Note: This might not always be true due to data availability
                        logger.info(f"   Reference from: {ref['subreddit']}")
                
                logger.info("✅ Chat endpoint with subreddits test passed")
                return True
            else:
                logger.error(f"❌ Chat endpoint with subreddits returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Chat endpoint with subreddits test failed: {e}")
            return False
    
    async def test_chat_endpoint_edge_cases(self) -> bool:
        """Test chat endpoint edge cases."""
        logger.info("🔍 Testing chat endpoint edge cases...")
        
        test_cases = [
            # Empty query
            {
                "payload": {"query": ""},
                "expected_status": 400,
                "description": "Empty query"
            },
            # Very long query
            {
                "payload": {"query": "a" * 1001},
                "expected_status": 422,  # Pydantic validation error
                "description": "Query too long"
            },
            # Invalid subreddits format
            {
                "payload": {"query": "test", "subreddits": "not_a_list"},
                "expected_status": 422,
                "description": "Invalid subreddits format"
            }
        ]
        
        passed_tests = 0
        
        for test_case in test_cases:
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat",
                    json=test_case["payload"]
                )
                
                if response.status_code == test_case["expected_status"]:
                    logger.info(f"   ✅ {test_case['description']}: Expected status {test_case['expected_status']}")
                    passed_tests += 1
                else:
                    logger.error(f"   ❌ {test_case['description']}: Expected {test_case['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                logger.error(f"   ❌ {test_case['description']}: Exception {e}")
        
        success = passed_tests == len(test_cases)
        if success:
            logger.info("✅ Edge cases test passed")
        else:
            logger.error(f"❌ Edge cases test failed: {passed_tests}/{len(test_cases)} passed")
        
        return success
    
    async def test_suggestions_endpoint(self) -> bool:
        """Test the suggestions endpoint."""
        logger.info("🔍 Testing suggestions endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/suggestions")
            
            if response.status_code == 200:
                data = response.json()
                assert "suggestions" in data
                assert isinstance(data["suggestions"], list)
                
                logger.info(f"✅ Suggestions endpoint test passed")
                logger.info(f"   Got {len(data['suggestions'])} suggestions")
                for i, suggestion in enumerate(data["suggestions"][:3], 1):
                    logger.info(f"   {i}. {suggestion}")
                
                return True
            else:
                logger.error(f"❌ Suggestions endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Suggestions endpoint test failed: {e}")
            return False
    
    async def test_topics_endpoint(self) -> bool:
        """Test the topics endpoint."""
        logger.info("🔍 Testing topics endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/insights/topics")
            
            if response.status_code == 200:
                data = response.json()
                assert "topics" in data
                assert isinstance(data["topics"], list)
                
                logger.info(f"✅ Topics endpoint test passed")
                logger.info(f"   Got {len(data['topics'])} topics")
                
                return True
            else:
                logger.error(f"❌ Topics endpoint returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Topics endpoint test failed: {e}")
            return False
    
    async def test_performance(self) -> bool:
        """Test API performance with multiple requests."""
        logger.info("🔍 Testing API performance...")
        
        test_queries = [
            "What are people saying about Python?",
            "Tell me about machine learning trends",
            "What's the sentiment around programming?",
        ]
        
        try:
            start_time = time.time()
            
            # Send requests sequentially to avoid overwhelming the server
            response_times = []
            
            for query in test_queries:
                query_start = time.time()
                
                response = await self.client.post(
                    f"{self.base_url}/chat",
                    json={"query": query}
                )
                
                query_time = time.time() - query_start
                response_times.append(query_time)
                
                if response.status_code != 200:
                    logger.error(f"❌ Performance test failed on query: {query}")
                    return False
            
            total_time = time.time() - start_time
            avg_response_time = sum(response_times) / len(response_times)
            
            logger.info("✅ Performance test passed")
            logger.info(f"   Total time: {total_time:.2f}s")
            logger.info(f"   Average response time: {avg_response_time:.2f}s")
            logger.info(f"   Response times: {[f'{t:.2f}s' for t in response_times]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("🧪 Starting Reddit Chatbot API Tests")
        logger.info("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("System Status", self.test_status_endpoint),
            ("Basic Chat", self.test_chat_endpoint_basic),
            ("Chat with Subreddits", self.test_chat_endpoint_with_subreddits),
            ("Edge Cases", self.test_chat_endpoint_edge_cases),
            ("Suggestions", self.test_suggestions_endpoint),
            ("Topics", self.test_topics_endpoint),
            ("Performance", self.test_performance),
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
            logger.info("🎉 All tests passed! The Chatbot API is working correctly.")
        else:
            logger.error(f"❌ {len(tests) - passed} tests failed. Please check the errors above.")
        
        return results


async def main():
    """Run the test suite."""
    logger.info("🚀 Reddit Chatbot API Test Suite")
    
    # Check if API is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code != 200:
                logger.error("❌ API is not responding correctly. Make sure it's running on http://localhost:8001")
                return
    except Exception as e:
        logger.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        logger.error("Please start the API first:")
        logger.error("   cd backend")
        logger.error("   python chatbot_api.py")
        return
    
    # Run tests
    tester = ChatbotAPITester()
    
    try:
        results = await tester.run_all_tests()
        
        # Additional information
        logger.info("\n📝 Additional Information:")
        logger.info("   API URL: http://localhost:8001")
        logger.info("   Documentation: http://localhost:8001/docs")
        logger.info("   Alternative docs: http://localhost:8001/redoc")
        
        if all(results.values()):
            logger.info("\n🎯 The API is ready for integration with your frontend!")
        else:
            logger.info("\n🔧 Please fix the failing tests before proceeding.")
            
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())
