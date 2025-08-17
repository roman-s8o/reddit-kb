"""
Example script demonstrating how to use the Reddit Chatbot API.

This script shows how to:
1. Send chat queries to the API
2. Handle responses with answer, references, and insights
3. Work with subreddit filtering
4. Use suggestions and topics endpoints
"""
import asyncio
import json
from typing import List, Optional
import httpx
from loguru import logger


class RedditChatbotClient:
    """Client for interacting with the Reddit Chatbot API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """Initialize the client."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def chat(self, query: str, subreddits: Optional[List[str]] = None) -> dict:
        """
        Send a chat query to the API.
        
        Args:
            query: User's question
            subreddits: Optional list of subreddits to filter by
            
        Returns:
            API response with answer, references, and insights
        """
        payload = {"query": query}
        if subreddits:
            payload["subreddits"] = subreddits
        
        response = await self.client.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def get_suggestions(self, subreddits: Optional[List[str]] = None) -> List[str]:
        """Get query suggestions."""
        params = {}
        if subreddits:
            params["subreddits"] = subreddits
        
        response = await self.client.get(f"{self.base_url}/suggestions", params=params)
        response.raise_for_status()
        return response.json()["suggestions"]
    
    async def get_topics(self, subreddits: Optional[List[str]] = None) -> List[dict]:
        """Get available topics."""
        params = {}
        if subreddits:
            params["subreddits"] = subreddits
        
        response = await self.client.get(f"{self.base_url}/insights/topics", params=params)
        response.raise_for_status()
        return response.json()["topics"]
    
    async def get_status(self) -> dict:
        """Get API status."""
        response = await self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()


def print_chat_response(response: dict):
    """Pretty print a chat response."""
    logger.info("🤖 Chat Response:")
    logger.info(f"   Query: {response['query']}")
    logger.info(f"   Confidence: {response['confidence']:.2f}")
    logger.info(f"   Timestamp: {response['timestamp']}")
    
    logger.info(f"\n💬 Answer:")
    logger.info(f"   {response['answer']}")
    
    if response['references']:
        logger.info(f"\n📚 References ({len(response['references'])}):")
        for i, ref in enumerate(response['references'], 1):
            logger.info(f"   {i}. [{ref['source_type']}] r/{ref['subreddit']} (score: {ref['similarity_score']:.3f})")
            if ref.get('title'):
                logger.info(f"      Title: {ref['title']}")
            if ref.get('url'):
                logger.info(f"      URL: {ref['url']}")
    
    if response['insights']:
        logger.info(f"\n💡 Related Insights ({len(response['insights'])}):")
        for i, insight in enumerate(response['insights'], 1):
            logger.info(f"   {i}. {insight['topic']} (relevance: {insight['relevance']:.2f})")
            logger.info(f"      Keywords: {', '.join(insight['keywords'])}")
            logger.info(f"      Documents: {insight['document_count']}")


async def demo_basic_chat():
    """Demonstrate basic chat functionality."""
    logger.info("🔍 Demo 1: Basic Chat Functionality")
    logger.info("=" * 50)
    
    client = RedditChatbotClient()
    
    try:
        # Test queries
        queries = [
            "What are people discussing about Python programming?",
            "Tell me about machine learning trends",
            "What's the community sentiment about data science?",
        ]
        
        for query in queries:
            logger.info(f"\n🔍 Query: {query}")
            response = await client.chat(query)
            print_chat_response(response)
            logger.info("\n" + "-" * 50)
    
    finally:
        await client.close()


async def demo_subreddit_filtering():
    """Demonstrate subreddit filtering."""
    logger.info("🎯 Demo 2: Subreddit Filtering")
    logger.info("=" * 50)
    
    client = RedditChatbotClient()
    
    try:
        query = "What are the latest trends?"
        
        # Without filtering
        logger.info(f"\n🔍 Query (all subreddits): {query}")
        response = await client.chat(query)
        print_chat_response(response)
        
        logger.info("\n" + "-" * 30)
        
        # With filtering
        subreddits = ["Python", "MachineLearning"]
        logger.info(f"\n🎯 Query (filtered by {subreddits}): {query}")
        response = await client.chat(query, subreddits=subreddits)
        print_chat_response(response)
    
    finally:
        await client.close()


async def demo_suggestions_and_topics():
    """Demonstrate suggestions and topics endpoints."""
    logger.info("💡 Demo 3: Suggestions and Topics")
    logger.info("=" * 50)
    
    client = RedditChatbotClient()
    
    try:
        # Get suggestions
        logger.info("\n💡 Query Suggestions:")
        suggestions = await client.get_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"   {i}. {suggestion}")
        
        logger.info("\n📊 Available Topics:")
        topics = await client.get_topics()
        for i, topic in enumerate(topics[:5], 1):  # Show top 5
            logger.info(f"   {i}. {topic['name']} ({topic['document_count']} docs)")
            logger.info(f"      Keywords: {', '.join(topic['keywords'])}")
    
    finally:
        await client.close()


async def demo_system_status():
    """Demonstrate system status check."""
    logger.info("🔧 Demo 4: System Status")
    logger.info("=" * 50)
    
    client = RedditChatbotClient()
    
    try:
        status = await client.get_status()
        
        logger.info(f"Status: {status['status']}")
        logger.info(f"Timestamp: {status['timestamp']}")
        
        if 'agents' in status:
            logger.info("\n🤖 Agent Status:")
            agents = status['agents']
            for agent_name, agent_status in agents.items():
                logger.info(f"   {agent_name}: {agent_status.get('status', 'unknown')}")
        
        if 'knowledge_base' in status:
            kb_stats = status['knowledge_base']
            logger.info(f"\n📚 Knowledge Base:")
            logger.info(f"   Documents: {kb_stats.get('total_documents', 0)}")
            logger.info(f"   Collections: {kb_stats.get('collections', 0)}")
        
        if 'ollama_configured' in status:
            ollama = status['ollama_configured']
            logger.info(f"\n🧠 Ollama Configuration:")
            logger.info(f"   Base URL: {ollama['base_url']}")
            logger.info(f"   Model: {ollama['model']}")
            logger.info(f"   Embedding Model: {ollama['embedding_model']}")
    
    finally:
        await client.close()


async def demo_error_handling():
    """Demonstrate error handling."""
    logger.info("⚠️  Demo 5: Error Handling")
    logger.info("=" * 50)
    
    client = RedditChatbotClient()
    
    try:
        # Test various error conditions
        test_cases = [
            ("Empty query", ""),
            ("Very long query", "a" * 1001),
        ]
        
        for description, query in test_cases:
            logger.info(f"\n🧪 Testing: {description}")
            try:
                response = await client.chat(query)
                logger.info(f"   Unexpected success: {response}")
            except httpx.HTTPStatusError as e:
                logger.info(f"   ✅ Expected error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                logger.info(f"   ❌ Unexpected error: {e}")
    
    finally:
        await client.close()


async def interactive_chat_demo():
    """Interactive chat demo (if running in terminal)."""
    logger.info("💬 Demo 6: Interactive Chat")
    logger.info("=" * 50)
    logger.info("Type your questions (or 'quit' to exit):")
    
    client = RedditChatbotClient()
    
    try:
        # Simulate interactive chat with pre-defined queries
        sample_queries = [
            "What are people saying about Python?",
            "Tell me about recent programming discussions",
            "What's the sentiment around machine learning?",
        ]
        
        for query in sample_queries:
            logger.info(f"\n👤 User: {query}")
            
            try:
                response = await client.chat(query)
                logger.info(f"🤖 Bot: {response['answer']}")
                
                if response['references']:
                    logger.info(f"📚 Found {len(response['references'])} relevant posts")
                
                if response['insights']:
                    topics = [insight['topic'] for insight in response['insights']]
                    logger.info(f"💡 Related topics: {', '.join(topics)}")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
            
            logger.info("-" * 30)
    
    finally:
        await client.close()


async def main():
    """Run all demos."""
    logger.info("🚀 Reddit Chatbot API Examples")
    logger.info("🔗 Make sure the API is running on http://localhost:8001")
    
    # Check if API is available
    try:
        client = RedditChatbotClient()
        await client.get_status()
        await client.close()
        logger.info("✅ API is available, starting demos...\n")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        logger.error("Please start the API first:")
        logger.error("   cd backend")
        logger.error("   python chatbot_api.py")
        return
    
    # Run demos
    demos = [
        ("Basic Chat", demo_basic_chat),
        ("Subreddit Filtering", demo_subreddit_filtering),
        ("Suggestions and Topics", demo_suggestions_and_topics),
        ("System Status", demo_system_status),
        ("Error Handling", demo_error_handling),
        ("Interactive Chat", interactive_chat_demo),
    ]
    
    for demo_name, demo_func in demos:
        try:
            await demo_func()
            logger.info("\n" + "=" * 60 + "\n")
        except Exception as e:
            logger.error(f"❌ Demo '{demo_name}' failed: {e}")
    
    logger.info("🎉 All demos completed!")
    logger.info("\n📝 Next steps:")
    logger.info("   1. Integrate this API with your React frontend")
    logger.info("   2. Use the /chat endpoint for user interactions")
    logger.info("   3. Display references and insights in your UI")
    logger.info("   4. Use /suggestions for query suggestions")


if __name__ == "__main__":
    asyncio.run(main())
