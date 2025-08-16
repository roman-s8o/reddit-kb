"""
Test script for the CollectorAgent.
Run this to verify the agent works correctly.
"""
import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from agents.collector_agent import CollectorAgent
from loguru import logger


async def test_collector_agent():
    """Test the CollectorAgent functionality."""
    
    # Load environment variables
    load_dotenv()
    
    # Check credentials
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "rediit-kb-test/1.0")
    
    if not client_id or not client_secret:
        logger.error("❌ Reddit API credentials not found!")
        logger.info("Please set the following environment variables:")
        logger.info("- REDDIT_CLIENT_ID")
        logger.info("- REDDIT_CLIENT_SECRET")
        logger.info("- REDDIT_USER_AGENT (optional)")
        return False
    
    logger.info("🚀 Starting CollectorAgent test...")
    
    try:
        # Initialize the collector
        collector = CollectorAgent(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Test with small numbers for quick testing
        test_subreddits = ["Python", "programming"]
        max_posts = 5  # Small number for testing
        max_comments = 10  # Small number for testing
        
        logger.info(f"📡 Testing collection from: {test_subreddits}")
        logger.info(f"📋 Parameters: {max_posts} posts, {max_comments} comments per post")
        
        # Run the collection
        output_path = await collector.run(
            subreddit_names=test_subreddits,
            max_posts=max_posts,
            max_comments_per_post=max_comments
        )
        
        # Verify the output file
        if Path(output_path).exists():
            logger.info(f"✅ Collection completed successfully!")
            logger.info(f"📁 Output file: {output_path}")
            
            # Load and display stats
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            metadata = data.get('metadata', {})
            logger.info("📊 Collection Statistics:")
            logger.info(f"   - Total subreddits: {metadata.get('total_subreddits', 0)}")
            logger.info(f"   - Total posts: {metadata.get('total_posts', 0)}")
            logger.info(f"   - Total comments: {metadata.get('total_comments', 0)}")
            logger.info(f"   - Timestamp: {metadata.get('collection_timestamp', 'N/A')}")
            
            # Show sample data structure
            reddit_data = data.get('data', {})
            for subreddit, posts in reddit_data.items():
                logger.info(f"   - r/{subreddit}: {len(posts)} posts")
                if posts:
                    sample_post = posts[0]
                    logger.info(f"     Sample post: '{sample_post.get('title', 'N/A')[:50]}...'")
                    logger.info(f"     Comments: {len(sample_post.get('comments', []))}")
            
            return True
        else:
            logger.error(f"❌ Output file not found: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False
    
    finally:
        # Clean up
        if 'collector' in locals():
            await collector.close()


async def test_langgraph_integration():
    """Test the LangGraph node integration."""
    
    logger.info("🔗 Testing LangGraph node integration...")
    
    try:
        from agents.collector_agent import collector_node
        
        # Create test state
        test_state = {
            "subreddit_names": ["Python"],
            "max_posts": 2,
            "max_comments_per_post": 5
        }
        
        # Test the node function
        result_state = await collector_node(test_state)
        
        if result_state.get("collection_status") == "completed":
            logger.info("✅ LangGraph node integration test passed!")
            logger.info(f"📁 Output: {result_state.get('collection_output_path')}")
            return True
        else:
            logger.error(f"❌ LangGraph node test failed: {result_state.get('collection_error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ LangGraph integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🧪 Running CollectorAgent Tests")
    logger.info("=" * 50)
    
    # Test 1: Basic functionality
    logger.info("\n📋 Test 1: Basic CollectorAgent functionality")
    test1_passed = await test_collector_agent()
    
    # Test 2: LangGraph integration
    logger.info("\n📋 Test 2: LangGraph node integration")
    test2_passed = await test_langgraph_integration()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Summary:")
    logger.info(f"   Basic functionality: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"   LangGraph integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! CollectorAgent is ready to use.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
