"""
Test script for the PreprocessorAgent.
Run this to verify the agent works correctly.
"""
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from agents.preprocessor import PreprocessorAgent, TextCleaner, TokenBasedTextSplitter, preprocessor_node
from loguru import logger


def test_text_cleaning():
    """Test the text cleaning functionality."""
    logger.info("🧹 Testing text cleaning functions...")
    
    # Test HTML removal
    html_text = "<p>This is a <strong>test</strong> with &amp; HTML entities &lt;script&gt;</p>"
    cleaned_html = TextCleaner.remove_html(html_text)
    logger.info(f"HTML removal: '{html_text}' -> '{cleaned_html}'")
    
    # Test emoji removal
    emoji_text = "This is a test with emojis 😀 🎉 🚀 and normal text"
    cleaned_emoji = TextCleaner.remove_emojis(emoji_text)
    logger.info(f"Emoji removal: '{emoji_text}' -> '{cleaned_emoji}'")
    
    # Test URL removal
    url_text = "Check out https://www.example.com and www.test.org for more info"
    cleaned_url = TextCleaner.remove_urls(url_text)
    logger.info(f"URL removal: '{url_text}' -> '{cleaned_url}'")
    
    # Test Reddit-specific cleaning
    reddit_text = "**Bold text** with /u/username and /r/subreddit mentions. EDIT: updated content"
    cleaned_reddit = TextCleaner.clean_reddit_text(reddit_text)
    logger.info(f"Reddit cleaning: '{reddit_text}' -> '{cleaned_reddit}'")
    
    # Test complete cleaning
    messy_text = "<p>**Bold** text with 😀 emojis, https://example.com URLs, and /u/user mentions &amp; HTML</p>"
    cleaned_complete = TextCleaner.clean_text(messy_text)
    logger.info(f"Complete cleaning: '{messy_text}' -> '{cleaned_complete}'")
    
    return True


def test_token_chunking():
    """Test the token-based text splitting."""
    logger.info("✂️ Testing token-based text chunking...")
    
    try:
        splitter = TokenBasedTextSplitter(chunk_size=50, chunk_overlap=10)
        
        # Test text
        long_text = """
        This is a long piece of text that should be split into multiple chunks based on token count.
        Each chunk should contain approximately 50 tokens with a 10-token overlap between consecutive chunks.
        The splitter should try to break at sentence boundaries when possible to maintain readability.
        This helps ensure that the chunks make sense when processed by language models.
        """
        
        # Test token counting
        token_count = splitter.count_tokens(long_text)
        logger.info(f"Token count for test text: {token_count}")
        
        # Test text splitting
        chunks = splitter.split_text(long_text)
        logger.info(f"Split into {len(chunks)} chunks:")
        
        for i, chunk in enumerate(chunks):
            chunk_tokens = splitter.count_tokens(chunk)
            logger.info(f"  Chunk {i+1}: {chunk_tokens} tokens - '{chunk[:100]}...'")
        
        return True
        
    except Exception as e:
        logger.error(f"Token chunking test failed: {e}")
        return False


async def test_preprocessor_agent():
    """Test the PreprocessorAgent functionality."""
    logger.info("🔧 Testing PreprocessorAgent...")
    
    # Create sample JSON data (simulating CollectorAgent output)
    sample_data = {
        "metadata": {
            "collection_timestamp": "2023-12-01T12:00:00Z",
            "total_subreddits": 1,
            "total_posts": 2,
            "total_comments": 3,
            "subreddits": ["test"]
        },
        "data": {
            "test": [
                {
                    "id": "post1",
                    "title": "Test Post 1 with **markdown** and https://example.com",
                    "selftext": "This is a test post with some content that includes HTML &amp; entities and 😀 emojis.",
                    "author": "test_user",
                    "subreddit": "test",
                    "score": 10,
                    "upvote_ratio": 0.9,
                    "num_comments": 2,
                    "created_utc": 1701432000,
                    "url": "https://reddit.com/r/test/post1",
                    "permalink": "/r/test/comments/post1",
                    "is_self": True,
                    "link_flair_text": "",
                    "comments": [
                        {
                            "id": "comment1",
                            "body": "This is a test comment with /u/username mentions and **bold** text.",
                            "author": "commenter1",
                            "score": 5,
                            "created_utc": 1701432100,
                            "parent_id": "post1",
                            "permalink": "/r/test/comments/post1/comment1",
                            "is_submitter": False,
                            "depth": 0
                        },
                        {
                            "id": "comment2",
                            "body": "Another comment with emojis 🎉 and URLs https://test.com for testing purposes.",
                            "author": "commenter2",
                            "score": 3,
                            "created_utc": 1701432200,
                            "parent_id": "post1",
                            "permalink": "/r/test/comments/post1/comment2",
                            "is_submitter": False,
                            "depth": 0
                        }
                    ]
                },
                {
                    "id": "post2",
                    "title": "Another Test Post",
                    "selftext": "More test content with different patterns and longer text to test chunking functionality.",
                    "author": "test_user2",
                    "subreddit": "test",
                    "score": 15,
                    "upvote_ratio": 0.85,
                    "num_comments": 1,
                    "created_utc": 1701432300,
                    "url": "https://reddit.com/r/test/post2",
                    "permalink": "/r/test/comments/post2",
                    "is_self": True,
                    "link_flair_text": "Discussion",
                    "comments": [
                        {
                            "id": "comment3",
                            "body": "A longer comment that might be split into multiple chunks when processed. This comment contains various elements like mentions, links, and formatting that should be cleaned properly.",
                            "author": "commenter3",
                            "score": 8,
                            "created_utc": 1701432400,
                            "parent_id": "post2",
                            "permalink": "/r/test/comments/post2/comment3",
                            "is_submitter": True,
                            "depth": 0
                        }
                    ]
                }
            ]
        }
    }
    
    # Save sample data to a temporary JSON file
    test_json_path = Path("data/raw/test_data.json")
    test_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created test JSON file: {test_json_path}")
    
    try:
        # Initialize preprocessor
        preprocessor = PreprocessorAgent()
        
        # Test document creation
        logger.info("📄 Testing document creation...")
        documents = preprocessor.create_documents_from_posts(sample_data["data"])
        logger.info(f"Created {len(documents)} documents")
        
        # Test chunking
        logger.info("✂️ Testing document chunking...")
        chunked_docs = preprocessor.chunk_documents(documents)
        logger.info(f"Split into {len(chunked_docs)} chunks")
        
        # Display chunk information
        for i, doc in enumerate(chunked_docs[:3]):  # Show first 3 chunks
            token_count = doc.metadata.get("token_count", "unknown")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"  Chunk {i+1}: {token_count} tokens - '{content_preview}'")
        
        logger.info("✅ PreprocessorAgent basic tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"PreprocessorAgent test failed: {e}")
        return False
    
    finally:
        # Clean up test file
        if test_json_path.exists():
            test_json_path.unlink()
            logger.info("Cleaned up test JSON file")


async def test_langgraph_integration():
    """Test the LangGraph node integration."""
    logger.info("🔗 Testing LangGraph node integration...")
    
    try:
        # Create test state
        test_state = {
            "collection_output_path": "data/raw/nonexistent_file.json"
        }
        
        # Test the node function with invalid path
        result_state = await preprocessor_node(test_state)
        
        if result_state.get("preprocessing_status") == "failed":
            logger.info("✅ LangGraph node correctly handled missing file")
            return True
        else:
            logger.error("❌ LangGraph node should have failed with missing file")
            return False
            
    except Exception as e:
        logger.error(f"LangGraph integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🧪 Running PreprocessorAgent Tests")
    logger.info("=" * 50)
    
    # Test 1: Text cleaning
    logger.info("\n📋 Test 1: Text cleaning functions")
    test1_passed = test_text_cleaning()
    
    # Test 2: Token chunking
    logger.info("\n📋 Test 2: Token-based text chunking")
    test2_passed = test_token_chunking()
    
    # Test 3: PreprocessorAgent functionality
    logger.info("\n📋 Test 3: PreprocessorAgent functionality")
    test3_passed = await test_preprocessor_agent()
    
    # Test 4: LangGraph integration
    logger.info("\n📋 Test 4: LangGraph node integration")
    test4_passed = await test_langgraph_integration()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Summary:")
    logger.info(f"   Text cleaning: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"   Token chunking: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    logger.info(f"   PreprocessorAgent: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    logger.info(f"   LangGraph integration: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    
    all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed])
    
    if all_passed:
        logger.info("🎉 All tests passed! PreprocessorAgent is ready to use.")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Make sure Ollama is running with the embedding model")
        logger.info("   2. Run the CollectorAgent to generate real data")
        logger.info("   3. Use PreprocessorAgent to process the collected data")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
