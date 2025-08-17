"""
Test suite for InsightAgent functionality.

This module tests the InsightAgent's ability to:
- Load embeddings from Chroma
- Cluster posts with k-means (automatic cluster detection)
- Extract top keywords per cluster with TF-IDF
- Run sentiment analysis on posts
- Save results as "insight documents" into SQLite
"""
import asyncio
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from loguru import logger

# Test data for validation
SAMPLE_REDDIT_DATA = {
    "documents": [
        "Python is a great programming language for beginners. It has simple syntax and powerful libraries.",
        "Machine learning with Python is amazing. Libraries like scikit-learn make it easy to get started.",
        "I love coding in Python for data science. Pandas and NumPy are essential tools.",
        "JavaScript is becoming more popular for web development. React and Node.js are game changers.",
        "Web development with JavaScript frameworks is fun. Vue.js is my favorite for frontend.",
        "Frontend development has evolved a lot. Modern frameworks make it easier to build apps.",
        "I'm feeling frustrated with debugging today. This bug is driving me crazy!",
        "Great news! My project got approved and I'm excited to start working on it.",
        "Just had an amazing experience with the new Python framework. Highly recommend it!",
        "Disappointed with the latest update. It broke several features that were working fine."
    ],
    "metadatas": [
        {"subreddit": "Python", "score": 15.5, "post_type": "post"},
        {"subreddit": "MachineLearning", "score": 22.3, "post_type": "post"},
        {"subreddit": "Python", "score": 18.7, "post_type": "comment"},
        {"subreddit": "webdev", "score": 12.1, "post_type": "post"},
        {"subreddit": "javascript", "score": 16.9, "post_type": "post"},
        {"subreddit": "webdev", "score": 14.2, "post_type": "comment"},
        {"subreddit": "Python", "score": 5.8, "post_type": "comment"},
        {"subreddit": "MachineLearning", "score": 25.4, "post_type": "post"},
        {"subreddit": "Python", "score": 19.1, "post_type": "comment"},
        {"subreddit": "webdev", "score": 8.3, "post_type": "comment"}
    ]
}


async def setup_test_environment():
    """Set up test environment with sample data."""
    logger.info("🔧 Setting up test environment...")
    
    try:
        # Import after environment setup
        from agents.insight import InsightAgent
        from agents.preprocessor import PreprocessorAgent
        
        # Create temporary directory for test data
        test_dir = Path(tempfile.mkdtemp(prefix="reddit_kb_test_"))
        logger.info(f"📁 Created test directory: {test_dir}")
        
        # Set up environment variables for testing
        os.environ["CHROMA_PERSIST_DIRECTORY"] = str(test_dir / "chroma_test")
        os.environ["SQLITE_DB_PATH"] = str(test_dir / "test_insights.db")
        
        # Initialize preprocessor to create test embeddings
        preprocessor = PreprocessorAgent()
        await preprocessor.initialize()
        
        # Create some test embeddings (simulate real embeddings)
        embeddings = []
        for doc in SAMPLE_REDDIT_DATA["documents"]:
            # Create mock embeddings (in reality these would come from Ollama)
            # Use simple word-based features for testing
            words = doc.lower().split()
            embedding = np.random.rand(384)  # Standard embedding dimension
            # Add some pattern based on content for realistic clustering
            if "python" in doc.lower():
                embedding[0:50] += 0.5  # Python cluster
            elif "javascript" in doc.lower() or "web" in doc.lower():
                embedding[50:100] += 0.5  # Web dev cluster
            elif "frustrat" in doc.lower() or "disappoint" in doc.lower():
                embedding[100:150] += 0.5  # Negative sentiment cluster
            elif "great" in doc.lower() or "amazing" in doc.lower() or "excited" in doc.lower():
                embedding[150:200] += 0.5  # Positive sentiment cluster
            
            embeddings.append(embedding.tolist())
        
        # Store test data in Chroma
        collection = preprocessor.collection
        ids = [f"test_doc_{i}" for i in range(len(SAMPLE_REDDIT_DATA["documents"]))]
        
        collection.add(
            ids=ids,
            documents=SAMPLE_REDDIT_DATA["documents"],
            metadatas=SAMPLE_REDDIT_DATA["metadatas"],
            embeddings=embeddings
        )
        
        logger.info(f"✅ Test environment ready with {len(SAMPLE_REDDIT_DATA['documents'])} test documents")
        
        await preprocessor.close()
        return test_dir
        
    except Exception as e:
        logger.error(f"❌ Failed to set up test environment: {e}")
        raise


async def test_insight_agent_initialization():
    """Test InsightAgent initialization and database setup."""
    logger.info("🧪 Testing InsightAgent initialization...")
    
    try:
        from agents.insight import InsightAgent
        
        # Initialize the agent
        insight_agent = InsightAgent()
        await insight_agent.initialize()
        
        # Test database connection
        assert insight_agent.db_path.exists(), "SQLite database file not created"
        
        # Test Chroma connection
        assert insight_agent.collection is not None, "Chroma collection not initialized"
        
        # Test database schema
        conn = sqlite3.connect(insight_agent.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "insights" in tables, "Insights table not created"
        assert "clusters" in tables, "Clusters table not created"
        
        conn.close()
        await insight_agent.close()
        
        logger.info("✅ InsightAgent initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ InsightAgent initialization test failed: {e}")
        return False


async def test_embeddings_loading():
    """Test loading embeddings from Chroma."""
    logger.info("🧪 Testing embeddings loading from Chroma...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        await insight_agent.initialize()
        
        # Load all embeddings
        embeddings, metadatas, documents = await insight_agent.get_embeddings_data()
        
        # Validate results
        assert len(embeddings) > 0, "No embeddings loaded"
        assert len(embeddings) == len(metadatas), "Embeddings and metadata count mismatch"
        assert len(embeddings) == len(documents), "Embeddings and documents count mismatch"
        assert embeddings.shape[1] > 0, "Invalid embedding dimensions"
        
        # Test subreddit filtering
        python_embeddings, python_metadatas, python_documents = await insight_agent.get_embeddings_data(
            subreddits=["Python"]
        )
        
        assert len(python_embeddings) > 0, "No Python embeddings found"
        assert len(python_embeddings) < len(embeddings), "Filtering not working"
        
        # Verify all filtered results are from Python subreddit
        for meta in python_metadatas:
            assert meta.get("subreddit") == "Python", f"Non-Python document found: {meta}"
        
        await insight_agent.close()
        
        logger.info(f"✅ Embeddings loading test passed - loaded {len(embeddings)} total, {len(python_embeddings)} Python documents")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embeddings loading test failed: {e}")
        return False


async def test_clustering_functionality():
    """Test k-means clustering with automatic cluster detection."""
    logger.info("🧪 Testing clustering functionality...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        await insight_agent.initialize()
        
        # Load embeddings
        embeddings, metadatas, documents = await insight_agent.get_embeddings_data()
        
        # Test automatic cluster detection (should be based on data size)
        cluster_labels = insight_agent.cluster_embeddings(embeddings, method="kmeans")
        
        assert len(cluster_labels) == len(embeddings), "Cluster labels count mismatch"
        assert len(set(cluster_labels)) > 1, "No clustering occurred (all same cluster)"
        assert len(set(cluster_labels)) <= len(embeddings), "More clusters than documents"
        
        # Test with specific number of clusters
        specific_clusters = 3
        cluster_labels_specific = insight_agent.cluster_embeddings(
            embeddings, method="kmeans", n_clusters=specific_clusters
        )
        
        unique_clusters = len(set(cluster_labels_specific))
        assert unique_clusters == specific_clusters, f"Expected {specific_clusters} clusters, got {unique_clusters}"
        
        # Test DBSCAN clustering
        dbscan_labels = insight_agent.cluster_embeddings(embeddings, method="dbscan")
        assert len(dbscan_labels) == len(embeddings), "DBSCAN cluster labels count mismatch"
        
        await insight_agent.close()
        
        logger.info(f"✅ Clustering test passed - K-means: {len(set(cluster_labels))} clusters, DBSCAN: {len(set(dbscan_labels))} clusters")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clustering test failed: {e}")
        return False


async def test_keyword_extraction():
    """Test TF-IDF keyword extraction."""
    logger.info("🧪 Testing TF-IDF keyword extraction...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        await insight_agent.initialize()
        
        # Load documents
        _, _, documents = await insight_agent.get_embeddings_data()
        
        # Extract keywords
        keywords = insight_agent.extract_keywords(documents)
        
        # Validate results
        assert len(keywords) > 0, "No keywords extracted"
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords), "Invalid keyword format"
        assert all(isinstance(kw[0], str) and isinstance(kw[1], (int, float)) for kw in keywords), "Invalid keyword types"
        
        # Check that keywords are sorted by score (descending)
        scores = [kw[1] for kw in keywords]
        assert scores == sorted(scores, reverse=True), "Keywords not sorted by score"
        
        # Test with different parameters
        limited_keywords = insight_agent.extract_keywords(documents, max_features=5)
        assert len(limited_keywords) <= 5, "Max features limit not respected"
        
        # Check for expected programming-related keywords
        keyword_names = [kw[0] for kw in keywords]
        programming_terms = ["python", "javascript", "programming", "development", "code", "language"]
        found_terms = [term for term in programming_terms if any(term in kw.lower() for kw in keyword_names)]
        assert len(found_terms) > 0, f"No programming terms found in keywords: {keyword_names[:10]}"
        
        await insight_agent.close()
        
        logger.info(f"✅ Keyword extraction test passed - extracted {len(keywords)} keywords, top 5: {keywords[:5]}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Keyword extraction test failed: {e}")
        return False


async def test_sentiment_analysis():
    """Test sentiment analysis using Ollama."""
    logger.info("🧪 Testing sentiment analysis...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        await insight_agent.initialize()
        
        # Test with a small subset of documents for speed
        test_texts = [
            "I love this new Python framework, it's amazing!",  # Positive
            "This is terrible, nothing works properly.",        # Negative
            "The weather is okay today, nothing special."       # Neutral
        ]
        
        # Run sentiment analysis
        sentiment_results = await insight_agent.analyze_sentiment(test_texts)
        
        # Validate results
        assert len(sentiment_results) == len(test_texts), "Sentiment results count mismatch"
        
        for i, result in enumerate(sentiment_results):
            assert hasattr(result, 'sentiment'), "Missing sentiment field"
            assert hasattr(result, 'confidence'), "Missing confidence field"
            assert hasattr(result, 'text_id'), "Missing text_id field"
            assert hasattr(result, 'scores'), "Missing scores field"
            
            assert result.sentiment in ['positive', 'negative', 'neutral'], f"Invalid sentiment: {result.sentiment}"
            assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
            assert result.text_id == f"text_{i}", f"Invalid text_id: {result.text_id}"
        
        # Check if sentiment detection makes sense (this might be unreliable with mock data)
        sentiments = [r.sentiment for r in sentiment_results]
        logger.info(f"Detected sentiments: {sentiments}")
        
        await insight_agent.close()
        
        logger.info(f"✅ Sentiment analysis test passed - analyzed {len(test_texts)} texts")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis test failed: {e}")
        logger.warning("Note: This test might fail if Ollama is not running or accessible")
        return False


async def test_topic_cluster_creation():
    """Test creation of topic clusters."""
    logger.info("🧪 Testing topic cluster creation...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        await insight_agent.initialize()
        
        # Load data
        embeddings, metadatas, documents = await insight_agent.get_embeddings_data()
        
        # Cluster embeddings
        cluster_labels = insight_agent.cluster_embeddings(embeddings, method="kmeans")
        
        # Create mock sentiment results
        sentiment_results = []
        for i in range(len(documents)):
            from agents.insight import SentimentAnalysis
            sentiment_results.append(SentimentAnalysis(
                text_id=f"text_{i}",
                sentiment="neutral",
                confidence=0.7,
                scores={"reasoning": "test"}
            ))
        
        # Create topic clusters
        clusters = insight_agent.create_topic_clusters(
            embeddings, metadatas, documents, cluster_labels, sentiment_results
        )
        
        # Validate results
        assert len(clusters) > 0, "No topic clusters created"
        
        for cluster in clusters:
            assert hasattr(cluster, 'cluster_id'), "Missing cluster_id"
            assert hasattr(cluster, 'name'), "Missing name"
            assert hasattr(cluster, 'description'), "Missing description"
            assert hasattr(cluster, 'keywords'), "Missing keywords"
            assert hasattr(cluster, 'document_count'), "Missing document_count"
            assert hasattr(cluster, 'avg_score'), "Missing avg_score"
            assert hasattr(cluster, 'representative_texts'), "Missing representative_texts"
            assert hasattr(cluster, 'subreddits'), "Missing subreddits"
            assert hasattr(cluster, 'sentiment_distribution'), "Missing sentiment_distribution"
            
            assert cluster.document_count > 0, "Empty cluster created"
            assert len(cluster.keywords) > 0, "No keywords in cluster"
            assert len(cluster.representative_texts) > 0, "No representative texts"
            assert len(cluster.subreddits) > 0, "No subreddits in cluster"
            
            # Check sentiment distribution sums to 1
            sentiment_sum = sum(cluster.sentiment_distribution.values())
            assert abs(sentiment_sum - 1.0) < 0.01, f"Sentiment distribution doesn't sum to 1: {sentiment_sum}"
        
        await insight_agent.close()
        
        logger.info(f"✅ Topic cluster creation test passed - created {len(clusters)} clusters")
        return True
        
    except Exception as e:
        logger.error(f"❌ Topic cluster creation test failed: {e}")
        return False


async def test_full_analysis_pipeline():
    """Test the complete insight analysis pipeline."""
    logger.info("🧪 Testing full analysis pipeline...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        
        # Run complete analysis
        insights = await insight_agent.run_analysis(
            subreddits=None,  # Analyze all data
            clustering_method="kmeans",
            n_clusters=None  # Auto-detect
        )
        
        # Validate InsightSummary structure
        assert hasattr(insights, 'id'), "Missing insights id"
        assert hasattr(insights, 'created_at'), "Missing created_at"
        assert hasattr(insights, 'subreddits'), "Missing subreddits"
        assert hasattr(insights, 'total_documents'), "Missing total_documents"
        assert hasattr(insights, 'clusters'), "Missing clusters"
        assert hasattr(insights, 'overall_sentiment'), "Missing overall_sentiment"
        assert hasattr(insights, 'top_keywords'), "Missing top_keywords"
        assert hasattr(insights, 'trending_topics'), "Missing trending_topics"
        assert hasattr(insights, 'key_insights'), "Missing key_insights"
        
        # Validate content
        assert insights.total_documents > 0, "No documents analyzed"
        assert len(insights.clusters) > 0, "No clusters generated"
        assert len(insights.top_keywords) > 0, "No keywords extracted"
        assert len(insights.key_insights) > 0, "No insights generated"
        
        # Check overall sentiment sums to 1
        sentiment_sum = sum(insights.overall_sentiment.values())
        assert abs(sentiment_sum - 1.0) < 0.01, f"Overall sentiment doesn't sum to 1: {sentiment_sum}"
        
        # Test with subreddit filtering
        python_insights = await insight_agent.run_analysis(
            subreddits=["Python"],
            clustering_method="kmeans"
        )
        
        assert python_insights.total_documents <= insights.total_documents, "Filtered analysis has more documents"
        assert "Python" in python_insights.subreddits, "Python not in filtered subreddits"
        
        await insight_agent.close()
        
        logger.info(f"✅ Full analysis pipeline test passed - analyzed {insights.total_documents} documents, "
                   f"generated {len(insights.clusters)} clusters, {len(insights.key_insights)} insights")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full analysis pipeline test failed: {e}")
        return False


async def test_database_storage():
    """Test SQLite storage of insight documents."""
    logger.info("🧪 Testing database storage...")
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        
        # Run analysis to generate insights
        insights = await insight_agent.run_analysis()
        
        # Verify insights were saved to database
        conn = sqlite3.connect(insight_agent.db_path)
        cursor = conn.cursor()
        
        # Check insights table
        cursor.execute("SELECT COUNT(*) FROM insights")
        insights_count = cursor.fetchone()[0]
        assert insights_count > 0, "No insights saved to database"
        
        # Check clusters table
        cursor.execute("SELECT COUNT(*) FROM clusters")
        clusters_count = cursor.fetchone()[0]
        assert clusters_count > 0, "No clusters saved to database"
        
        # Verify data integrity
        cursor.execute("SELECT id, data FROM insights ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        assert row is not None, "No insight records found"
        
        insight_id, data_json = row
        data = json.loads(data_json)
        
        assert data['id'] == insight_id, "Insight ID mismatch"
        assert 'clusters' in data, "Missing clusters in stored data"
        assert 'total_documents' in data, "Missing total_documents in stored data"
        
        # Test retrieval of latest insights
        latest_insights = insight_agent.get_latest_insights(limit=5)
        assert len(latest_insights) > 0, "No insights retrieved"
        assert latest_insights[0]['id'] == insights.id, "Latest insight ID mismatch"
        
        conn.close()
        await insight_agent.close()
        
        logger.info(f"✅ Database storage test passed - stored {insights_count} insights, {clusters_count} clusters")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database storage test failed: {e}")
        return False


async def test_langgraph_integration():
    """Test LangGraph workflow integration."""
    logger.info("🧪 Testing LangGraph workflow integration...")
    
    try:
        from workflow.reddit_workflow import RedditWorkflow
        from langgraph.graph.message import HumanMessage
        
        # Create workflow
        workflow = RedditWorkflow()
        
        # Test insight node directly
        test_state = {
            "messages": [HumanMessage(content="Test insight generation")],
            "collected_data": None,
            "processed_data": {"success": True},  # Mock processed data
            "insights_data": None,
            "chat_response": None,
            "error": None,
            "metadata": {},
            "subreddits": ["Python"],
            "user_query": None,
            "workflow_type": "batch"
        }
        
        # Run insight node
        result_state = await workflow._insight_node(test_state)
        
        # Validate results
        assert "insights_data" in result_state, "No insights_data in result state"
        assert result_state["insights_data"] is not None, "insights_data is None"
        
        insights_data = result_state["insights_data"]
        assert "clusters" in insights_data, "No clusters in insights_data"
        assert "total_documents" in insights_data, "No total_documents in insights_data"
        assert "key_insights" in insights_data, "No key_insights in insights_data"
        
        # Check that messages were added
        assert len(result_state["messages"]) > len(test_state["messages"]), "No messages added"
        
        await workflow.close_all_agents()
        
        logger.info("✅ LangGraph integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ LangGraph integration test failed: {e}")
        return False


async def cleanup_test_environment(test_dir: Path):
    """Clean up test environment."""
    logger.info("🧹 Cleaning up test environment...")
    
    try:
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        logger.info("✅ Test environment cleaned up")
    except Exception as e:
        logger.error(f"❌ Failed to clean up test environment: {e}")


async def main():
    """Run all InsightAgent tests."""
    logger.info("🧪 Running InsightAgent Tests")
    logger.info("=" * 60)
    
    # Setup test environment
    test_dir = await setup_test_environment()
    
    test_results = {}
    
    try:
        # Test 1: Initialization
        logger.info("\n📋 Test 1: InsightAgent initialization")
        test_results["initialization"] = await test_insight_agent_initialization()
        
        # Test 2: Embeddings loading
        logger.info("\n📋 Test 2: Embeddings loading from Chroma")
        test_results["embeddings_loading"] = await test_embeddings_loading()
        
        # Test 3: Clustering functionality
        logger.info("\n📋 Test 3: K-means clustering with auto detection")
        test_results["clustering"] = await test_clustering_functionality()
        
        # Test 4: Keyword extraction
        logger.info("\n📋 Test 4: TF-IDF keyword extraction")
        test_results["keyword_extraction"] = await test_keyword_extraction()
        
        # Test 5: Sentiment analysis (might fail if Ollama not available)
        logger.info("\n📋 Test 5: Sentiment analysis with Ollama")
        test_results["sentiment_analysis"] = await test_sentiment_analysis()
        
        # Test 6: Topic cluster creation
        logger.info("\n📋 Test 6: Topic cluster creation")
        test_results["topic_clusters"] = await test_topic_cluster_creation()
        
        # Test 7: Full analysis pipeline
        logger.info("\n📋 Test 7: Full analysis pipeline")
        test_results["full_pipeline"] = await test_full_analysis_pipeline()
        
        # Test 8: Database storage
        logger.info("\n📋 Test 8: SQLite database storage")
        test_results["database_storage"] = await test_database_storage()
        
        # Test 9: LangGraph integration
        logger.info("\n📋 Test 9: LangGraph workflow integration")
        test_results["langgraph_integration"] = await test_langgraph_integration()
        
    finally:
        # Clean up
        await cleanup_test_environment(test_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Summary:")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! InsightAgent is fully functional and ready to use.")
        logger.info("\n📝 The InsightAgent successfully:")
        logger.info("   ✅ Loads embeddings from ChromaDB")
        logger.info("   ✅ Performs k-means clustering with automatic cluster detection")
        logger.info("   ✅ Extracts keywords using TF-IDF")
        logger.info("   ✅ Analyzes sentiment using Ollama")
        logger.info("   ✅ Stores insight documents in SQLite")
        logger.info("   ✅ Integrates with LangGraph workflow")
    else:
        failed_tests = [name for name, result in test_results.items() if not result]
        logger.error(f"❌ {total_tests - passed_tests} tests failed: {', '.join(failed_tests)}")
        logger.error("Please check the errors above and ensure:")
        logger.error("   - Ollama is running with the correct model")
        logger.error("   - All dependencies are installed")
        logger.error("   - Environment variables are set correctly")


if __name__ == "__main__":
    asyncio.run(main())
