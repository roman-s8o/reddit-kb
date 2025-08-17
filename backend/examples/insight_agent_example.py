"""
Example script demonstrating InsightAgent functionality.

This script shows how to:
1. Run the InsightAgent to analyze Reddit data
2. Display the generated insights
3. Save results to database

Prerequisites:
- Run CollectorAgent and PreprocessorAgent first to have data in ChromaDB
- Ensure Ollama is running with the correct model
"""
import asyncio
import json
from datetime import datetime
from loguru import logger

async def run_insight_analysis_example():
    """Run a complete insight analysis example."""
    logger.info("🚀 Starting InsightAgent Example")
    logger.info("=" * 50)
    
    try:
        from agents.insight import InsightAgent
        
        # Initialize the InsightAgent
        logger.info("🔧 Initializing InsightAgent...")
        insight_agent = InsightAgent()
        
        # Run complete analysis
        logger.info("📊 Running insight analysis...")
        insights = await insight_agent.run_analysis(
            subreddits=None,  # Analyze all available data
            clustering_method="kmeans",
            n_clusters=None  # Auto-detect optimal number of clusters
        )
        
        # Display results
        logger.info("\n" + "=" * 50)
        logger.info("📈 ANALYSIS RESULTS")
        logger.info("=" * 50)
        
        logger.info(f"🆔 Insight ID: {insights.id}")
        logger.info(f"📅 Created: {insights.created_at}")
        logger.info(f"📂 Subreddits: {', '.join(insights.subreddits)}")
        logger.info(f"📄 Total Documents: {insights.total_documents}")
        logger.info(f"🎯 Number of Clusters: {len(insights.clusters)}")
        
        # Overall sentiment
        logger.info(f"\n😊 Overall Sentiment:")
        for sentiment, percentage in insights.overall_sentiment.items():
            logger.info(f"   {sentiment.capitalize()}: {percentage:.1%}")
        
        # Top keywords
        logger.info(f"\n🔤 Top Keywords:")
        for i, (keyword, score) in enumerate(insights.top_keywords[:10], 1):
            logger.info(f"   {i:2d}. {keyword} (score: {score:.3f})")
        
        # Topic clusters
        logger.info(f"\n🎯 Topic Clusters:")
        for i, cluster in enumerate(insights.clusters, 1):
            logger.info(f"\n   Cluster {i}: {cluster.name}")
            logger.info(f"      📄 Documents: {cluster.document_count}")
            logger.info(f"      ⭐ Avg Score: {cluster.avg_score:.1f}")
            logger.info(f"      📂 Subreddits: {', '.join(cluster.subreddits)}")
            logger.info(f"      🔤 Keywords: {', '.join(cluster.keywords[:5])}")
            
            # Sentiment distribution
            main_sentiment = max(cluster.sentiment_distribution.items(), key=lambda x: x[1])
            logger.info(f"      😊 Main Sentiment: {main_sentiment[0]} ({main_sentiment[1]:.1%})")
            
            # Representative text (first one, truncated)
            if cluster.representative_texts:
                text = cluster.representative_texts[0]
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"      💬 Sample: \"{preview}\"")
        
        # Key insights
        logger.info(f"\n💡 Key Insights:")
        for i, insight in enumerate(insights.key_insights, 1):
            logger.info(f"   {i}. {insight}")
        
        # Trending topics
        if insights.trending_topics:
            logger.info(f"\n📈 Trending Topics:")
            for i, topic in enumerate(insights.trending_topics, 1):
                logger.info(f"   {i}. {topic}")
        
        # Show database storage confirmation
        logger.info(f"\n💾 Results saved to database with ID: {insights.id}")
        
        # Clean up
        await insight_agent.close()
        
        logger.info("\n✅ InsightAgent example completed successfully!")
        return insights
        
    except Exception as e:
        logger.error(f"❌ InsightAgent example failed: {e}")
        raise


async def demonstrate_filtered_analysis():
    """Demonstrate analysis with subreddit filtering."""
    logger.info("\n🎯 Demonstrating Filtered Analysis")
    logger.info("=" * 50)
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        
        # Analyze only Python-related subreddits
        python_insights = await insight_agent.run_analysis(
            subreddits=["Python", "MachineLearning"],
            clustering_method="kmeans"
        )
        
        logger.info(f"🐍 Python/ML Analysis Results:")
        logger.info(f"   📄 Documents: {python_insights.total_documents}")
        logger.info(f"   🎯 Clusters: {len(python_insights.clusters)}")
        logger.info(f"   🔤 Top Keywords: {', '.join([kw[0] for kw in python_insights.top_keywords[:5]])}")
        
        await insight_agent.close()
        
        return python_insights
        
    except Exception as e:
        logger.error(f"❌ Filtered analysis failed: {e}")
        return None


async def show_database_insights():
    """Show insights stored in the database."""
    logger.info("\n💾 Database Insights History")
    logger.info("=" * 50)
    
    try:
        from agents.insight import InsightAgent
        
        insight_agent = InsightAgent()
        
        # Get latest insights from database
        latest_insights = insight_agent.get_latest_insights(limit=5)
        
        if not latest_insights:
            logger.info("📭 No insights found in database")
            return
        
        logger.info(f"📚 Found {len(latest_insights)} insights in database:")
        
        for i, insight_data in enumerate(latest_insights, 1):
            created_at = datetime.fromisoformat(insight_data['created_at'].replace('Z', '+00:00'))
            logger.info(f"\n   {i}. ID: {insight_data['id']}")
            logger.info(f"      📅 Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"      📂 Subreddits: {', '.join(insight_data['subreddits'])}")
            logger.info(f"      📄 Documents: {insight_data['total_documents']}")
            logger.info(f"      🎯 Clusters: {len(insight_data['clusters'])}")
        
        await insight_agent.close()
        
    except Exception as e:
        logger.error(f"❌ Database insights retrieval failed: {e}")


async def main():
    """Run all examples."""
    logger.info("🎯 InsightAgent Examples")
    logger.info("🔗 Make sure you have run CollectorAgent and PreprocessorAgent first!")
    logger.info("🤖 Make sure Ollama is running with the correct model!")
    
    try:
        # Example 1: Full analysis
        insights = await run_insight_analysis_example()
        
        # Example 2: Filtered analysis
        await demonstrate_filtered_analysis()
        
        # Example 3: Show database history
        await show_database_insights()
        
        logger.info("\n🎉 All examples completed successfully!")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Check the generated insights in your SQLite database")
        logger.info("   2. Use the ChatbotAgent to ask questions about the analyzed data")
        logger.info("   3. Run the full LangGraph workflow for automated processing")
        
    except Exception as e:
        logger.error(f"❌ Examples failed: {e}")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("   1. Ensure you have run data collection and preprocessing first")
        logger.info("   2. Check that Ollama is running: ollama serve")
        logger.info("   3. Verify your model is available: ollama list")
        logger.info("   4. Check your environment variables in .env file")


if __name__ == "__main__":
    asyncio.run(main())
