"""
Example LangGraph workflow using the CollectorAgent.
"""
import asyncio
import os
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from loguru import logger

from agents.collector_agent import collector_node


class WorkflowState(TypedDict):
    """State structure for the collection workflow."""
    subreddit_names: list[str]
    max_posts: int
    max_comments_per_post: int
    collection_output_path: str
    collection_status: str
    collection_error: str
    collection_timestamp: str


def create_collector_workflow() -> StateGraph:
    """Create a simple LangGraph workflow with the CollectorAgent."""
    
    # Create workflow
    workflow = StateGraph(WorkflowState)
    
    # Add the collector node
    workflow.add_node("collector", collector_node)
    
    # Set entry point and end
    workflow.set_entry_point("collector")
    workflow.add_edge("collector", END)
    
    return workflow.compile()


async def run_collection_workflow():
    """Run the collection workflow example."""
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv("REDDIT_CLIENT_ID") or not os.getenv("REDDIT_CLIENT_SECRET"):
        logger.error("Reddit API credentials not found in environment variables")
        logger.info("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return
    
    # Create workflow
    workflow = create_collector_workflow()
    
    # Define initial state
    initial_state = WorkflowState(
        subreddit_names=["Python", "MachineLearning", "programming"],
        max_posts=100,
        max_comments_per_post=50,
        collection_output_path="",
        collection_status="",
        collection_error="",
        collection_timestamp=""
    )
    
    try:
        logger.info("Starting collection workflow...")
        
        # Run the workflow
        final_state = await workflow.ainvoke(initial_state)
        
        # Check results
        if final_state["collection_status"] == "completed":
            logger.info(f"✅ Collection completed successfully!")
            logger.info(f"📁 Output file: {final_state['collection_output_path']}")
            logger.info(f"🕒 Timestamp: {final_state['collection_timestamp']}")
            
            # Print some stats from the output file
            import json
            with open(final_state['collection_output_path'], 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                logger.info(f"📊 Stats: {metadata.get('total_posts', 0)} posts, "
                           f"{metadata.get('total_comments', 0)} comments from "
                           f"{metadata.get('total_subreddits', 0)} subreddits")
        else:
            logger.error(f"❌ Collection failed: {final_state.get('collection_error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_collection_workflow())
