"""
LangGraph workflow for orchestrating Reddit data collection, processing, and analysis.
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from loguru import logger

from agents.collector import CollectorAgent
from agents.preprocessor import PreprocessorAgent
from agents.insight import InsightAgent
from agents.chatbot import ChatbotAgent
from config import settings


class WorkflowState(TypedDict):
    """State structure for the workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    collected_data: Optional[Dict[str, Any]]
    processed_data: Optional[Dict[str, Any]]
    insights_data: Optional[Dict[str, Any]]
    chat_response: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]
    subreddits: Optional[List[str]]
    user_query: Optional[str]
    workflow_type: str  # 'batch' or 'chat'


class RedditWorkflow:
    """LangGraph workflow for Reddit data processing and chatbot functionality."""
    
    def __init__(self):
        """Initialize the workflow with all agents."""
        self.collector = CollectorAgent()
        self.preprocessor = PreprocessorAgent()
        self.insight_agent = InsightAgent()
        self.chatbot = ChatbotAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("collector", self._collector_node)
        workflow.add_node("preprocessor", self._preprocessor_node)
        workflow.add_node("insight_generator", self._insight_node)
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("router", self._router_node)
        
        # Define the workflow paths
        workflow.set_entry_point("router")
        
        # Router decides the workflow type
        workflow.add_conditional_edges(
            "router",
            self._route_workflow,
            {
                "batch_collection": "collector",
                "chat_query": "chatbot",
                "end": END
            }
        )
        
        # Batch processing flow: Collector → Preprocessor → Insight
        workflow.add_edge("collector", "preprocessor")
        workflow.add_edge("preprocessor", "insight_generator")
        workflow.add_edge("insight_generator", END)
        
        # Chat flow: Chatbot → END
        workflow.add_edge("chatbot", END)
        
        return workflow.compile()
    
    def _route_workflow(self, state: WorkflowState) -> str:
        """Route the workflow based on the workflow type."""
        workflow_type = state.get("workflow_type", "batch")
        
        if workflow_type == "chat":
            return "chat_query"
        elif workflow_type == "batch":
            return "batch_collection"
        else:
            logger.error(f"Unknown workflow type: {workflow_type}")
            return "end"
    
    async def _collector_node(self, state: WorkflowState) -> WorkflowState:
        """Collector agent node."""
        logger.info("Starting data collection...")
        
        try:
            subreddits = state.get("subreddits") or settings.subreddit_list
            
            # Run collection
            result = await self.collector.run_collection(
                subreddits=subreddits,
                save_data=True
            )
            
            if result["success"]:
                state["collected_data"] = result
                state["messages"].append(
                    AIMessage(content=f"Successfully collected data from {len(subreddits)} subreddits")
                )
                logger.info("Data collection completed successfully")
            else:
                error_msg = f"Data collection failed: {result.get('error', 'Unknown error')}"
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Collector node error: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            logger.error(error_msg)
        
        return state
    
    async def _preprocessor_node(self, state: WorkflowState) -> WorkflowState:
        """Preprocessor agent node."""
        logger.info("Starting data preprocessing...")
        
        try:
            collected_data = state.get("collected_data")
            if not collected_data or not collected_data.get("success"):
                error_msg = "No collected data available for preprocessing"
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                return state
            
            # Extract posts data from collection result
            posts_data = collected_data.get("data", {})
            
            # Run preprocessing
            result = await self.preprocessor.process_reddit_data(posts_data)
            
            if result["success"]:
                state["processed_data"] = result
                stats = result["statistics"]
                message = (f"Successfully processed {stats['total_posts']} posts and "
                          f"{stats['total_comments']} comments into "
                          f"{stats['stored_documents']} document chunks")
                state["messages"].append(AIMessage(content=message))
                logger.info("Data preprocessing completed successfully")
            else:
                error_msg = f"Data preprocessing failed: {result.get('error', 'Unknown error')}"
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Preprocessor node error: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            logger.error(error_msg)
        
        return state
    
    async def _insight_node(self, state: WorkflowState) -> WorkflowState:
        """Insight agent node."""
        logger.info("Starting insight generation...")
        
        try:
            processed_data = state.get("processed_data")
            if not processed_data or not processed_data.get("success"):
                error_msg = "No processed data available for insight generation"
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                return state
            
            # Run insight analysis
            subreddits = state.get("subreddits")
            insights = await self.insight_agent.run_analysis(
                subreddits=subreddits,
                clustering_method="kmeans"
            )
            
            # Convert insights to dictionary for storage
            insights_dict = asdict(insights)
            state["insights_data"] = insights_dict
            
            message = (f"Generated insights: {len(insights.clusters)} topic clusters, "
                      f"{len(insights.key_insights)} key insights from "
                      f"{insights.total_documents} documents")
            state["messages"].append(AIMessage(content=message))
            logger.info("Insight generation completed successfully")
            
        except Exception as e:
            error_msg = f"Insight node error: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            logger.error(error_msg)
        
        return state
    
    async def _chatbot_node(self, state: WorkflowState) -> WorkflowState:
        """Chatbot agent node."""
        logger.info("Processing chat query...")
        
        try:
            user_query = state.get("user_query")
            if not user_query:
                error_msg = "No user query provided for chat"
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                return state
            
            # Get subreddit filter if provided
            subreddits = state.get("subreddits")
            
            # Process chat query
            chat_response = await self.chatbot.chat(
                query=user_query,
                subreddits=subreddits,
                k=10,
                include_insights=True
            )
            
            # Convert to dictionary for storage
            chat_response_dict = {
                "response": chat_response.response,
                "query": chat_response.query,
                "timestamp": chat_response.timestamp,
                "confidence": chat_response.confidence,
                "sources": [
                    {
                        "document": source.document[:200] + "..." if len(source.document) > 200 else source.document,
                        "metadata": source.metadata,
                        "similarity_score": source.similarity_score,
                        "source_type": source.source_type
                    }
                    for source in chat_response.sources[:5]  # Top 5 sources
                ]
            }
            
            state["chat_response"] = chat_response_dict
            state["messages"].append(AIMessage(content=chat_response.response))
            
            logger.info(f"Chat query processed with confidence {chat_response.confidence:.2f}")
            
        except Exception as e:
            error_msg = f"Chatbot node error: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            logger.error(error_msg)
        
        return state
    
    async def _router_node(self, state: WorkflowState) -> WorkflowState:
        """Router node to initialize workflow metadata."""
        logger.info(f"Routing workflow type: {state.get('workflow_type', 'batch')}")
        
        # Initialize metadata
        if "metadata" not in state:
            state["metadata"] = {}
        
        state["metadata"]["workflow_start"] = datetime.now(timezone.utc).isoformat()
        state["metadata"]["workflow_type"] = state.get("workflow_type", "batch")
        
        return state
    
    async def run_batch_workflow(
        self,
        subreddits: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the batch workflow (Collector → Preprocessor → Insight).
        
        Args:
            subreddits: List of subreddits to process
            **kwargs: Additional workflow parameters
        
        Returns:
            Dictionary with workflow results
        """
        logger.info("Starting batch workflow...")
        
        initial_state = WorkflowState(
            messages=[HumanMessage(content="Starting batch data collection and analysis")],
            collected_data=None,
            processed_data=None,
            insights_data=None,
            chat_response=None,
            error=None,
            metadata={},
            subreddits=subreddits,
            user_query=None,
            workflow_type="batch"
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Extract results
            result = {
                "success": final_state.get("error") is None,
                "error": final_state.get("error"),
                "collected_data": final_state.get("collected_data"),
                "processed_data": final_state.get("processed_data"),
                "insights_data": final_state.get("insights_data"),
                "metadata": final_state.get("metadata", {}),
                "messages": [msg.content for msg in final_state.get("messages", [])]
            }
            
            result["metadata"]["workflow_end"] = datetime.now(timezone.utc).isoformat()
            
            if result["success"]:
                logger.info("Batch workflow completed successfully")
            else:
                logger.error(f"Batch workflow failed: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch workflow error: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "workflow_start": initial_state["metadata"].get("workflow_start"),
                    "workflow_end": datetime.now(timezone.utc).isoformat(),
                    "workflow_type": "batch"
                }
            }
    
    async def run_chat_workflow(
        self,
        user_query: str,
        subreddits: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the chat workflow (Chatbot only).
        
        Args:
            user_query: User's question
            subreddits: Optional subreddit filter
            **kwargs: Additional workflow parameters
        
        Returns:
            Dictionary with chat response
        """
        logger.info(f"Starting chat workflow for query: {user_query[:100]}...")
        
        initial_state = WorkflowState(
            messages=[HumanMessage(content=user_query)],
            collected_data=None,
            processed_data=None,
            insights_data=None,
            chat_response=None,
            error=None,
            metadata={},
            subreddits=subreddits,
            user_query=user_query,
            workflow_type="chat"
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Extract results
            result = {
                "success": final_state.get("error") is None,
                "error": final_state.get("error"),
                "chat_response": final_state.get("chat_response"),
                "metadata": final_state.get("metadata", {}),
                "messages": [msg.content for msg in final_state.get("messages", [])]
            }
            
            result["metadata"]["workflow_end"] = datetime.now(timezone.utc).isoformat()
            
            if result["success"]:
                logger.info("Chat workflow completed successfully")
            else:
                logger.error(f"Chat workflow failed: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Chat workflow error: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "workflow_start": initial_state["metadata"].get("workflow_start"),
                    "workflow_end": datetime.now(timezone.utc).isoformat(),
                    "workflow_type": "chat"
                }
            }
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get the current status of all agents."""
        try:
            status = {
                "collector": {
                    "status": "ready",
                    "reddit_client": self.collector.reddit is not None
                },
                "preprocessor": {
                    "status": "ready",
                    "chroma_connected": self.preprocessor.collection is not None,
                    "collection_stats": self.preprocessor.get_collection_stats()
                },
                "insight_agent": {
                    "status": "ready",
                    "database_connected": self.insight_agent.db_path.exists(),
                    "latest_insights": len(self.insight_agent.get_latest_insights(limit=5))
                },
                "chatbot": {
                    "status": "ready",
                    "knowledge_base": self.chatbot.get_chat_statistics()
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}
    
    async def initialize_all_agents(self):
        """Initialize all agents."""
        try:
            await self.collector.initialize()
            await self.preprocessor.initialize()
            await self.insight_agent.initialize()
            await self.chatbot.initialize()
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    async def close_all_agents(self):
        """Close all agent connections."""
        try:
            await self.collector.close()
            await self.preprocessor.close()
            await self.insight_agent.close()
            await self.chatbot.close()
            logger.info("All agents closed successfully")
        except Exception as e:
            logger.error(f"Error closing agents: {e}")


# Global workflow instance
reddit_workflow = RedditWorkflow()


# Utility functions for easier workflow execution
async def run_data_collection_and_analysis(
    subreddits: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the complete batch workflow.
    
    Args:
        subreddits: List of subreddits to process
    
    Returns:
        Workflow results
    """
    return await reddit_workflow.run_batch_workflow(subreddits=subreddits)


async def ask_reddit_question(
    question: str,
    subreddits: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to ask a question to the Reddit chatbot.
    
    Args:
        question: User's question
        subreddits: Optional subreddit filter
    
    Returns:
        Chat response
    """
    return await reddit_workflow.run_chat_workflow(
        user_query=question,
        subreddits=subreddits
    )


async def get_system_status() -> Dict[str, Any]:
    """Get the current system status."""
    return await reddit_workflow.get_workflow_status()


# Scheduled workflow functions
async def scheduled_data_refresh():
    """Function to be called by scheduler for regular data refresh."""
    logger.info("Starting scheduled data refresh...")
    
    try:
        result = await run_data_collection_and_analysis()
        
        if result["success"]:
            logger.info("Scheduled data refresh completed successfully")
        else:
            logger.error(f"Scheduled data refresh failed: {result['error']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Scheduled data refresh error: {e}")
        return {"success": False, "error": str(e)}
