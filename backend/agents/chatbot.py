"""
ChatbotAgent - Retrieves relevant information from Chroma and generates responses using Ollama.
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import chromadb
import numpy as np
from loguru import logger
import httpx

from config import settings
from agents.preprocessor import OllamaEmbeddings


@dataclass
class RetrievalResult:
    """Data structure for retrieval results."""
    document: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_type: str  # 'post' or 'comment'


@dataclass
class ChatResponse:
    """Data structure for chat responses."""
    response: str
    sources: List[RetrievalResult]
    query: str
    timestamp: str
    context_used: str
    confidence: float


class ChatbotAgent:
    """Agent responsible for answering queries using retrieved Reddit data."""
    
    def __init__(self):
        """Initialize the chatbot agent."""
        self.chroma_client = None
        self.collection = None
        self.embeddings = OllamaEmbeddings()
        self.llm_client = httpx.AsyncClient(timeout=120.0)
        self.insights_db_path = settings.sqlite_db_path
        
        # Response templates
        self.system_prompt = """You are a helpful assistant that answers questions based on Reddit discussions. 
Use the provided context from Reddit posts and comments to give accurate, informative responses.

Guidelines:
1. Base your answers primarily on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific posts or comments when relevant
4. Maintain a conversational but informative tone
5. If asked about trends or insights, refer to the analysis data provided
6. Be objective and present different viewpoints when they exist in the data"""
    
    async def initialize(self):
        """Initialize Chroma database connection."""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory
            )
            self.collection = self.chroma_client.get_collection(
                name=settings.chroma_collection_name
            )
            logger.info("Connected to Chroma collection for chatbot")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma connection: {e}")
            raise
    
    async def retrieve_relevant_documents(
        self,
        query: str,
        k: int = 10,
        subreddits: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        min_score: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents from Chroma based on query.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            subreddits: Filter by specific subreddits
            document_types: Filter by document types ('post', 'comment')
            min_score: Minimum similarity score threshold
        
        Returns:
            List of RetrievalResult objects
        """
        if not self.collection:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)
            
            # Build where clause for filtering
            where_clause = {}
            if subreddits:
                where_clause["subreddit"] = {"$in": subreddits}
            if document_types:
                where_clause["type"] = {"$in": document_types}
            
            # Query Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                logger.warning("No relevant documents found for query")
                return []
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 - distance
                
                # Apply minimum score filter
                if min_score and similarity_score < min_score:
                    continue
                
                retrieval_result = RetrievalResult(
                    document=doc,
                    metadata=metadata,
                    similarity_score=similarity_score,
                    source_type=metadata.get("type", "unknown")
                )
                retrieval_results.append(retrieval_result)
            
            # Sort by similarity score
            retrieval_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Retrieved {len(retrieval_results)} relevant documents for query")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def get_relevant_insights(
        self,
        query: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Get relevant insights from the insights database."""
        try:
            conn = sqlite3.connect(self.insights_db_path)
            cursor = conn.cursor()
            
            # Get recent insights
            cursor.execute("""
                SELECT id, created_at, subreddits, total_documents, data
                FROM insights
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            insights = []
            for row in cursor.fetchall():
                insight_data = json.loads(row[4])
                
                # Check if query keywords match insight topics
                query_lower = query.lower()
                insight_keywords = []
                
                for cluster in insight_data.get("clusters", []):
                    insight_keywords.extend(cluster.get("keywords", []))
                
                # Simple keyword matching
                relevance_score = sum(
                    1 for keyword in insight_keywords
                    if keyword.lower() in query_lower
                )
                
                if relevance_score > 0 or limit <= 3:  # Include if relevant or if we need more data
                    insight_data["relevance_score"] = relevance_score
                    insights.append(insight_data)
            
            conn.close()
            
            # Sort by relevance
            insights.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return insights[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving insights: {e}")
            return []
    
    def format_context_for_llm(
        self,
        retrieval_results: List[RetrievalResult],
        insights: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> str:
        """Format retrieved documents and insights for LLM context."""
        context_parts = []
        current_length = 0
        
        # Add insights first (they're usually more valuable)
        if insights:
            context_parts.append("=== ANALYSIS INSIGHTS ===")
            current_length += len(context_parts[-1])
            
            for insight in insights:
                insight_text = f"\nAnalysis from {insight.get('created_at', 'recent')}:\n"
                
                # Add key insights
                key_insights = insight.get("key_insights", [])
                if key_insights:
                    insight_text += "Key Insights:\n"
                    for insight_item in key_insights[:3]:
                        insight_text += f"- {insight_item}\n"
                
                # Add top topics
                clusters = insight.get("clusters", [])
                if clusters:
                    insight_text += "\nTop Discussion Topics:\n"
                    for cluster in clusters[:3]:
                        insight_text += f"- {cluster.get('name', 'Unknown')}: {cluster.get('document_count', 0)} discussions\n"
                
                # Add overall sentiment
                overall_sentiment = insight.get("overall_sentiment", {})
                if overall_sentiment:
                    dominant_sentiment = max(overall_sentiment.items(), key=lambda x: x[1])
                    insight_text += f"\nOverall Sentiment: {dominant_sentiment[0]} ({dominant_sentiment[1]:.1%})\n"
                
                if current_length + len(insight_text) < max_context_length * 0.3:  # Reserve 30% for insights
                    context_parts.append(insight_text)
                    current_length += len(insight_text)
        
        # Add retrieved documents
        if retrieval_results:
            context_parts.append("\n=== RELEVANT REDDIT DISCUSSIONS ===")
            current_length += len(context_parts[-1])
            
            for i, result in enumerate(retrieval_results):
                if current_length >= max_context_length * 0.9:  # Use 90% of max length
                    break
                
                # Format document with metadata
                doc_text = f"\n[Source {i+1}] "
                
                # Add source information
                metadata = result.metadata
                if metadata.get("type") == "post":
                    doc_text += f"Post from r/{metadata.get('subreddit', 'unknown')}"
                    if metadata.get("title"):
                        doc_text += f" - '{metadata['title']}'"
                else:
                    doc_text += f"Comment from r/{metadata.get('subreddit', 'unknown')}"
                
                doc_text += f" (Score: {metadata.get('score', 'N/A')}, Similarity: {result.similarity_score:.2f})\n"
                doc_text += f"Content: {result.document[:500]}{'...' if len(result.document) > 500 else ''}\n"
                
                if current_length + len(doc_text) < max_context_length:
                    context_parts.append(doc_text)
                    current_length += len(doc_text)
        
        return "".join(context_parts)
    
    async def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 500
    ) -> Tuple[str, float]:
        """
        Generate response using Ollama LLM.
        
        Returns:
            Tuple of (response_text, confidence_score)
        """
        try:
            prompt = f"""{self.system_prompt}

Context Information:
{context}

User Question: {query}

Based on the context provided above, please provide a helpful and accurate response. If the context doesn't contain enough information to fully answer the question, please mention this and provide what information you can."""

            response = await self.llm_client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            response_text = result.get("response", "").strip()
            
            # Simple confidence estimation based on response length and context usage
            confidence = min(1.0, len(response_text) / 200)  # Basic heuristic
            if "I don't have enough information" in response_text or "based on the context" in response_text.lower():
                confidence *= 0.8
            
            return response_text, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again later.", 0.0
    
    async def chat(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        k: int = 10,
        include_insights: bool = True
    ) -> ChatResponse:
        """
        Main chat function that handles the complete pipeline.
        
        Args:
            query: User query
            subreddits: Filter by specific subreddits
            k: Number of documents to retrieve
            include_insights: Whether to include analysis insights
        
        Returns:
            ChatResponse object
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize if needed
            if not self.collection:
                await self.initialize()
            
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing query: {query[:100]}...")
            retrieval_results = await self.retrieve_relevant_documents(
                query=query,
                k=k,
                subreddits=subreddits,
                min_score=0.3  # Filter out very low similarity results
            )
            
            # Step 2: Get relevant insights
            insights = []
            if include_insights:
                insights = await self.get_relevant_insights(query)
            
            # Step 3: Format context
            context = self.format_context_for_llm(retrieval_results, insights)
            
            # Step 4: Generate response
            response_text, confidence = await self.generate_response(query, context)
            
            # Create response object
            chat_response = ChatResponse(
                response=response_text,
                sources=retrieval_results,
                query=query,
                timestamp=start_time.isoformat(),
                context_used=context[:500] + "..." if len(context) > 500 else context,
                confidence=confidence
            )
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"Chat response generated in {duration:.2f}s with confidence {confidence:.2f}")
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return ChatResponse(
                response="I apologize, but I encountered an error while processing your question. Please try again later.",
                sources=[],
                query=query,
                timestamp=start_time.isoformat(),
                context_used="",
                confidence=0.0
            )
    
    async def get_topic_summary(
        self,
        topic: str,
        subreddits: Optional[List[str]] = None,
        max_documents: int = 20
    ) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a specific topic.
        
        Args:
            topic: Topic to summarize
            subreddits: Filter by specific subreddits
            max_documents: Maximum number of documents to analyze
        
        Returns:
            Dictionary with topic summary information
        """
        try:
            # Retrieve documents related to the topic
            retrieval_results = await self.retrieve_relevant_documents(
                query=topic,
                k=max_documents,
                subreddits=subreddits
            )
            
            if not retrieval_results:
                return {
                    "topic": topic,
                    "summary": "No relevant discussions found for this topic.",
                    "document_count": 0,
                    "subreddits": [],
                    "key_points": [],
                    "sentiment": "neutral"
                }
            
            # Analyze the retrieved documents
            documents = [result.document for result in retrieval_results]
            subreddit_counts = {}
            total_score = 0
            score_count = 0
            
            for result in retrieval_results:
                subreddit = result.metadata.get("subreddit", "unknown")
                subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
                
                if "score" in result.metadata:
                    try:
                        score = float(result.metadata["score"])
                        total_score += score
                        score_count += 1
                    except (ValueError, TypeError):
                        pass
            
            avg_score = total_score / score_count if score_count > 0 else 0
            
            # Generate summary using LLM
            context = "\n\n".join(documents[:10])  # Use top 10 documents
            summary_prompt = f"""
Analyze the following Reddit discussions about "{topic}" and provide a concise summary.

Content:
{context[:2000]}

Please provide:
1. A brief summary of the main points discussed
2. Key insights or common themes
3. Overall sentiment (positive/negative/neutral)

Keep the response concise and factual."""
            
            summary_response, _ = await self.generate_response(
                topic + " summary",
                summary_prompt,
                max_tokens=300
            )
            
            return {
                "topic": topic,
                "summary": summary_response,
                "document_count": len(retrieval_results),
                "subreddits": list(subreddit_counts.keys()),
                "subreddit_distribution": subreddit_counts,
                "average_score": avg_score,
                "sources": retrieval_results[:5]  # Top 5 sources
            }
            
        except Exception as e:
            logger.error(f"Error generating topic summary: {e}")
            return {
                "topic": topic,
                "summary": f"Error generating summary: {str(e)}",
                "document_count": 0,
                "subreddits": [],
                "key_points": [],
                "sentiment": "neutral"
            }
    
    async def suggest_related_topics(
        self,
        query: str,
        k: int = 5
    ) -> List[str]:
        """
        Suggest related topics based on the query.
        
        Args:
            query: Original query
            k: Number of suggestions to return
        
        Returns:
            List of related topic suggestions
        """
        try:
            # Get insights to find related topics
            insights = await self.get_relevant_insights(query, limit=5)
            
            related_topics = set()
            
            for insight in insights:
                clusters = insight.get("clusters", [])
                for cluster in clusters[:3]:  # Top 3 clusters per insight
                    cluster_name = cluster.get("name", "")
                    keywords = cluster.get("keywords", [])
                    
                    if cluster_name and cluster_name.lower() not in query.lower():
                        related_topics.add(cluster_name)
                    
                    for keyword in keywords[:2]:  # Top 2 keywords per cluster
                        if keyword.lower() not in query.lower() and len(keyword) > 3:
                            related_topics.add(keyword.title())
            
            # Convert to list and limit
            suggestions = list(related_topics)[:k]
            
            # If we don't have enough suggestions, add some generic ones
            if len(suggestions) < k:
                generic_suggestions = [
                    "Recent trends",
                    "Popular discussions",
                    "Community opinions",
                    "Technical discussions",
                    "User experiences"
                ]
                for suggestion in generic_suggestions:
                    if len(suggestions) >= k:
                        break
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating topic suggestions: {e}")
            return ["Popular topics", "Recent discussions", "Community insights"]
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get statistics about the chatbot's knowledge base."""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            # Get collection stats
            total_docs = self.collection.count()
            
            if total_docs > 0:
                # Sample some documents to get distribution info
                sample_size = min(100, total_docs)
                results = self.collection.get(limit=sample_size)
                
                subreddit_counts = {}
                type_counts = {}
                
                for metadata in results["metadatas"]:
                    subreddit = metadata.get("subreddit", "unknown")
                    doc_type = metadata.get("type", "unknown")
                    
                    subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                return {
                    "total_documents": total_docs,
                    "sample_size": sample_size,
                    "subreddit_distribution": subreddit_counts,
                    "document_type_distribution": type_counts,
                    "status": "ready"
                }
            else:
                return {
                    "total_documents": 0,
                    "status": "empty"
                }
                
        except Exception as e:
            logger.error(f"Error getting chat statistics: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close connections and clean up resources."""
        if self.embeddings:
            await self.embeddings.close()
        if self.llm_client:
            await self.llm_client.aclose()
        logger.info("Chatbot agent closed")
