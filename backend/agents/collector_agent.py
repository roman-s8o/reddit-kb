"""
CollectorAgent - LangGraph-compatible Reddit data collection agent.
Fetches posts and comments from specified subreddits and saves to JSON.
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import asyncpraw
from loguru import logger


@dataclass
class RedditPost:
    """Data structure for Reddit posts."""
    id: str
    title: str
    selftext: str
    author: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    url: str
    permalink: str
    is_self: bool
    link_flair_text: Optional[str] = None
    comments: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.comments is None:
            self.comments = []


class CollectorAgent:
    """LangGraph-compatible agent for collecting Reddit data."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "rediit-kb-collector/1.0"
    ):
        """
        Initialize the CollectorAgent.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for Reddit API
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        
        # Ensure data directory exists
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize_reddit(self) -> None:
        """Initialize the Reddit API client."""
        if not self.reddit:
            try:
                self.reddit = asyncpraw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API client: {e}")
                raise
    
    async def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        max_posts: int = 100,
        max_comments_per_post: int = 50
    ) -> List[RedditPost]:
        """
        Fetch posts and comments from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            max_posts: Maximum number of posts to fetch
            max_comments_per_post: Maximum comments per post
        
        Returns:
            List of RedditPost objects
        """
        if not self.reddit:
            await self.initialize_reddit()
        
        posts = []
        
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            logger.info(f"Fetching {max_posts} posts from r/{subreddit_name}")
            
            # Fetch hot posts
            async for submission in subreddit.hot(limit=max_posts):
                try:
                    # Create post object
                    post = RedditPost(
                        id=submission.id,
                        title=submission.title,
                        selftext=submission.selftext or "",
                        author=str(submission.author) if submission.author else "[deleted]",
                        subreddit=subreddit_name,
                        score=submission.score,
                        upvote_ratio=submission.upvote_ratio,
                        num_comments=submission.num_comments,
                        created_utc=submission.created_utc,
                        url=submission.url,
                        permalink=submission.permalink,
                        is_self=submission.is_self,
                        link_flair_text=submission.link_flair_text
                    )
                    
                    # Fetch comments for this post
                    post.comments = await self._fetch_post_comments(
                        submission, max_comments_per_post
                    )
                    
                    posts.append(post)
                    logger.debug(f"Collected post: {submission.id}")
                    
                except Exception as e:
                    logger.warning(f"Error processing post {submission.id}: {e}")
                    continue
            
            logger.info(f"Successfully collected {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            return []
    
    async def _fetch_post_comments(
        self,
        submission,
        max_comments: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments from a Reddit post.
        
        Args:
            submission: Reddit submission object
            max_comments: Maximum number of comments to fetch
        
        Returns:
            List of comment dictionaries
        """
        comments = []
        comment_count = 0
        
        try:
            # Expand comment tree (limit to avoid rate limits)
            await submission.comments.replace_more(limit=0)
            
            # Collect top-level and nested comments
            for comment in submission.comments.list():
                if comment_count >= max_comments:
                    break
                
                try:
                    comment_data = {
                        "id": comment.id,
                        "body": comment.body,
                        "author": str(comment.author) if comment.author else "[deleted]",
                        "score": comment.score,
                        "created_utc": comment.created_utc,
                        "parent_id": comment.parent_id,
                        "post_id": submission.id,
                        "subreddit": str(submission.subreddit),
                        "permalink": comment.permalink,
                        "is_submitter": comment.is_submitter,
                        "depth": self._calculate_comment_depth(comment)
                    }
                    
                    comments.append(comment_data)
                    comment_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing comment {comment.id}: {e}")
                    continue
            
            logger.debug(f"Collected {len(comments)} comments for post {submission.id}")
            return comments
            
        except Exception as e:
            logger.warning(f"Error collecting comments for post {submission.id}: {e}")
            return []
    
    def _calculate_comment_depth(self, comment) -> int:
        """Calculate the depth of a comment in the thread."""
        depth = 0
        try:
            parent = comment.parent()
            while hasattr(parent, 'parent') and depth < 10:  # Prevent infinite loops
                depth += 1
                parent = parent.parent()
        except:
            pass
        return depth
    
    async def collect_from_subreddits(
        self,
        subreddit_names: List[str],
        max_posts: int = 100,
        max_comments_per_post: int = 50
    ) -> Dict[str, List[RedditPost]]:
        """
        Collect posts from multiple subreddits.
        
        Args:
            subreddit_names: List of subreddit names
            max_posts: Maximum posts per subreddit
            max_comments_per_post: Maximum comments per post
        
        Returns:
            Dictionary mapping subreddit names to posts
        """
        logger.info(f"Starting collection from {len(subreddit_names)} subreddits")
        
        all_posts = {}
        
        for subreddit_name in subreddit_names:
            try:
                posts = await self.fetch_subreddit_posts(
                    subreddit_name, max_posts, max_comments_per_post
                )
                all_posts[subreddit_name] = posts
                
                # Small delay between requests to be respectful
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to collect from r/{subreddit_name}: {e}")
                all_posts[subreddit_name] = []
        
        total_posts = sum(len(posts) for posts in all_posts.values())
        total_comments = sum(
            sum(len(post.comments) for post in posts)
            for posts in all_posts.values()
        )
        
        logger.info(f"Collection completed: {total_posts} posts, {total_comments} comments")
        return all_posts
    
    async def save_to_json(
        self,
        data: Dict[str, List[RedditPost]],
        filename: Optional[str] = None
    ) -> str:
        """
        Save collected data to JSON file.
        
        Args:
            data: Dictionary of posts by subreddit
            filename: Optional custom filename
        
        Returns:
            Path to the saved JSON file
        """
        if not filename:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"reddit_data_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert dataclasses to dictionaries
        serializable_data = {}
        for subreddit, posts in data.items():
            serializable_data[subreddit] = [asdict(post) for post in posts]
        
        # Create output structure
        output_data = {
            "metadata": {
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_subreddits": len(data),
                "total_posts": sum(len(posts) for posts in data.values()),
                "total_comments": sum(
                    sum(len(post.comments) for post in posts)
                    for posts in data.values()
                ),
                "subreddits": list(data.keys())
            },
            "data": serializable_data
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Data saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
            raise
    
    async def run(
        self,
        subreddit_names: List[str],
        max_posts: int = 100,
        max_comments_per_post: int = 50,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Main execution method for LangGraph integration.
        
        Args:
            subreddit_names: List of subreddit names to collect from
            max_posts: Maximum posts per subreddit (default: 100)
            max_comments_per_post: Maximum comments per post (default: 50)
            output_filename: Optional custom filename for output
        
        Returns:
            Path to the saved JSON file
        """
        try:
            logger.info(f"CollectorAgent starting collection from: {subreddit_names}")
            
            # Initialize Reddit client
            await self.initialize_reddit()
            
            # Collect data from all subreddits
            collected_data = await self.collect_from_subreddits(
                subreddit_names, max_posts, max_comments_per_post
            )
            
            # Save to JSON
            output_path = await self.save_to_json(collected_data, output_filename)
            
            logger.info(f"CollectorAgent completed successfully. Output: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"CollectorAgent failed: {e}")
            raise
        finally:
            # Clean up Reddit client
            if self.reddit:
                await self.reddit.close()
    
    async def close(self):
        """Clean up resources."""
        if self.reddit:
            await self.reddit.close()
            logger.info("Reddit API client closed")


# LangGraph node function
async def collector_node(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    LangGraph node function for the CollectorAgent.
    
    Args:
        state: Current workflow state
        config: Optional configuration
    
    Returns:
        Updated state with collection results
    """
    try:
        # Extract parameters from state
        subreddit_names = state.get("subreddit_names", [])
        max_posts = state.get("max_posts", 100)
        max_comments_per_post = state.get("max_comments_per_post", 50)
        
        # Get Reddit credentials from config or environment
        if config and "reddit_credentials" in config:
            creds = config["reddit_credentials"]
            client_id = creds["client_id"]
            client_secret = creds["client_secret"]
            user_agent = creds.get("user_agent", "rediit-kb-collector/1.0")
        else:
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT", "rediit-kb-collector/1.0")
        
        if not client_id or not client_secret:
            raise ValueError("Reddit API credentials not provided")
        
        # Create and run collector agent
        collector = CollectorAgent(client_id, client_secret, user_agent)
        
        output_path = await collector.run(
            subreddit_names=subreddit_names,
            max_posts=max_posts,
            max_comments_per_post=max_comments_per_post
        )
        
        # Update state with results
        state.update({
            "collection_output_path": output_path,
            "collection_status": "completed",
            "collection_timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.info(f"CollectorAgent node completed: {output_path}")
        return state
        
    except Exception as e:
        logger.error(f"CollectorAgent node failed: {e}")
        state.update({
            "collection_status": "failed",
            "collection_error": str(e),
            "collection_timestamp": datetime.now(timezone.utc).isoformat()
        })
        return state


# Example usage
async def main():
    """Example usage of the CollectorAgent."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize collector
    collector = CollectorAgent(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="rediit-kb-collector/1.0"
    )
    
    # Run collection
    try:
        output_path = await collector.run(
            subreddit_names=["Python", "MachineLearning"],
            max_posts=10,  # Small number for testing
            max_comments_per_post=20
        )
        print(f"Collection completed. Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Collection failed: {e}")
    
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
