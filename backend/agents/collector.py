"""
CollectorAgent - Fetches subreddit posts and comments using asyncpraw.
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import asyncpraw
from loguru import logger

from config import settings


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


@dataclass
class RedditComment:
    """Data structure for Reddit comments."""
    id: str
    body: str
    author: str
    score: int
    created_utc: float
    parent_id: str
    post_id: str
    subreddit: str
    permalink: str
    is_submitter: bool = False
    depth: int = 0


class CollectorAgent:
    """Agent responsible for collecting Reddit posts and comments."""
    
    def __init__(self):
        """Initialize the collector agent."""
        self.reddit = None
        self.data_dir = Path("./data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the Reddit API client."""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent
            )
            logger.info("Reddit API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API client: {e}")
            raise
    
    async def collect_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = None,
        time_filter: str = "day",
        sort_by: str = "hot"
    ) -> List[RedditPost]:
        """
        Collect posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to collect
            time_filter: Time filter for posts (day, week, month, year, all)
            sort_by: Sort method (hot, new, top, rising)
        
        Returns:
            List of RedditPost objects
        """
        if not self.reddit:
            await self.initialize()
        
        limit = limit or settings.max_posts_per_subreddit
        posts = []
        
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            logger.info(f"Collecting {limit} posts from r/{subreddit_name} ({sort_by})")
            
            # Get posts based on sort method
            if sort_by == "hot":
                submissions = subreddit.hot(limit=limit)
            elif sort_by == "new":
                submissions = subreddit.new(limit=limit)
            elif sort_by == "top":
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_by == "rising":
                submissions = subreddit.rising(limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)
            
            async for submission in submissions:
                try:
                    # Create RedditPost object
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
                    
                    # Collect comments for the post
                    post.comments = await self.collect_post_comments(submission)
                    
                    posts.append(post)
                    logger.debug(f"Collected post: {submission.id} - {submission.title[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"Error processing submission {submission.id}: {e}")
                    continue
            
            logger.info(f"Successfully collected {len(posts)} posts from r/{subreddit_name}")
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting posts from r/{subreddit_name}: {e}")
            return []
    
    async def collect_post_comments(
        self,
        submission,
        max_comments: int = 50,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Collect comments from a Reddit post.
        
        Args:
            submission: Reddit submission object
            max_comments: Maximum number of comments to collect
            max_depth: Maximum comment thread depth
        
        Returns:
            List of comment dictionaries
        """
        comments = []
        comment_count = 0
        
        try:
            # Expand comment tree
            await submission.comments.replace_more(limit=0)
            
            # Collect comments recursively
            for comment in submission.comments.list():
                if comment_count >= max_comments:
                    break
                
                try:
                    # Calculate comment depth
                    depth = 0
                    parent = comment.parent()
                    while hasattr(parent, 'parent') and depth < max_depth:
                        depth += 1
                        parent = parent.parent()
                    
                    if depth > max_depth:
                        continue
                    
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
                        "depth": depth
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
    
    async def collect_all_subreddits(
        self,
        subreddits: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, List[RedditPost]]:
        """
        Collect posts from multiple subreddits.
        
        Args:
            subreddits: List of subreddit names
            **kwargs: Additional arguments for collect_subreddit_posts
        
        Returns:
            Dictionary mapping subreddit names to lists of posts
        """
        subreddits = subreddits or settings.subreddit_list
        all_posts = {}
        
        logger.info(f"Starting collection from {len(subreddits)} subreddits")
        
        for subreddit_name in subreddits:
            try:
                posts = await self.collect_subreddit_posts(subreddit_name, **kwargs)
                all_posts[subreddit_name] = posts
                
                # Add small delay between subreddit requests
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to collect from r/{subreddit_name}: {e}")
                all_posts[subreddit_name] = []
        
        total_posts = sum(len(posts) for posts in all_posts.values())
        logger.info(f"Collection completed. Total posts: {total_posts}")
        
        return all_posts
    
    async def save_collected_data(
        self,
        data: Dict[str, List[RedditPost]],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Save collected data to JSON file.
        
        Args:
            data: Dictionary of collected posts by subreddit
            timestamp: Optional timestamp for filename
        
        Returns:
            Path to saved file
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        filename = f"reddit_data_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename
        
        # Convert dataclasses to dictionaries
        serializable_data = {}
        for subreddit, posts in data.items():
            serializable_data[subreddit] = [asdict(post) for post in posts]
        
        # Add metadata
        output_data = {
            "metadata": {
                "collection_timestamp": timestamp.isoformat(),
                "total_subreddits": len(data),
                "total_posts": sum(len(posts) for posts in data.values()),
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
            logger.error(f"Failed to save data: {e}")
            raise
    
    async def run_collection(
        self,
        subreddits: Optional[List[str]] = None,
        save_data: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete collection process.
        
        Args:
            subreddits: List of subreddit names
            save_data: Whether to save collected data to file
            **kwargs: Additional arguments for collection
        
        Returns:
            Dictionary with collection results and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize Reddit client
            await self.initialize()
            
            # Collect data from all subreddits
            collected_data = await self.collect_all_subreddits(subreddits, **kwargs)
            
            # Save data if requested
            saved_file = None
            if save_data:
                saved_file = await self.save_collected_data(collected_data, start_time)
            
            # Calculate statistics
            total_posts = sum(len(posts) for posts in collected_data.values())
            total_comments = sum(
                sum(len(post.comments) for post in posts)
                for posts in collected_data.values()
            )
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "metadata": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "total_subreddits": len(collected_data),
                    "total_posts": total_posts,
                    "total_comments": total_comments,
                    "subreddits": list(collected_data.keys()),
                    "saved_file": saved_file
                },
                "data": collected_data
            }
            
            logger.info(f"Collection completed successfully in {duration:.2f}s")
            logger.info(f"Collected {total_posts} posts and {total_comments} comments")
            
            return result
            
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "start_time": start_time.isoformat(),
                    "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            }
    
    async def close(self):
        """Close the Reddit API client."""
        if self.reddit:
            await self.reddit.close()
            logger.info("Reddit API client closed")
