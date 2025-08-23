"""
Centralized logging configuration for Reddit KB
"""
import sys
from loguru import logger
from pathlib import Path

def setup_logging():
    """Setup centralized logging configuration."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with consistent format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler (optional, but keeps logs)
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="100 MB",
        retention="7 days",
        compression="zip"
    )
    
    logger.info("Logging configured successfully")
    return logger

# Setup logging when module is imported
setup_logging()
