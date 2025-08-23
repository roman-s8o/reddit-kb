# 🚀 Quick Start Guide

## Prerequisites

1. **Python 3.8+** and **Node.js 18+**
2. **Ollama** installed and running
3. **Reddit API credentials**

## One-Minute Setup

### 1. Install Ollama Models
```bash
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

### 2. Set Up Reddit Credentials
```bash
python setup_reddit_credentials.py
```

### 3. Install Dependencies
```bash
# Backend
cd backend && pip install -r requirements.txt

# Frontend  
cd ../frontend && npm install
```

### 4. Launch System
```bash
python start_app.py
```

## Access Points

- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Quick Test

1. **Collect Data**:
   ```bash
   curl -X POST "http://localhost:8000/collect/sync" \
     -H "Content-Type: application/json" \
     -d '{"subreddits": ["MachineLearning"], "max_posts_per_subreddit": 10}'
   ```

2. **Ask a Question**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the latest trends in machine learning?", "subreddits": ["MachineLearning"]}'
   ```

## Features

✅ **Multi-Agent System**: Collector → Preprocessor → Insights → Chatbot  
✅ **FAISS Vector Database**: High-performance similarity search  
✅ **LangGraph Orchestration**: Complete workflow management  
✅ **React Frontend**: Modern UI with real-time monitoring  
✅ **FastAPI Backend**: Comprehensive REST API  
✅ **Production Ready**: One-command launch, error handling, monitoring  

## Troubleshooting

- **Ollama not running**: `ollama serve`
- **Port conflicts**: Check if ports 8000/3001 are free
- **Reddit API errors**: Verify credentials in `.env`
- **FAISS issues**: Reset with `rm -rf data/vector_db/`

## Next Steps

1. Explore the web interface at http://localhost:3001
2. Try different subreddits in the configuration
3. Ask questions in the chat interface
4. View insights and analytics in the dashboard

---

**Status**: ✅ **Production Ready**  
**Architecture**: LangGraph + FAISS + React + FastAPI
