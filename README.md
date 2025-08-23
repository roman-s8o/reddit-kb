
# Reddit Knowledge Base

A multi-agent system built with LangGraph that collects, processes, and analyzes Reddit discussions to provide intelligent insights and chatbot capabilities.

## 🎯 Current Status

### ✅ Fully Implemented & Production Ready
- **CollectorAgent**: Complete Reddit data collection system
  - Fetches latest posts and comments from specified subreddits
  - Uses asyncpraw for efficient Reddit API access
  - Saves results to JSON in `data/raw/`
  - LangGraph node integration complete

- **PreprocessorAgent**: Advanced text processing and embedding system
  - Enhanced text cleaning (HTML, emojis, URLs, Reddit formatting)
  - 500-token chunking with smart boundary detection
  - Ollama embeddings via LangChain integration
  - **FAISS vector database** storage with comprehensive metadata
  - LangGraph node integration complete

- **InsightAgent**: Intelligent analysis and clustering system
  - K-means clustering with automatic cluster number detection
  - TF-IDF keyword extraction per cluster
  - Sentiment analysis using Ollama LLM
  - SQLite storage for insight documents
  - LangGraph node integration complete

- **ChatbotAgent**: RAG-based question answering system
  - Retrieves relevant documents from FAISS vector database
  - Generates contextual responses using Ollama LLM
  - Returns sources and confidence scores
  - LangGraph node integration complete

- **FastAPI Backend**: Complete REST API
  - Health monitoring and system status
  - Chat endpoint with subreddit filtering
  - Data collection and workflow management
  - Model information and configuration
  - CORS support for frontend integration

- **React Frontend**: Modern web interface
  - Interactive chat interface with markdown support
  - Insights dashboard with charts and analytics
  - Configuration panel for system management
  - Real-time system health monitoring
  - Subreddit selection and filtering

- **LangGraph Orchestration**: Complete workflow management
  - Batch mode: Collector → Preprocessor → Insights
  - Query mode: Chatbot with FAISS retrieval
  - Scheduled workflows with APScheduler
  - Workflow state persistence and recovery
  - Comprehensive monitoring and metrics

## Architecture

The system consists of 4 main agents orchestrated by LangGraph:

1. **CollectorAgent** → Fetches subreddit posts/comments via asyncpraw ✅
2. **PreprocessorAgent** → Cleans, chunks, embeds text via Ollama + stores in FAISS ✅
3. **InsightAgent** → Clusters embeddings, extracts keywords, sentiment analysis ✅
4. **ChatbotAgent** → Retrieves top-k results from FAISS and generates responses ✅

### Workflow

- **Sequential Flow**: Collector → Preprocessor → Insight (batch processing) ✅
- **On-Demand Flow**: Chatbot (interactive queries) ✅

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, LangGraph ✅
- **LLM & Embeddings**: Ollama (gemma3:latest, nomic-embed-text) ✅
- **Vector Database**: **FAISS** (high-performance vector search) ✅
- **Traditional Database**: SQLite (for insights storage) ✅
- **Reddit API**: asyncpraw (async Reddit API wrapper) ✅
- **Text Processing**: tiktoken (token counting), emoji, BeautifulSoup4 ✅
- **Frontend**: React, Material-UI, Recharts ✅
- **Data Processing**: scikit-learn, pandas, numpy ✅
- **Orchestration**: APScheduler, psutil ✅

## Prerequisites

1. **Ollama** installed and running with models:
   ```bash
   ollama pull gemma3:latest
   ollama pull nomic-embed-text
   ```

2. **Reddit API Credentials**:
   - Create a Reddit app at https://www.reddit.com/prefs/apps
   - Get client_id and client_secret

3. **Node.js** (for frontend):
   - Install Node.js 18+ for React frontend

## Installation

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment configuration:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` file with your Reddit API credentials:
   ```bash
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret
   REDDIT_USER_AGENT=reddit-kb-bot/1.0
   
   # Ollama Configuration
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gemma3:latest
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   
   # Subreddits to monitor
   SUBREDDITS=MachineLearning,Python,programming,datascience
   ```

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Quick Start

### One-Command Launch

Use the provided startup script to launch the entire system:

```bash
python start_app.py
```

This will:
- Start the FastAPI backend on port 8000
- Start the React frontend on port 3001
- Handle graceful shutdown with Ctrl+C

### Manual Launch

1. **Start Backend**:
   ```bash
   cd backend
   python -c "import uvicorn; from api.main import app; uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')"
   ```

2. **Start Frontend**:
   ```bash
   cd frontend
   npm start
   ```

3. **Access the Application**:
   - Frontend: http://localhost:3001
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Testing Individual Components

1. **Test Data Collection**:
   ```bash
   cd backend
   curl -X POST "http://localhost:8000/collect/sync" \
     -H "Content-Type: application/json" \
     -d '{"subreddits": ["MachineLearning"], "max_posts_per_subreddit": 10}'
   ```

2. **Test Chat**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the latest trends in machine learning?", "subreddits": ["MachineLearning"]}'
   ```

## Usage

### Web Interface

1. **Chat Tab**: Interactive chatbot interface ✅
   - Ask questions about Reddit discussions
   - Filter by specific subreddits
   - View sources and confidence scores
   - Real-time model information display

2. **Insights Tab**: Analytics dashboard ✅
   - View generated insights and topic clusters
   - Sentiment analysis charts
   - Topic distribution graphs
   - Run data collection workflows

3. **Configuration Tab**: System management ✅
   - Monitor agent status and health
   - View database statistics
   - Execute quick actions
   - System metrics and performance

### API Endpoints

The backend exposes comprehensive REST API endpoints:

- `GET /health` - Health check ✅
- `GET /status` - System status and statistics ✅
- `GET /model/info` - Current LLM model information ✅
- `POST /chat` - Send chat message with subreddit filtering ✅
- `POST /collect/sync` - Start synchronous data collection ✅
- `POST /collect/async` - Start asynchronous data collection ✅
- `POST /insights/generate` - Generate insights from stored data ✅
- `POST /workflow/batch` - Run full batch workflow ✅
- `GET /insights/dashboard` - Get dashboard data ✅
- `GET /orchestration/status` - Workflow orchestration status ✅

### Programmatic Usage

```python
from workflow.reddit_workflow import run_batch_workflow, run_chat_workflow

# Run batch workflow
result = await run_batch_workflow(
    subreddits=['MachineLearning', 'Python']
)

# Ask a question
response = await run_chat_workflow(
    "What are the latest trends in machine learning?",
    subreddits=['MachineLearning']
)
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Data Collection
SUBREDDITS=MachineLearning,Python,programming,datascience
MAX_POSTS_PER_SUBREDDIT=100
COLLECTION_INTERVAL_HOURS=24

# Vector Database (FAISS)
CHROMA_PERSIST_DIRECTORY=./data/vector_db
CHROMA_COLLECTION_NAME=reddit_posts

# Traditional Database
SQLITE_DB_PATH=./data/insights.db
```

### Subreddit Configuration

Modify the `SUBREDDITS` environment variable to monitor different communities:

```bash
SUBREDDITS=MachineLearning,Python,programming,datascience,artificial,deeplearning
```

## Data Flow

1. **Collection**: CollectorAgent fetches posts and comments from Reddit ✅
2. **Processing**: PreprocessorAgent cleans text, creates chunks, generates embeddings ✅
3. **Storage**: Embeddings stored in **FAISS**, raw data in JSON files ✅
4. **Analysis**: InsightAgent performs clustering, keyword extraction, sentiment analysis ✅
5. **Insights**: Results stored in SQLite database ✅
6. **Interaction**: ChatbotAgent retrieves relevant information for user queries ✅

## Implementation Details

### CollectorAgent Features ✅
- **Async Reddit API**: Uses asyncpraw for efficient data collection
- **Configurable Limits**: Default 100 posts, 50 comments per subreddit
- **Rich Metadata**: Post scores, timestamps, author info, permalinks
- **Error Handling**: Graceful handling of deleted/removed content
- **LangGraph Integration**: Complete node function with state management
- **JSON Output**: Structured data with metadata for downstream processing

### PreprocessorAgent Features ✅
- **Advanced Text Cleaning**:
  - HTML tag and entity removal with BeautifulSoup4
  - Emoji removal using emoji library
  - URL detection and removal with multiple patterns
  - Reddit-specific formatting cleanup (markdown, mentions, etc.)
- **Token-based Chunking**:
  - Precise 500-token chunks using tiktoken
  - Smart boundary detection at sentence endings
  - 50-token overlap for context preservation
- **Ollama Integration**:
  - Async embedding generation via HTTP API
  - Batch processing for efficiency
  - Configurable models (default: nomic-embed-text)
- **FAISS Storage**:
  - High-performance vector storage
  - Rich metadata (post_id, subreddit, author, created_utc)
  - Batch operations for large datasets
  - Persistent storage with automatic indexing

### InsightAgent Features ✅
- **K-means Clustering**: Automatic cluster number detection
- **TF-IDF Analysis**: Keyword extraction per cluster
- **Sentiment Analysis**: Using Ollama LLM for emotional tone detection
- **SQLite Storage**: Structured insight documents
- **LangGraph Integration**: Complete workflow node

### ChatbotAgent Features ✅
- **FAISS Retrieval**: High-performance similarity search
- **RAG Pipeline**: Context-aware response generation
- **Source Attribution**: Returns relevant post IDs and metadata
- **Confidence Scoring**: Response quality assessment
- **Subreddit Filtering**: Query-specific data filtering

### Frontend Features ✅
- **React Interface**: Modern, responsive web application
- **Material-UI**: Professional design system
- **Real-time Updates**: Live system status and health monitoring
- **Interactive Charts**: Insights visualization with Recharts
- **Markdown Support**: Rich text rendering in chat
- **Multi-tab Interface**: Chat, Insights, and Configuration panels

## File Structure

```
reddit-kb/
├── backend/
│   ├── agents/
│   │   ├── collector_agent.py    # ✅ Reddit data collection
│   │   ├── collector.py          # ✅ Alternative implementation
│   │   ├── preprocessor.py       # ✅ Text processing & embeddings  
│   │   ├── insight.py            # ✅ Analysis & insights
│   │   └── chatbot.py            # ✅ Query processing
│   ├── workflow/
│   │   └── reddit_workflow.py    # ✅ LangGraph orchestration
│   ├── api/
│   │   └── main.py              # ✅ FastAPI application
│   ├── utils/
│   │   ├── faiss_manager.py     # ✅ FAISS vector database manager
│   │   └── chroma_manager.py    # ✅ Legacy ChromaDB manager
│   ├── examples/
│   │   └── collector_workflow.py # ✅ Usage examples
│   ├── config.py                # ✅ Configuration management
│   ├── test_collector.py        # ✅ CollectorAgent tests
│   ├── test_preprocessor.py     # ✅ PreprocessorAgent tests
│   └── requirements.txt         # ✅ Python dependencies
├── frontend/                    # ✅ React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.js  # ✅ Interactive chat
│   │   │   ├── InsightsDashboard.js # ✅ Analytics dashboard
│   │   │   └── ConfigurationPanel.js # ✅ System management
│   │   ├── services/
│   │   │   └── apiService.js     # ✅ API integration
│   │   └── App.js               # ✅ Main application
│   └── package.json             # ✅ Frontend dependencies
├── data/
│   ├── raw/                     # ✅ JSON files from CollectorAgent
│   ├── vector_db/              # ✅ FAISS vector storage
│   └── insights.db             # ✅ SQLite insights database
├── start_app.py                # ✅ One-command startup script
├── docker-compose.yml          # ✅ Docker configuration
└── README.md                   # ✅ This file
```

**Legend**: ✅ Fully Implemented & Production Ready

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`
   - Verify model names in `.env` file

2. **Reddit API Authentication**:
   - Verify credentials in `.env` file
   - Check Reddit app configuration
   - Use the provided `setup_reddit_credentials.py` script

3. **FAISS Database Issues**:
   - Ensure write permissions for `./data/vector_db/` directory
   - Check disk space for vector database
   - Reset database if needed: `rm -rf data/vector_db/`

4. **Frontend Connection Issues**:
   - Verify backend is running on port 8000
   - Check CORS configuration in FastAPI
   - Ensure frontend is running on port 3001

### Performance Tips

1. **Reduce Data Volume**:
   - Lower `MAX_POSTS_PER_SUBREDDIT` for testing
   - Monitor fewer subreddits initially

2. **Optimize Embeddings**:
   - Use smaller embedding models for faster processing
   - Batch embedding generation

3. **Database Performance**:
   - Regular cleanup of old data
   - Monitor FAISS database size
   - Use appropriate chunk sizes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Built with LangChain and LangGraph
- Uses Ollama for local LLM inference
- Reddit data via asyncpraw
- Vector storage with FAISS
- UI with React and Material-UI
- Orchestration with APScheduler
