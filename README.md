# Reddit Knowledge Base

A multi-agent system built with LangGraph that collects, processes, and analyzes Reddit discussions to provide intelligent insights and chatbot capabilities.

## 🎯 Current Status

### ✅ Implemented Components
- **CollectorAgent**: Fully implemented and tested
  - Fetches latest 100 posts and top 50 comments per subreddit
  - Uses asyncpraw for efficient Reddit API access
  - Saves results to JSON in `data/raw/`
  - LangGraph node integration complete

- **PreprocessorAgent**: Fully implemented and tested
  - Enhanced text cleaning (HTML, emojis, URLs, Reddit formatting)
  - 500-token chunking with smart boundary detection
  - Ollama embeddings via LangChain integration
  - ChromaDB storage with comprehensive metadata
  - LangGraph node integration complete

### 🚧 In Development
- **InsightAgent**: Analysis and clustering functionality
- **ChatbotAgent**: RAG-based question answering
- **Frontend**: React-based web interface
- **API**: FastAPI backend endpoints

## Architecture

The system consists of 4 main agents orchestrated by LangGraph:

1. **CollectorAgent** → Fetches subreddit posts/comments via asyncpraw ✅
2. **PreprocessorAgent** → Cleans, chunks, embeds text via Ollama + stores in ChromaDB ✅
3. **InsightAgent** → Clusters embeddings, extracts keywords, sentiment analysis 🚧
4. **ChatbotAgent** → Retrieves top-k results from Chroma and generates responses 🚧

### Workflow

- **Sequential Flow**: Collector → Preprocessor → Insight (batch processing)
- **On-Demand Flow**: Chatbot (interactive queries)

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, LangGraph
- **LLM & Embeddings**: Ollama (llama2:7b, nomic-embed-text)
- **Vector Database**: ChromaDB (persistent storage)
- **Traditional Database**: SQLite (for insights storage)
- **Reddit API**: asyncpraw (async Reddit API wrapper)
- **Text Processing**: tiktoken (token counting), emoji, BeautifulSoup4
- **Frontend**: React, Material-UI (planned)
- **Data Processing**: scikit-learn, pandas, numpy

## Prerequisites

1. **Ollama** installed and running with models:
   ```bash
   ollama pull llama2:7b
   ollama pull nomic-embed-text
   ```

2. **Reddit API Credentials**:
   - Create a Reddit app at https://www.reddit.com/prefs/apps
   - Get client_id and client_secret

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
   REDDIT_USER_AGENT=rediit-kb-bot/1.0
   
   # Ollama Configuration
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2:7b
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

### Testing Current Implementation

1. **Test CollectorAgent**:
   ```bash
   cd backend
   python test_collector.py
   ```

2. **Test PreprocessorAgent**:
   ```bash
   cd backend
   python test_preprocessor.py
   ```

### Running Individual Agents

1. **Collect Reddit Data**:
   ```python
   # Run CollectorAgent example
   cd backend
   python -c "
   import asyncio
   from agents.collector_agent import CollectorAgent
   import os
   
   async def main():
       collector = CollectorAgent(
           client_id=os.getenv('REDDIT_CLIENT_ID'),
           client_secret=os.getenv('REDDIT_CLIENT_SECRET')
       )
       output_path = await collector.run(['Python', 'MachineLearning'])
       print(f'Data saved to: {output_path}')
   
   asyncio.run(main())
   "
   ```

2. **Process Collected Data**:
   ```python
   # Run PreprocessorAgent example
   cd backend
   python -c "
   import asyncio
   from agents.preprocessor import PreprocessorAgent
   
   async def main():
       preprocessor = PreprocessorAgent()
       result = await preprocessor.process_from_json_file('data/raw/reddit_data_*.json')
       print(f'Processed: {result}')
   
   asyncio.run(main())
   "
   ```

### Full System (Coming Soon)

1. **Start Backend** (when API is implemented):
   ```bash
   cd backend
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start Frontend** (when implemented):
   ```bash
   cd frontend
   npm start
   ```

## Usage

### Web Interface

1. **Chat Tab**: Interactive chatbot interface
   - Ask questions about Reddit discussions
   - Filter by specific subreddits
   - View sources and confidence scores

2. **Insights Tab**: Analytics dashboard
   - View generated insights and topic clusters
   - Sentiment analysis charts
   - Topic distribution graphs
   - Run data collection workflows

3. **Configuration Tab**: System management
   - Monitor agent status
   - View database statistics
   - Execute quick actions

### API Endpoints

The backend exposes REST API endpoints:

- `GET /health` - Health check
- `GET /status` - System status
- `POST /chat` - Send chat message
- `POST /collect` - Start data collection
- `POST /insights/generate` - Generate insights
- `POST /workflow/batch` - Run full workflow
- `GET /insights/dashboard` - Get dashboard data

### Programmatic Usage

```python
from workflow.reddit_workflow import run_data_collection_and_analysis, ask_reddit_question

# Run batch workflow
result = await run_data_collection_and_analysis(
    subreddits=['MachineLearning', 'Python']
)

# Ask a question
response = await ask_reddit_question(
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
OLLAMA_MODEL=llama2:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Data Collection
SUBREDDITS=MachineLearning,Python,programming,datascience
MAX_POSTS_PER_SUBREDDIT=100
COLLECTION_INTERVAL_HOURS=24

# Database
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
SQLITE_DB_PATH=./data/insights.db
```

### Subreddit Configuration

Modify the `SUBREDDITS` environment variable to monitor different communities:

```bash
SUBREDDITS=MachineLearning,Python,programming,datascience,artificial,deeplearning
```

## Data Flow

1. **Collection**: CollectorAgent fetches posts and comments from Reddit
2. **Processing**: PreprocessorAgent cleans text, creates chunks, generates embeddings
3. **Storage**: Embeddings stored in Chroma, raw data in JSON files
4. **Analysis**: InsightAgent performs clustering, keyword extraction, sentiment analysis
5. **Insights**: Results stored in SQLite database
6. **Interaction**: ChatbotAgent retrieves relevant information for user queries

## Implementation Details

### CollectorAgent Features
- **Async Reddit API**: Uses asyncpraw for efficient data collection
- **Configurable Limits**: Default 100 posts, 50 comments per subreddit
- **Rich Metadata**: Post scores, timestamps, author info, permalinks
- **Error Handling**: Graceful handling of deleted/removed content
- **LangGraph Integration**: Complete node function with state management
- **JSON Output**: Structured data with metadata for downstream processing

### PreprocessorAgent Features
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
- **ChromaDB Storage**:
  - Persistent vector storage
  - Rich metadata (post_id, subreddit, author, created_utc)
  - Batch operations for large datasets

### Testing
Both agents include comprehensive test suites:
- **Unit Tests**: Individual component testing
- **Integration Tests**: LangGraph node testing
- **Error Handling**: Edge case validation
- **Performance Tests**: Token counting and chunking validation

## File Structure

```
reddit-kb/
├── backend/
│   ├── agents/
│   │   ├── collector_agent.py    # ✅ Reddit data collection
│   │   ├── collector.py          # ✅ Alternative implementation
│   │   ├── preprocessor.py       # ✅ Text processing & embeddings  
│   │   ├── insight.py            # 🚧 Analysis & insights
│   │   └── chatbot.py            # 🚧 Query processing
│   ├── workflow/
│   │   └── reddit_workflow.py    # 🚧 LangGraph orchestration
│   ├── api/
│   │   └── main.py              # 🚧 FastAPI application
│   ├── examples/
│   │   └── collector_workflow.py # ✅ Usage examples
│   ├── config.py                # ✅ Configuration management
│   ├── test_collector.py        # ✅ CollectorAgent tests
│   ├── test_preprocessor.py     # ✅ PreprocessorAgent tests
│   └── requirements.txt         # ✅ Python dependencies
├── frontend/                    # 🚧 React frontend (planned)
├── data/
│   ├── raw/                     # ✅ JSON files from CollectorAgent
│   └── chroma_db/              # ✅ ChromaDB vector storage
├── docker-compose.yml          # 🚧 Docker configuration
└── README.md                   # ✅ This file
```

**Legend**: ✅ Implemented | 🚧 In Development

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`

2. **Reddit API Authentication**:
   - Verify credentials in `.env` file
   - Check Reddit app configuration

3. **Database Issues**:
   - Ensure write permissions for `./data/` directory
   - Check disk space for Chroma database

4. **Frontend Connection Issues**:
   - Verify backend is running on port 8000
   - Check CORS configuration in FastAPI

### Performance Tips

1. **Reduce Data Volume**:
   - Lower `MAX_POSTS_PER_SUBREDDIT` for testing
   - Monitor fewer subreddits initially

2. **Optimize Embeddings**:
   - Use smaller embedding models for faster processing
   - Batch embedding generation

3. **Database Performance**:
   - Regular cleanup of old data
   - Monitor Chroma database size

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
- Vector storage with Chroma
- UI with React and Material-UI
