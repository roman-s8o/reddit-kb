#!/bin/bash

# Reddit Knowledge Base - Complete System Startup Script
# This script starts all components of the Reddit Knowledge Base system

set -e

echo "🚀 Starting Reddit Knowledge Base - Complete System"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
}

# Check if Node.js is available
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        exit 1
    fi
    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed"
        exit 1
    fi
    print_success "Node.js found: $(node --version)"
    print_success "npm found: $(npm --version)"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    cd backend
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in backend directory"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
    cd ..
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    cd frontend
    
    if [ ! -f "package.json" ]; then
        print_error "package.json not found in frontend directory"
        exit 1
    fi
    
    npm install
    print_success "Node.js dependencies installed"
    cd ..
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/chroma_db
    mkdir -p data/workflow_states
    mkdir -p data/checkpoints
    mkdir -p data/recovery
    mkdir -p logs
    print_success "Directories created"
}

# Create environment file if it doesn't exist
create_env_file() {
    if [ ! -f "backend/.env" ]; then
        print_status "Creating backend environment file..."
        cat > backend/.env << EOF
# Reddit API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=reddit-kb-bot/1.0

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
SQLITE_DB_PATH=./data/insights.db

# Application Configuration
SUBREDDITS=MachineLearning,Python,programming,datascience
MAX_POSTS_PER_SUBREDDIT=100
COLLECTION_INTERVAL_HOURS=24
INSIGHT_GENERATION_INTERVAL_HOURS=6

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
LOG_LEVEL=INFO
EOF
        print_warning "Created backend/.env with default values"
        print_warning "Please update Reddit API credentials and Ollama configuration"
    fi
    
    if [ ! -f "frontend/.env" ]; then
        print_status "Creating frontend environment file..."
        cat > frontend/.env << EOF
# API Endpoints
REACT_APP_MAIN_API_URL=http://localhost:8000
REACT_APP_CHATBOT_API_URL=http://localhost:8001
REACT_APP_ORCHESTRATION_API_URL=http://localhost:8002

# Development Settings
REACT_APP_ENV=development
REACT_APP_DEBUG=true
EOF
        print_success "Created frontend/.env"
    fi
}

# Check if Ollama is running
check_ollama() {
    print_status "Checking Ollama service..."
    if curl -s http://localhost:11434/api/version > /dev/null; then
        print_success "Ollama is running"
    else
        print_warning "Ollama is not running on localhost:11434"
        print_warning "Please start Ollama before running the system"
        print_warning "Visit: https://ollama.ai for installation instructions"
    fi
}

# Start backend services
start_backends() {
    print_status "Starting backend services..."
    cd backend
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start Main API (port 8000)
    print_status "Starting Main API on port 8000..."
    python api/main.py &
    MAIN_API_PID=$!
    echo $MAIN_API_PID > ../pids/main_api.pid
    
    # Wait a moment for the service to start
    sleep 3
    
    # Start Chatbot API (port 8001)
    print_status "Starting Chatbot API on port 8001..."
    python chatbot_api.py &
    CHATBOT_API_PID=$!
    echo $CHATBOT_API_PID > ../pids/chatbot_api.pid
    
    # Wait a moment for the service to start
    sleep 3
    
    # Start Orchestration API (port 8002)
    print_status "Starting Orchestration API on port 8002..."
    python api/orchestration_api.py &
    ORCHESTRATION_API_PID=$!
    echo $ORCHESTRATION_API_PID > ../pids/orchestration_api.pid
    
    # Wait for services to start
    sleep 5
    
    print_success "Backend services started"
    cd ..
}

# Start frontend
start_frontend() {
    print_status "Starting frontend on port 3000..."
    cd frontend
    
    # Start React development server
    npm start &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../pids/frontend.pid
    
    print_success "Frontend started"
    cd ..
}

# Test system health
test_system_health() {
    print_status "Testing system health..."
    sleep 10  # Wait for all services to fully start
    
    # Test Main API
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Main API is healthy (port 8000)"
    else
        print_error "Main API is not responding (port 8000)"
    fi
    
    # Test Chatbot API
    if curl -s http://localhost:8001/health > /dev/null; then
        print_success "Chatbot API is healthy (port 8001)"
    else
        print_error "Chatbot API is not responding (port 8001)"
    fi
    
    # Test Orchestration API
    if curl -s http://localhost:8002/health > /dev/null; then
        print_success "Orchestration API is healthy (port 8002)"
    else
        print_error "Orchestration API is not responding (port 8002)"
    fi
    
    # Test Frontend
    if curl -s http://localhost:3000 > /dev/null; then
        print_success "Frontend is healthy (port 3000)"
    else
        print_warning "Frontend may still be starting (port 3000)"
    fi
}

# Create PID directory
mkdir -p pids

# Main execution
main() {
    echo
    print_status "Phase 1: System Prerequisites"
    check_python
    check_node
    check_ollama
    
    echo
    print_status "Phase 2: Environment Setup"
    create_directories
    create_env_file
    
    echo
    print_status "Phase 3: Dependencies Installation"
    install_python_deps
    install_node_deps
    
    echo
    print_status "Phase 4: Starting Services"
    start_backends
    start_frontend
    
    echo
    print_status "Phase 5: Health Check"
    test_system_health
    
    echo
    echo "=================================================="
    print_success "🎉 Reddit Knowledge Base System Started!"
    echo "=================================================="
    echo
    echo "📱 Access Points:"
    echo "   Frontend:        http://localhost:3000"
    echo "   Main API:        http://localhost:8000/docs"
    echo "   Chatbot API:     http://localhost:8001/docs"
    echo "   Orchestration:   http://localhost:8002/docs"
    echo
    echo "🔧 Management:"
    echo "   Stop System:     ./stop_system.sh"
    echo "   View Logs:       tail -f logs/*.log"
    echo "   Test System:     cd frontend && node test_phase5.js"
    echo
    echo "📚 Documentation:"
    echo "   README.md"
    echo "   PHASE5_FRONTEND_STATUS.md"
    echo "   ROADMAP.md"
    echo
    print_warning "Note: Make sure to configure Reddit API credentials in backend/.env"
    print_warning "Note: Ensure Ollama is running with required models"
    echo
    echo "Press Ctrl+C to stop all services"
    
    # Wait for user interrupt
    trap 'echo; print_status "Shutting down..."; ./stop_system.sh; exit 0' INT
    
    # Keep script running
    while true; do
        sleep 1
    done
}

# Run main function
main "$@"
