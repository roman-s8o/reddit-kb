#!/bin/bash

# Reddit Knowledge Base - System Shutdown Script
# This script stops all components of the Reddit Knowledge Base system

echo "🛑 Stopping Reddit Knowledge Base System"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to stop a service by PID file
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            print_status "Stopping $service_name (PID: $pid)..."
            kill "$pid"
            sleep 2
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                print_warning "Force stopping $service_name..."
                kill -9 "$pid"
            fi
            
            print_success "$service_name stopped"
        else
            print_warning "$service_name was not running"
        fi
        rm -f "$pid_file"
    else
        print_warning "No PID file found for $service_name"
    fi
}

# Stop all services
if [ -d "pids" ]; then
    stop_service "Main API" "pids/main_api.pid"
    stop_service "Chatbot API" "pids/chatbot_api.pid"
    stop_service "Orchestration API" "pids/orchestration_api.pid"
    stop_service "Frontend" "pids/frontend.pid"
else
    print_warning "PID directory not found, attempting to stop by port"
    
    # Try to stop by port
    print_status "Attempting to stop services by port..."
    
    # Stop processes on known ports
    for port in 8000 8001 8002 3000; do
        pid=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$pid" ]; then
            print_status "Stopping process on port $port (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
        fi
    done
fi

# Clean up any remaining Python processes
print_status "Cleaning up remaining processes..."
pkill -f "python.*api" 2>/dev/null || true
pkill -f "python.*chatbot_api" 2>/dev/null || true
pkill -f "python.*orchestration_api" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true

# Remove PID directory
rm -rf pids

print_success "🎉 All services stopped successfully"
echo
echo "To restart the system, run: ./start_system.sh"
