#!/usr/bin/env python3
"""
Simple startup script for the Reddit Knowledge Base application
"""
import subprocess
import time
import os
import signal
import sys

def cleanup_processes():
    """Clean up all running processes."""
    print("Cleaning up processes...")
    os.system("pkill -f 'python.*uvicorn' 2>/dev/null")
    os.system("pkill -f 'npm start' 2>/dev/null")
    os.system("pkill -f 'react-scripts' 2>/dev/null")
    print("Cleanup complete")

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\nReceived interrupt signal. Cleaning up...")
    cleanup_processes()
    sys.exit(0)

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting Reddit Knowledge Base")
    print("=" * 40)
    
    # Kill any existing processes
    print("Stopping existing processes...")
    cleanup_processes()
    
    # Clear vector database completely
    print("Clearing vector database...")
    os.system("rm -rf data/vector_db")
    os.system("mkdir -p data/vector_db")
    
    # Start only the main API first (this will handle vector database initialization)
    print("Starting Main API (Port 8000)...")
    # Start backend with logs redirected to file
    os.makedirs("logs", exist_ok=True)
    backend_log = open("logs/backend.log", "w")
    main_api = subprocess.Popen([
        "cd backend && python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info --no-use-colors --no-access-log"
    ], shell=True, stdout=backend_log, stderr=backend_log)
    
    print("Main API started (PID: {})".format(main_api.pid))
    
    # Wait for main API to initialize vector database
    print("Waiting for Main API to initialize...")
    time.sleep(10)
    
    # Start frontend
    print("Starting Frontend (Port 3001)...")
    frontend = subprocess.Popen([
        "cd frontend && PORT=3001 npm start"
    ], shell=True)
    
    print("Frontend started")
    
    print("Services started!")
    print("Service URLs:")
    print("Main API: http://localhost:8000")
    print("Frontend: http://localhost:3001")
    
    print("Open your browser and go to: http://localhost:3001")
    print("Next steps:")
    print("1. Wait for the frontend to load")
    print("2. The main API handles all backend functionality")
    print("3. Go to Configuration tab")
    print("4. Click 'Run Full Workflow' to collect Reddit data")
    
    print("Press Ctrl+C to stop all services")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
        
        # Terminate processes gracefully first
        main_api.terminate()
        frontend.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(3)
        
        # Force kill if still running
        if main_api.poll() is None:
            print("Force killing main API...")
            main_api.kill()
        
        if frontend.poll() is None:
            print("Force killing frontend...")
            frontend.kill()
        
        # Kill any remaining processes
        cleanup_processes()
        
        print("All services stopped")

if __name__ == "__main__":
    main()
    print("1. Wait for the frontend to load")
    print("2. The main API handles all backend functionality")
    print("3. Go to Configuration tab")
    print("4. Click 'Run Full Workflow' to collect Reddit data")
    
    print("Press Ctrl+C to stop all services")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
        
        # Terminate processes gracefully first
        main_api.terminate()
        frontend.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(3)
        
        # Force kill if still running
        if main_api.poll() is None:
            print("Force killing main API...")
            main_api.kill()
        
        if frontend.poll() is None:
            print("Force killing frontend...")
            frontend.kill()
        
        # Kill any remaining processes
        cleanup_processes()
        
        print("All services stopped")

if __name__ == "__main__":
    main()
