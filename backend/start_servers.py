#!/usr/bin/env python3
"""
Start both FastAPI and MCP servers for SL-IT-AI
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def start_mcp_server():
    """Start the MCP server in a subprocess"""
    print("Starting MCP server...")
    mcp_process = subprocess.Popen([
        sys.executable, "mcp_tools.py"
    ], cwd=Path(__file__).parent)
    return mcp_process

def start_fastapi_server():
    """Start the FastAPI server in a subprocess"""
    print("Starting FastAPI server...")
    fastapi_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app", 
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ], cwd=Path(__file__).parent)
    return fastapi_process

def main():
    print("Starting SL-IT-AI servers...")
    print("FastAPI will be available at: http://localhost:8000")
    print("MCP server will be available at: http://localhost:8001")
    print("Frontend should connect to: http://localhost:3000")
    print("\nPress Ctrl+C to stop all servers")
    
    try:
        # Start MCP server first
        mcp_process = start_mcp_server()
        time.sleep(3)  # Give MCP server time to start
        
        # Start FastAPI server
        fastapi_process = start_fastapi_server()
        
        # Wait for both processes
        mcp_process.wait()
        fastapi_process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping servers...")
        if 'mcp_process' in locals():
            mcp_process.terminate()
        if 'fastapi_process' in locals():
            fastapi_process.terminate()
        print("Servers stopped.")

if __name__ == "__main__":
    main() 