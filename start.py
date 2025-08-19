#!/usr/bin/env python3
"""
Simple starter script for the recommendation system
"""
import os
import sys
import subprocess
import time

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_python_deps():
    """Check if Python dependencies are available"""
    try:
        import torch
        import numpy
        import pandas
        return True
    except ImportError:
        return False

def run_with_docker():
    """Run with Docker Compose"""
    print("🐳 Starting with Docker...")
    
    # Build and start services
    subprocess.run(['docker-compose', 'up', '--build', '-d'])
    
    print("✅ Services started!")
    print()
    print("🔗 Available endpoints:")
    print("  • API: http://localhost:8000")
    print("  • Docs: http://localhost:8000/docs")
    print("  • Health: http://localhost:8000/health")
    print("  • Metrics: http://localhost:8000/metrics")
    print("  • Prometheus: http://localhost:9090")
    print("  • Grafana: http://localhost:3000 (admin/admin123)")
    print()
    print("🛑 To stop: docker-compose down")

def run_simple_demo():
    """Run simple demo without dependencies"""
    print("🚀 Running simple demo...")
    subprocess.run([sys.executable, 'run_demo.py'])

def run_with_python():
    """Run with Python directly"""
    print("🐍 Starting with Python...")
    
    # Try to start the API server
    try:
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'src.serving.api:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

def main():
    print("🚀 Product Recommendation Engine Starter")
    print("=" * 50)
    
    # Check what's available
    has_docker = check_docker()
    has_python_deps = check_python_deps()
    
    print("🔍 System check:")
    print(f"  Docker: {'✅' if has_docker else '❌'}")
    print(f"  Python deps: {'✅' if has_python_deps else '❌'}")
    print()
    
    # Choose best option
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        if has_docker:
            mode = 'docker'
        elif has_python_deps:
            mode = 'python'
        else:
            mode = 'demo'
    
    if mode == 'docker' and has_docker:
        run_with_docker()
    elif mode == 'python' and has_python_deps:
        run_with_python()
    elif mode == 'demo':
        run_simple_demo()
    else:
        print("❌ Cannot start in requested mode")
        print()
        print("Available modes:")
        print("  python start.py demo    # Simple demo (no deps)")
        print("  python start.py python  # Python server (requires deps)")
        print("  python start.py docker  # Docker services (requires Docker)")

if __name__ == "__main__":
    main()