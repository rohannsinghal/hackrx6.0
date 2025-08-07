# run.py

import uvicorn
import sys
import os

# This line is crucial for running from the top-level directory
# It tells Python to look for modules inside the 'app' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from main_api import app

if __name__ == "__main__":
    print("🚀 Starting HackRx 6.0 RAG Server...")
    print("API Documentation available at http://127.0.0.1:8000/docs")
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True, app_dir="app")