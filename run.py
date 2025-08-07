# run.py

import uvicorn
import os

if __name__ == "__main__":
    # This makes the app compatible with hosting providers like Render.
    # It will use the PORT environment variable if it exists, otherwise it defaults to 8000.
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting HackRx 6.0 RAG Server on port {port}...")
    
    # Use the standard 'app.module:variable' format for Uvicorn
    uvicorn.run("app.main_api:app", host="0.0.0.0", port=port, reload=False)