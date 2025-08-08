# Fixed run.py

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting HackRx 6.0 RAG Server on port {port}...")
    
    # Use the correct path - adjust based on your file structure
    # If main_api.py is in the root directory:
    #uvicorn.run("main_api:app", host="0.0.0.0", port=port, reload=False)
    
    # If main_api.py is in app/ directory, use:
    uvicorn.run("app.main_api:app", host="0.0.0.0", port=port, reload=False)