# --- Minimal Test App ---
from fastapi import FastAPI, Body

app = FastAPI(title="Minimal Test App")

@app.get("/")
def read_root():
    return {"status": "Minimal App is Running!"}

@app.post("/api/v1/hackrx/run")
def post_test(data: dict = Body(...)):
    # This just confirms we received the POST request and echoes back a count
    return {
        "status": "POST request successful!",
        "received_documents": len(data.get("documents", [])),
        "received_questions": len(data.get("questions", []))
    }