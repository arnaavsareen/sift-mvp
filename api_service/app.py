from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    """Health check endpoint. Real logic to be implemented on Day 3."""
    return {"status": "ok"}
