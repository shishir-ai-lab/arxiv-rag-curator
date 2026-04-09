from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="arXiv RAG Curator API",
    description="Production RAG system for arXiv papers",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "arxiv-rag-curator"}

@app.get("/")
async def root():
    return {
        "message": "Welcome to arXiv RAG Curator API",
        "docs": "/docs",
        "progress": "Infrastructure Ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)