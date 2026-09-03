"""Backend startup script for ED Triage Assist RAG API"""
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print("  ED Triage Assist — RAG Backend API")
    print("=" * 60)
    print("  API Docs: http://localhost:8000/docs")
    print("  Health:   http://localhost:8000/health")
    print("=" * 60)

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
