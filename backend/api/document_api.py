from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from config import settings
from sqlalchemy.orm import Session
from db.databse import SessionLocal
from workers.document_manager import upload_document
from db.databse import get_db
from workers.vector_db import search_in_document, search_across_documents


document_router = APIRouter()

@document_router.post("/upload")
def upload_document_api(file: UploadFile = File(...), db: Session = Depends(get_db)):
    result = upload_document(file, db)
    return result



@document_router.get("/{document_id}/search")
def search_document(document_id: str, query: str, top_k: int = 5):
    results = search_in_document(document_id, query, top_k)
    return {"results": results}

@document_router.get("/search")
def search_all(query: str, top_k: int = 5):
    """
    Search across all documents
    """
    results = search_across_documents(query, top_k)
    return {"results": results}
