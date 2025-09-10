import uuid
from pathlib import Path
from datetime import datetime

from sqlalchemy.orm import Session
from models.models import Document
from crud.document_crud import save_document_to_disk,extract_text,chunk_text
from workers.vector_db import store_chunks

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

def upload_document(file, db: Session) -> dict:

    save_file_path = save_document_to_disk(file)
    print(save_file_path)
    document_id = save_file_path[0]
    file_path = save_file_path[1]

    text = extract_text(file_path)

    chunks = chunk_text(text)

    store_chunks(chunks, document_id)



    document = Document(
        document_id=document_id,
        filename=file.filename,
        path=str(file_path),
        uploaded_at=datetime.utcnow()
    )
    
    # Save to database
    db.add(document)
    db.commit()
    db.refresh(document)  # refresh instance with DB state
    
    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "path": document.path,
        "uploaded_at": document.uploaded_at.isoformat()
    }





