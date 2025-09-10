import uuid
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Adjust the path
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader
import docx2txt
from sqlalchemy.orm import Session
from backend.models.models import Document


UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def get_document(db: Session, document_id: str):
    """
    Fetch a single document's metadata from the database by document_id.
    """
    return db.query(Document).filter(Document.document_id == document_id).first()



def list_documents(db: Session):
    """
    Fetch all documents' metadata from the database.
    """
    return db.query(Document).all()


def save_document_to_disk(file):
    document_id = str(uuid.uuid4())
    save_file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
    with open(save_file_path, "wb") as f:
        f.write(file.file.read())

    return [document_id, str(save_file_path)]



def extract_text(file_path: str) -> str:
    """
    Extract raw text from PDF, DOCX, or TXT files.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".docx":
        text = docx2txt.process(path)
    elif ext == ".txt":
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return text.strip()




def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def get_dict_from_json_document(file_path: str) -> dict:
    import json
    path = Path(file_path)
    if path.suffix.lower() != ".json":
        raise ValueError("File must be a JSON document.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


