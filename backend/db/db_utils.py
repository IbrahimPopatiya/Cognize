import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Adjust the path

from models.models import ChatMessage
from sqlalchemy.orm import Session

def get_chat_history(session_id: int, db: Session):
    messages = db.query(ChatMessage).filter_by(session_id=session_id).all()
    return [{"role": m.role, "content": m.content} for m in messages]

def save_message(session_id: int, role: str, content: str, db: Session):
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
