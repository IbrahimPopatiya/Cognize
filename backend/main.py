# backend/main.py
from fastapi import FastAPI
from config import settings
from models.models import Base
from db.databse import engine
from api.document_api import document_router


Base.metadata.create_all(bind=engine)

# Initialized an APP
app = FastAPI()

app.include_router(document_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app = "main:app",
        host="localhost",
        port = 8000,
        reload=True
    )