import os
import sys
import uuid
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant_client import models
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FilterSelector,
    PointIdsList,
    FieldCondition,
    MatchValue
)

from openai import OpenAI
from config import settings

# ------------------------------
# Initialize clients
# ------------------------------
qdrant = QdrantClient(url=settings.QDRANT_URL)
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Collection name
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_DIM = 1536  # text-embedding-3-small vector size

# ------------------------------
# Collection operations
# ------------------------------
def create_collection(dim: int = EMBEDDING_DIM):
    """Create Qdrant collection if it doesn't exist."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def clear_collection():
    """Delete all vectors from the collection."""
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector = FilterSelector(filter=Filter(must=[]))
    )

# ------------------------------
# Embedding operations
# ------------------------------
def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for a single text."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts at once (more efficient)."""
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

# ------------------------------
# Qdrant CRUD operations
# ------------------------------
def upsert_embedding(vector: List[float], metadata: dict, point_id: Optional[str] = None):
    """Insert or update a vector with metadata."""
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=point_id or str(uuid.uuid4()),
            vector=vector,
            payload=metadata
        )]
    )

def delete_by_id(point_id: str):
    """Delete a vector by its ID."""
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[point_id])
    )

def search_embeddings(query_vector: List[float], top_k: int = 5):
    """Search for similar embeddings."""
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return results

# ------------------------------
# Store document chunks
# ------------------------------
def store_chunks(chunks: List[dict], document_id: str):
    # Ensure collection exists
    create_collection()

    embeddings = generate_embeddings_batch(chunks)

    points = []
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": text
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "success", "stored_chunks": len(points)}






def search_in_document(document_id: str, query: str, top_k: int = 5):
    query_vector = generate_embedding(query)

    points = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            ]
        ),
        limit=top_k
    ).points  

    results = []
    for p in points:
        results.append({
            "id": str(p.id),
            "score": p.score,
            "chunk_id": p.payload.get("chunk_id"),
            "text": p.payload.get("text")
        })

    return results




def search_across_documents(query: str, top_k: int = 5):
    query_vector = generate_embedding(query)

    points = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points  

    results = []
    for p in points:
        results.append({
            "id": str(p.id),
            "score": p.score,
            "document_id": p.payload.get("document_id"),
            "chunk_id": p.payload.get("chunk_id"),
            "text": p.payload.get("text")
        })

    return results


