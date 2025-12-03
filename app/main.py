import io
import uuid
import os
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import httpx
from pypdf import PdfReader
import chardet

# ---- Configuración Básica ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"
COLLECTION_NAME = "docs"

app = FastAPI(title="Demo RAG Simple")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(url=QDRANT_URL)

@app.on_event("startup")
def startup_event():
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]
    if COLLECTION_NAME not in names:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Colección '{COLLECTION_NAME}' creada.")

# ---- Modelos de Datos (Schemas) ----
class AskRequest(BaseModel):
    filename: str
    question: str

# ---- Utilidades Internas ----
async def get_embedding(text: str) -> List[float]:
    """Obtiene el vector usando Ollama"""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

def extract_text(content: bytes, filename: str) -> str:
    """Extrae texto plano de PDF o Texto"""
    filename = filename.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        else:
            text = content.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
    return text

def chunk_text(text: str, chunk_size=800, overlap=100) -> List[str]:
    """Divide el texto en fragmentos más pequeños"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# ---- Endpoints Finales ----

@app.get("/health")
def health():
    return {"status": "ok", "mode": "demo"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    1. Lee el archivo.
    2. Lo divide en chunks.
    3. Genera vectores (embeddings).
    4. Guarda en Qdrant con el nombre del archivo como filtro.
    """
    content = await file.read()
    text = extract_text(content, file.filename)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="El archivo está vacío o no se pudo leer.")

    chunks = chunk_text(text)
    points = []

    print(f"Procesando {len(chunks)} fragmentos para {file.filename}...")

    for i, chunk in enumerate(chunks):
        vector = await get_embedding(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "filename": file.filename,
                    "text": chunk,
                    "chunk_index": i
                }
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {"status": "ok", "filename": file.filename, "chunks": len(points)}

@app.post("/ask")
async def ask_question(req: AskRequest):
    """
    1. Vectoriza la pregunta.
    2. Busca en Qdrant filtrando POR NOMBRE DE ARCHIVO.
    3. Envía contexto + pregunta a Ollama.
    """
    query_vector = await get_embedding(req.question)

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        query_filter=Filter(
            must=[
                FieldCondition(key="filename", match=MatchValue(value=req.filename))
            ]
        )
    )

    if not search_result:
        return {"answer": "No encontré información en ese archivo. Asegúrate de que el nombre sea exacto."}

    context_text = "\n\n".join([hit.payload["text"] for hit in search_result])

    prompt = f"""
    Usa la siguiente información del documento '{req.filename}' para responder a la pregunta.
    Si la respuesta no está en el texto, di "No lo sé basándome en el documento".

    CONTEXTO:
    {context_text}

    PREGUNTA:
    {req.question}
    
    RESPUESTA:
    """

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False 
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {resp.text}")
            
        result_text = resp.json().get("response", "")

    return {
        "filename_used": req.filename,
        "question": req.question,
        "answer": result_text
    }