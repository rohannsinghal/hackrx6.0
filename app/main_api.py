# app/main_api.py
import psutil
import os
import json
import uuid
from typing import List, Dict, Any, Optional
import logging
import asyncio
from itertools import cycle

# FastAPI and core dependencies
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Embeddings and Vector DB
from sentence_transformers import SentenceTransformer
import chromadb

# LLM Integration
import groq

# Direct import from our local parser module
from .parser import FastDocumentParserService

# HTTP Client for downloading documents
import httpx

# NEW: Library to load environment variables from .env file
from dotenv import load_dotenv

# Setup
load_dotenv() # Load environment variables from .env file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRx 6.0 RAG System", version="FINAL")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CONFIGURATION & INITIALIZATION ---
# API Key Rotation Setup
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(',')
if not all(GROQ_API_KEYS):
    logger.warning("GROQ_API_KEYS not found in .env file. Using placeholder.")
    # Provide a fallback or raise an error if keys are essential
    GROQ_API_KEYS = ["gsk_YourDefaultKeyHere"] 

api_key_cycler = cycle(GROQ_API_KEYS)

def get_next_api_key():
    return next(api_key_cycler)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "./app/chroma_db"
UPLOAD_DIR = "/tmp/docs" # Use the writable temporary directory

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # The client's API key will be updated per-request
    groq_client = groq.Groq(api_key=get_next_api_key())
    parsing_service = FastDocumentParserService()
except Exception as e:
    logger.error(f"FATAL: Could not initialize models. Error: {e}")

# Pydantic Models for Hackathon
class SubmissionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class SubmissionResponse(BaseModel):
    answers: List[Answer]

class RAGPipeline:
    def __init__(self, collection_name: str, request: Request):
        # --- FIX: Get models and clients from the app state via the request ---
        self.collection_name = collection_name
        self.request = request
        self.chroma_client = request.app.state.chroma_client
        self.embedding_model = request.app.state.embedding_model
        self.groq_client = request.app.state.groq_client
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, chunks: List[Dict]):
        if not chunks:
            logger.warning("No chunks provided to add_documents.")
            return
        
        logger.info(f"Starting to add {len(chunks)} chunks...")
        
        # Use instance variables to access models
        contents = [c['content'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        ids = [c['chunk_id'] for c in chunks]
        
        self.collection.add(
            embeddings=self.embedding_model.encode(contents, show_progress_bar=True).tolist(),
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Finished adding {len(chunks)} chunks to collection '{self.collection_name}'")

    def query_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.collection.count(): return []
        
        results = self.collection.query(
            # Use instance variable for the model
            query_embeddings=self.embedding_model.encode([query]).tolist(),
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas"]
        )
        return [{"content": doc, "metadata": meta} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

    async def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        context = "\n\n".join([f"--- REFERENCE TEXT ---\n{doc['content']}" for doc in context_docs])
        system_prompt = "You are an expert AI assistant. Your task is to answer the user's question based *only* on the provided reference text. Do not use any outside knowledge. If the answer is not contained within the text, you must state 'The answer could not be found in the provided document.' Be concise and directly answer the question."
        user_prompt = f"REFERENCE TEXT:\n{context}\n\nQUESTION: {query}"
        
        try:
            # --- FIX: Access the API key cycler from the app state ---
            self.groq_client.api_key = next(self.request.app.state.api_key_cycler)
            logger.info(f"Using Groq API key ending in ...{self.groq_client.api_key[-4:]}")
            
            response = await asyncio.to_thread(
                # Use instance variable for the client
                self.groq_client.chat.completions.create,
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return "Error: Could not generate an answer from the language model."

@app.post("/hackrx/run", response_model=SubmissionResponse)
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    
    # --- FIX: Access services from the application state via the request object ---
    chroma_client = request.app.state.chroma_client
    parsing_service = request.app.state.parsing_service

    # 1. Cleanup and Setup
    try:
        for collection in chroma_client.list_collections():
            if collection.name.startswith("hackrx_session_"):
                chroma_client.delete_collection(name=collection.name)
    except Exception as e:
        logger.warning(f"Could not clean up old collections: {e}")

    session_collection_name = f"hackrx_session_{uuid.uuid4().hex}"
    # --- FIX: Pass the request object to the RAG pipeline ---
    rag_pipeline = RAGPipeline(collection_name=session_collection_name, request=request)
    
    # 2. Download and Process Documents
    all_chunks = []
    # The UPLOAD_DIR variable should be defined at the top of your file
    UPLOAD_DIR = "/tmp/docs" 
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for doc_url in submission_request.documents:
            try:
                logger.info(f"Downloading document from: {doc_url}")
                response = await client.get(doc_url, follow_redirects=True)
                response.raise_for_status()
                
                file_name = os.path.basename(doc_url.split('?')[0])
                temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file_name}")
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(temp_file_path, "wb") as f:
                    f.write(response.content)
                
                # This call now correctly uses the parsing_service loaded in the app state
                chunks = parsing_service.process_pdf_ultrafast(temp_file_path)
                all_chunks.extend(chunks)
                os.remove(temp_file_path)

            except Exception as e:
                logger.error(f"Failed to process document at {doc_url}: {e}", exc_info=True)
                continue
    
    if not all_chunks:
        failed_answers = [Answer(question=q, answer="A valid document could not be processed, so an answer could not be found.") for q in submission_request.questions]
        return SubmissionResponse(answers=failed_answers)

    # 3. Add to Vector DB
    rag_pipeline.add_documents(all_chunks)

    # 4. Asynchronously answer all questions
    async def answer_question(question: str):
        relevant_docs = rag_pipeline.query_documents(question)
        answer_text = await rag_pipeline.generate_answer(question, relevant_docs)
        return Answer(question=question, answer=answer_text)

    tasks = [answer_question(q) for q in submission_request.questions]
    answers = await asyncio.gather(*tasks)

    return SubmissionResponse(answers=answers)

@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 RAG System is running. See /docs for API details."}

@app.get("/memory")
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "memory_usage_mb": mem_info.rss / (1024 * 1024)  # RSS in MB
    }