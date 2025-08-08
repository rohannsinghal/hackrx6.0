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

# Startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize all services and store them in app state"""
    try:
        logger.info("Initializing services...")
        
        # Initialize embedding model
        app.state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Embedding model initialized")
        
        # Initialize Chroma client
        app.state.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info("✅ ChromaDB client initialized")
        
        # Initialize Groq client
        app.state.groq_client = groq.Groq(api_key=get_next_api_key())
        logger.info("✅ Groq client initialized")
        
        # Initialize parsing service
        app.state.parsing_service = FastDocumentParserService()
        logger.info("✅ Parsing service initialized")
        
        # Initialize API key cycler
        app.state.api_key_cycler = api_key_cycler
        logger.info("✅ API key cycler initialized")
        
        logger.info("🚀 All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"FATAL: Could not initialize services. Error: {e}")
        raise e

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
        # Access models and clients from the app state via the request
        self.collection_name = collection_name
        self.request = request
        self.chroma_client = request.app.state.chroma_client
        self.embedding_model = request.app.state.embedding_model
        self.groq_client = request.app.state.groq_client
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, chunks: List[Any]):
        if not chunks:
            logger.warning("No chunks provided to add_documents.")
            return
        
        logger.info(f"Starting to add {len(chunks)} chunks...")
        
        # Handle both DocumentChunk objects and dictionaries
        contents = []
        metadatas = []
        ids = []
        
        for c in chunks:
            if hasattr(c, 'content'):  # DocumentChunk object
                contents.append(c.content)
                metadatas.append(c.metadata)
                ids.append(c.chunk_id)
            else:  # Dictionary
                contents.append(c['content'])
                metadatas.append(c['metadata'])
                ids.append(c['chunk_id'])
        
        self.collection.add(
            embeddings=self.embedding_model.encode(contents, show_progress_bar=True).tolist(),
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Finished adding {len(chunks)} chunks to collection '{self.collection_name}'")

    def query_documents(self, query: str, n_results: int = 10) -> List[Dict]:
        if not self.collection.count(): return []
        
        # Enhanced query with multiple variations to improve retrieval
        query_variations = [
            query,
            f"policy terms conditions {query}",
            f"insurance coverage {query}",
        ]
        
        all_results = []
        seen_docs = set()
        
        for q in query_variations:
            results = self.collection.query(
                query_embeddings=self.embedding_model.encode([q]).tolist(),
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                # Use distance threshold to ensure quality
                if distance < 1.2 and doc not in seen_docs:  # Lower distance = better match
                    all_results.append({
                        "content": doc, 
                        "metadata": meta,
                        "distance": distance
                    })
                    seen_docs.add(doc)
        
        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n_results]

    async def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        # Sort context docs by relevance (distance) if available
        if context_docs and 'distance' in context_docs[0]:
            context_docs.sort(key=lambda x: x.get('distance', 0))
        
        # Use more context but prioritize most relevant chunks
        context = ""
        for i, doc in enumerate(context_docs[:8]):  # Use top 8 most relevant
            context += f"--- DOCUMENT SECTION {i+1} ---\n{doc['content']}\n\n"
        
        system_prompt = """You are an expert insurance policy analyst. Your task is to provide precise, accurate answers based ONLY on the provided policy document sections.

IMPORTANT INSTRUCTIONS:
1. Answer directly and concisely - avoid unnecessary phrases like "According to the reference text"
2. Include specific numbers, percentages, and timeframes when mentioned
3. If multiple conditions exist, list them clearly
4. For waiting periods, grace periods, limits - provide exact values
5. If the answer is not in the documents, state "The information is not available in the provided policy document."
6. Focus on the most specific and relevant information available
7. When quoting policy terms, be precise with terminology"""
        
        user_prompt = f"""POLICY DOCUMENT SECTIONS:
{context}

QUESTION: {query}

Provide a direct, accurate answer based on the policy document. Include specific details like timeframes, amounts, conditions, and percentages where available."""
        
        try:
            # Access the API key cycler from the app state
            self.groq_client.api_key = next(self.request.app.state.api_key_cycler)
            logger.info(f"Using Groq API key ending in ...{self.groq_client.api_key[-4:]}")
            
            response = await asyncio.to_thread(
                # Use instance variable for the client
                self.groq_client.chat.completions.create,
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1,  # Lower temperature for more consistent answers
                max_tokens=400,   # Increased token limit
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return "Error: Could not generate an answer from the language model."

@app.post("/hackrx/run", response_model=SubmissionResponse)
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    
    # Access services from the application state via the request object
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
    # Pass the request object to the RAG pipeline
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
                # Convert DocumentChunk objects to dictionaries for the RAG pipeline
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                all_chunks.extend(chunk_dicts)
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