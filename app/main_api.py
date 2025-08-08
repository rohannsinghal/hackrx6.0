# app/main_api.py
import psutil
import os
import json
import uuid
import time
import re
from typing import List, Dict, Any, Optional
import logging
import asyncio
from collections import defaultdict

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

# --- NEW: API KEY MANAGER CLASS ---
class GroqAPIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [key.strip() for key in api_keys if key.strip()]  # Clean keys
        self.key_usage_count = defaultdict(int)
        self.key_last_used = defaultdict(float)
        self.current_key_index = 0
        self.max_requests_per_key = 45  # Conservative limit per hour (Groq free tier is ~100)
        logger.info(f"🔑 API Key Manager initialized with {len(self.api_keys)} keys")
        
    def get_next_api_key(self):
        """Get the next available API key with smart rotation"""
        current_time = time.time()
        
        # Reset counters every hour
        for key in self.api_keys:
            if current_time - self.key_last_used[key] > 3600:  # 1 hour
                self.key_usage_count[key] = 0
        
        # Find the key with lowest usage
        best_key = min(self.api_keys, 
                      key=lambda k: self.key_usage_count[k])
        
        # If all keys are at limit, use round-robin as fallback
        if self.key_usage_count[best_key] >= self.max_requests_per_key:
            best_key = self.api_keys[self.current_key_index % len(self.api_keys)]
            self.current_key_index += 1
        
        # Update usage tracking
        self.key_usage_count[best_key] += 1
        self.key_last_used[best_key] = current_time
        
        return best_key
    
    def get_key_stats(self):
        """Get usage statistics for all keys"""
        return {
            f"...{key[-4:]}": {
                "usage_count": self.key_usage_count[key],
                "last_used": self.key_last_used[key]
            }
            for key in self.api_keys
        }

# --- NEW: RESPONSE CLEANER CLASS ---
class ResponseCleaner:
    """Clean and format RAG responses for better readability"""
    
    def __init__(self):
        # Patterns to remove
        self.removal_patterns = [
            r'Document Section \d+[,\s]*',  # Remove "Document Section X"
            r'Section \d+\.\d+[,\s]*',      # Remove "Section X.X"
            r'\*\*.*?\*\*',                 # Remove bold markdown
            r'According to [^,]*,\s*',      # Remove "According to..."
            r'The policy document states[^,]*,\s*',  # Remove policy document references
            r'As mentioned in [^,]*,\s*',   # Remove "As mentioned in..."
            r'Based on [^,]*,\s*',          # Remove "Based on..."
            r'Additionally,\s*',            # Remove "Additionally,"
            r'Furthermore,\s*',             # Remove "Furthermore,"
            r'Moreover,\s*',                # Remove "Moreover,"
        ]
    
    def clean_response(self, response: str) -> str:
        """Clean and format a response"""
        if not response or response.strip() == "":
            return "The information is not available in the provided policy document."
        
        cleaned = response
        
        # Step 1: Remove escaped characters and newlines
        cleaned = cleaned.replace('\\n', ' ')
        cleaned = cleaned.replace('\\t', ' ')
        cleaned = cleaned.replace('\\"', '"')
        cleaned = re.sub(r'\n+', ' ', cleaned)  # Multiple newlines to single space
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
        
        # Step 2: Remove unwanted patterns
        for pattern in self.removal_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Step 3: Fix formatting issues
        cleaned = re.sub(r',\s*,', ',', cleaned)  # Double commas
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)  # Space around commas
        cleaned = re.sub(r'\s*\.\s*', '. ', cleaned)  # Space around periods
        cleaned = re.sub(r'\s+([.,:;!?])', r'\1', cleaned)  # Remove space before punctuation
        cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)  # Ensure space after sentence end
        
        # Step 4: Capitalize first letter and ensure proper ending
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 0:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned.strip()

# --- CONFIGURATION & INITIALIZATION ---
# API Key Setup
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(',')
if not all(GROQ_API_KEYS) or GROQ_API_KEYS == [""]:
    logger.warning("GROQ_API_KEYS not found in .env file. Using placeholder.")
    GROQ_API_KEYS = ["gsk_YourDefaultKeyHere"] 

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "./app/chroma_db"
UPLOAD_DIR = "/tmp/docs"

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
        
        # Initialize enhanced API key manager
        app.state.api_key_manager = GroqAPIKeyManager(GROQ_API_KEYS)
        logger.info(f"✅ API key manager initialized with {len(GROQ_API_KEYS)} keys")
        
        # Initialize Groq client with first key
        first_key = app.state.api_key_manager.get_next_api_key()
        app.state.groq_client = groq.Groq(api_key=first_key)
        logger.info("✅ Groq client initialized")
        
        # Initialize parsing service
        app.state.parsing_service = FastDocumentParserService()
        logger.info("✅ Parsing service initialized")
        
        # Initialize response cleaner
        app.state.response_cleaner = ResponseCleaner()
        logger.info("✅ Response cleaner initialized")
        
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
1. Answer directly and concisely - avoid unnecessary phrases
2. Include specific numbers, percentages, and timeframes when mentioned
3. If multiple conditions exist, list them clearly
4. For waiting periods, grace periods, limits - provide exact values
5. If the answer is not in the documents, state "The information is not available in the provided policy document."
6. Focus on the most specific and relevant information available
7. When quoting policy terms, be precise with terminology
8. Remove any formatting characters like \\n from your response
9. Do not include section headings or reference markers in your final answer
10. Provide clean, readable text without markdown or special characters"""
        
        user_prompt = f"""POLICY DOCUMENT SECTIONS:
{context}

QUESTION: {query}

Provide a direct, accurate answer based on the policy document. Include specific details like timeframes, amounts, conditions, and percentages where available. Make sure your response is clean and readable without any formatting characters."""
        
        try:
            # Get next available API key using the manager
            api_key = self.request.app.state.api_key_manager.get_next_api_key()
            self.groq_client.api_key = api_key
            logger.info(f"Using Groq API key ending in ...{api_key[-4:]}")
            
            response = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,  # Even lower for more consistency
                max_tokens=400,
                top_p=0.85
            )
            
            # Get raw response
            raw_answer = response.choices[0].message.content.strip()
            
            # Clean the response using the ResponseCleaner
            cleaned_answer = self.request.app.state.response_cleaner.clean_response(raw_answer)
            
            return cleaned_answer
            
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

# NEW: Debug endpoint to check API key usage
@app.get("/debug/api-keys")
async def get_api_key_stats(request: Request):
    """Get API key usage statistics"""
    try:
        stats = request.app.state.api_key_manager.get_key_stats()
        return {
            "total_keys": len(request.app.state.api_key_manager.api_keys),
            "key_usage": stats
        }
    except Exception as e:
        logger.error(f"Error getting API key stats: {e}")
        return {"error": "Could not retrieve API key statistics"}

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