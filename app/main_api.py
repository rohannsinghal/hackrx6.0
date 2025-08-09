# --- KAGGLE-POWERED RAG SYSTEM (NO LOCAL MODELS) ---

import os
import json
import uuid
import time
import re
import asyncio
import logging
import hashlib
import httpx
from typing import List, Dict, Any, Optional
from collections import defaultdict
from itertools import cycle
from pathlib import Path

# FastAPI and core dependencies
from fastapi import FastAPI, Body, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports (NO SENTENCE TRANSFORMERS!)
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document as LangChainDocument

# Multi-format document processing (KEEPING ALL YOUR PROCESSORS)
import fitz  # PyMuPDF
import pdfplumber
import docx
import openpyxl
import csv
import zipfile
import email
from email.policy import default
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# LLM providers (KEEPING YOUR MULTI-LLM SETUP)
import groq
import openai
import google.generativeai as genai

import cachetools
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kaggle-Powered Hackathon RAG", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KAGGLE MODEL CLIENT (REPLACES ALL LOCAL MODELS) ---
class KaggleModelClient:
    def __init__(self, kaggle_endpoint: str):
        self.kaggle_endpoint = kaggle_endpoint.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"🎯 Kaggle Model Client initialized: {kaggle_endpoint}")
        
    async def health_check(self) -> bool:
        """Check if Kaggle model server is healthy"""
        try:
            response = await self.client.get(f"{self.kaggle_endpoint}/health")
            return response.status_code == 200
        except:
            return False
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Kaggle GPU"""
        try:
            response = await self.client.post(
                f"{self.kaggle_endpoint}/embed",
                json={"texts": texts}
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"🎯 Kaggle embeddings: {result.get('count', 0)} texts in {result.get('processing_time', 0):.2f}s")
            return result["embeddings"]
        except Exception as e:
            logger.error(f"Kaggle embedding error: {e}")
            return []
    
    async def rerank_documents(self, query: str, documents: List[str], k: int = 8) -> List[str]:
        """Rerank documents using Kaggle GPU"""
        try:
            response = await self.client.post(
                f"{self.kaggle_endpoint}/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "k": k
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"🎯 Kaggle reranking: {k} docs in {result.get('processing_time', 0):.2f}s")
            return result["reranked_documents"]
        except Exception as e:
            logger.error(f"Kaggle reranking error: {e}")
            return documents[:k]  # Fallback to original order

# --- NO MORE SEMANTIC PROCESSOR CLASS (MOVED TO KAGGLE) ---
class LightweightQueryProcessor:
    def __init__(self, kaggle_client: KaggleModelClient):
        self.kaggle_client = kaggle_client
        self.cache = cachetools.TTLCache(maxsize=200, ttl=1800)
        
    async def enhance_query_semantically(self, question: str) -> str:
        """Lightweight query enhancement (no heavy models)"""
        cache_key = hashlib.md5(question.encode()).hexdigest()[:8]
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simple domain expansion (no models needed)
        key_expansions = {
            'grace period': 'payment deadline premium due',
            'waiting period': 'exclusion time coverage delay',
            'pre-existing': 'prior medical condition',
            'coverage': 'policy benefits protection',
            'exclusion': 'limitations restrictions',
            'premium': 'insurance cost payment',
            'claim': 'benefit request reimbursement'
        }
        
        query_lower = question.lower()
        for key_term, expansion in key_expansions.items():
            if key_term in query_lower:
                enhanced = f"{question}. Also: {expansion}"
                self.cache[cache_key] = enhanced
                return enhanced
        
        self.cache[cache_key] = question
        return question

# --- ANTI-JAILBREAK SECURITY (KEEPING THIS LOCAL) ---
class SecurityGuard:
    def __init__(self):
        self.jailbreak_patterns = [
            r'ignore.*previous.*instructions',
            r'act.*as.*different.*character',
            r'generate.*code.*(?:javascript|python|html)',
            r'write.*program',
            r'roleplay.*as',
            r'pretend.*you.*are'
        ]
    
    def detect_jailbreak(self, text: str) -> bool:
        """Detect jailbreak attempts"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.jailbreak_patterns)
    
    def sanitize_response(self, question: str, answer: str) -> str:
        """Sanitize responses against jailbreaks"""
        if self.detect_jailbreak(question):
            return "I can only provide information based on the document content provided."
        return answer

# --- MULTI-LLM MANAGER (KEEPING YOUR EXCELLENT SETUP) ---
class MultiLLMManager:
    def __init__(self):
        self.providers = ['groq']
        self.groq_keys = cycle([k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(',') if k.strip()])
        
        # Optional providers
        openai_keys = [k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(',') if k.strip()]
        if openai_keys:
            self.providers.append('openai')
            self.openai_keys = cycle(openai_keys)
        
        self.current_provider_index = 0
        logger.info(f"🔑 Multi-LLM Manager: {len(self.providers)} providers")
    
    async def get_response(self, prompt: str, max_tokens: int = 900) -> str:
        """Get response with automatic fallback"""
        for attempt in range(len(self.providers)):
            try:
                provider = self.providers[self.current_provider_index]
                
                if provider == 'groq':
                    return await self._groq_response(prompt, max_tokens)
                elif provider == 'openai':
                    return await self._openai_response(prompt, max_tokens)
                    
            except Exception as e:
                logger.warning(f"{provider} failed: {e}")
                self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
                continue
        
        return "Error: All LLM providers failed"
    
    async def _groq_response(self, prompt: str, max_tokens: int) -> str:
        key = next(self.groq_keys)
        client = groq.Groq(api_key=key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()

# --- UNIVERSAL DOCUMENT PROCESSOR (KEEPING ALL YOUR EXCELLENT FEATURES) ---
class UniversalDocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_chunks = 200
        self.max_pages = 18
        self.cache = cachetools.TTLCache(maxsize=50, ttl=1800)
        
        self.processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.xlsx': self.process_excel,
            '.csv': self.process_csv,
            '.txt': self.process_text,
            '.html': self.process_html,
            '.json': self.process_json
        }
        
        logger.info("⚡ Universal Document Processor (No Local Models)")
    
    # ... (KEEPING ALL YOUR EXISTING DOCUMENT PROCESSING METHODS)
    # I'll just show the key ones for brevity
    
    async def process_document(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process document (same as your existing method)"""
        file_hash = hashlib.md5(content).hexdigest()[:8]
        
        if file_hash in self.cache:
            logger.info(f"📦 Cache hit")
            return self.cache[file_hash]
        
        file_ext = Path(file_path).suffix.lower()
        if not file_ext:
            file_ext = self._detect_file_type(content)
        
        processor = self.processors.get(file_ext, self.process_text)
        
        try:
            chunks = await processor(file_path, content)
            self.cache[file_hash] = chunks
            logger.info(f"✅ Processed {os.path.basename(file_path)}: {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return []
    
    # ... (ALL YOUR EXISTING PROCESSING METHODS STAY THE SAME)

# --- LIGHTWEIGHT EMBEDDING WRAPPER (FOR CHROMA) ---
class KaggleEmbeddingWrapper:
    def __init__(self, kaggle_client: KaggleModelClient):
        self.kaggle_client = kaggle_client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Kaggle (sync wrapper for Chroma)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.kaggle_client.generate_embeddings(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using Kaggle (sync wrapper for Chroma)"""
        loop = asyncio.get_event_loop()
        embeddings = loop.run_until_complete(self.kaggle_client.generate_embeddings([text]))
        return embeddings[0] if embeddings else []

# --- KAGGLE-POWERED RAG PIPELINE ---
class KagglePoweredRAGPipeline:
    def __init__(self, collection_name: str, llm_manager: MultiLLMManager, kaggle_client: KaggleModelClient):
        self.collection_name = collection_name
        self.llm_manager = llm_manager
        self.kaggle_client = kaggle_client
        self.security_guard = SecurityGuard()
        self.query_processor = LightweightQueryProcessor(kaggle_client)
        
        # Use Kaggle for embeddings via wrapper
        self.embedding_function = KaggleEmbeddingWrapper(kaggle_client)
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory="/tmp/chroma_kaggle"
        )
        
        logger.info(f"🎯 Kaggle-Powered RAG Pipeline initialized")
    
    async def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents using Kaggle embeddings"""
        if not chunks:
            return
        
        logger.info(f"📚 Processing {len(chunks)} chunks with Kaggle...")
        
        # Quick quality filtering
        quality_chunks = [
            chunk for chunk in chunks 
            if not chunk['metadata'].get('error') and len(chunk['content']) > 100
        ][:100]  # Limit for speed
        
        documents = [
            LangChainDocument(
                page_content=chunk['content'],
                metadata=chunk['metadata']
            )
            for chunk in quality_chunks
        ]
        
        if documents:
            # This will call Kaggle for embeddings
            self.vectorstore.add_documents(documents)
            logger.info(f"✅ Added {len(documents)} documents using Kaggle embeddings")
    
    async def answer_question(self, question: str) -> str:
        """Answer question using Kaggle for reranking"""
        
        # Security check
        if self.security_guard.detect_jailbreak(question):
            return self.security_guard.sanitize_response(question, "")
        
        try:
            # Lightweight query enhancement
            enhanced_question = await self.query_processor.enhance_query_semantically(question)
            
            # Local retrieval (using Kaggle embeddings)
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 15}
            )
            
            relevant_docs = retriever.get_relevant_documents(enhanced_question)
            
            if not relevant_docs:
                return "I don't have sufficient information to answer this question."
            
            # Use Kaggle GPU for reranking
            doc_contents = [doc.page_content for doc in relevant_docs]
            
            if await self.kaggle_client.health_check():
                logger.info("🎯 Using Kaggle GPU for reranking")
                top_docs_content = await self.kaggle_client.rerank_documents(
                    enhanced_question, doc_contents, k=6
                )
            else:
                logger.warning("📦 Kaggle unavailable, using first 6 docs")
                top_docs_content = doc_contents[:6]
            
            # Prepare context
            context = "\n\n".join(top_docs_content)
            
            # Create prompt
            prompt = f"""You are an expert insurance policy analyst.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Provide a clear, accurate answer with specific details from the policy.

ANSWER:"""
            
            # Get response from LLM
            response = await self.llm_manager.get_response(prompt)
            
            # Clean and return
            response = self.security_guard.sanitize_response(question, response)
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Question processing failed: {e}")
            return "An error occurred while processing your question."

# --- GLOBAL INSTANCES ---
multi_llm = MultiLLMManager()
doc_processor = UniversalDocumentProcessor()

# Set your Kaggle ngrok endpoint here
KAGGLE_ENDPOINT = os.getenv("KAGGLE_ENDPOINT", "https://f946fa884fe6.ngrok-free.app")
kaggle_client = KaggleModelClient(KAGGLE_ENDPOINT)

# --- API MODELS ---
class SubmissionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- AUTHENTICATION ---
async def verify_bearer_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    return authorization.replace("Bearer ", "")

# --- MAIN ENDPOINT ---
@app.post("/hackrx/run", response_model=SubmissionResponse, dependencies=[Depends(verify_bearer_token)])
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    start_time = time.time()
    logger.info(f"🎯 KAGGLE-POWERED PROCESSING: {len(submission_request.documents)} docs, {len(submission_request.questions)} questions")
    
    try:
        # Check Kaggle health
        if not await kaggle_client.health_check():
            logger.error("❌ Kaggle endpoint not available!")
            return SubmissionResponse(answers=[
                "Model service unavailable" for _ in submission_request.questions
            ])
        
        session_id = f"kaggle_{uuid.uuid4().hex[:6]}"
        rag_pipeline = KagglePoweredRAGPipeline(session_id, multi_llm, kaggle_client)
        
        # Process documents (same as your existing logic)
        all_chunks = []
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            async def process_document(doc_idx: int, doc_url: str):
                try:
                    logger.info(f"📥 Downloading document {doc_idx + 1}")
                    response = await client.get(doc_url, follow_redirects=True)
                    response.raise_for_status()
                    
                    filename = os.path.basename(doc_url.split('?')[0]) or f"document_{doc_idx}"
                    chunks = await doc_processor.process_document(filename, response.content)
                    
                    logger.info(f"✅ Document {doc_idx + 1}: {len(chunks)} chunks")
                    return chunks
                except Exception as e:
                    logger.error(f"❌ Document {doc_idx + 1} failed: {e}")
                    return []
            
            # Process all documents concurrently
            tasks = [process_document(i, url) for i, url in enumerate(submission_request.documents)]
            results = await asyncio.gather(*tasks)
            
            for chunks in results:
                all_chunks.extend(chunks)
        
        logger.info(f"📊 Total chunks: {len(all_chunks)}")
        
        if not all_chunks:
            return SubmissionResponse(answers=[
                "No content extracted" for _ in submission_request.questions
            ])
        
        # Add to RAG pipeline (will use Kaggle for embeddings)
        await rag_pipeline.add_documents(all_chunks)
        
        # Answer questions (will use Kaggle for reranking)
        tasks = [rag_pipeline.answer_question(q) for q in submission_request.questions]
        answers = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 KAGGLE-POWERED SUCCESS! Processed in {elapsed:.2f}s")
        
        return SubmissionResponse(answers=answers)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"💥 ERROR: {elapsed:.2f}s: {e}")
        
        return SubmissionResponse(answers=[
            "Processing error" for _ in submission_request.questions
        ])

@app.get("/")
def read_root():
    return {
        "message": "🎯 KAGGLE-POWERED HACKATHON RAG",
        "version": "5.0.0",
        "status": "No local models, all GPU processing on Kaggle!",
        "kaggle_endpoint": KAGGLE_ENDPOINT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
