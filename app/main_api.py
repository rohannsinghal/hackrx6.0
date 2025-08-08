# --- FINAL main_api.py ---

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
from pydantic import BaseModel

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.document import Document as LangChainDocument

# LLM Integration
import groq

# Document processing and environment
from .parser import FastDocumentParserService
import httpx
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simplified and Corrected RAG System", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CUSTOM GROQ LLM FOR LANGCHAIN ---
class GroqLLM(LLM):
    """Custom Groq LLM wrapper for LangChain"""
    groq_client: Any
    api_key_manager: Any
    
    class Config:
        arbitrary_types_allowed = True
        
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        try:
            api_key = self.api_key_manager.get_next_api_key()
            self.groq_client.api_key = api_key
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05, max_tokens=500, top_p=0.85, stop=stop
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return "Error generating response"
        
# --- THE SIMPLE AND RELIABLE RAG PIPELINE ---
class LangChainRAGPipeline:
    """A simplified and reliable RAG pipeline using standard LangChain components."""
    
    def __init__(self, collection_name: str, request: Request):
        self.collection_name = collection_name
        self.embedding_model = request.app.state.embedding_model
        self.groq_llm = request.app.state.groq_llm
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
        self.qa_chain = None
        logger.info(f"✅ Simple RAG pipeline initialized for collection: {collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Adds documents to the vectorstore and creates the QA chain."""
        if not chunks:
            return
        
        langchain_docs = [LangChainDocument(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in chunks]
        self.vectorstore.add_documents(langchain_docs)
        
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert insurance policy analyst. Use the following pieces of context to answer the question at the end.
Provide a direct and precise answer based ONLY on the provided context. If the answer is not in the context, state "The information is not available in the provided policy document."

Context:
{context}

Question: {question}

Answer:"""
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.groq_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template}
        )
        logger.info(f"✅ QA Chain is ready.")
    
    async def answer_question(self, question: str) -> str:
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please add documents first."
        
        logger.info(f"🤔 Answering question with Simple RAG Pipeline: {question}")
        try:
            result = await asyncio.to_thread(self.qa_chain, {"query": question})
            return result.get("result", "Failed to get an answer.")
        except Exception as e:
            logger.error(f"Error during QA chain execution: {e}")
            return "An error occurred while processing the question."

# --- CONFIGURATION & API KEY MANAGER ---
class GroqAPIKeyManager:
    # (Your GroqAPIKeyManager code is correct, no changes needed here)
    def __init__(self, api_keys: List[str]):
        self.api_keys = [key.strip() for key in api_keys if key.strip()]
        self.key_usage_count = defaultdict(int)
        self.key_last_used = defaultdict(float)
        self.current_key_index = 0
        self.max_requests_per_key = 45
        logger.info(f"🔑 API Key Manager initialized with {len(self.api_keys)} keys")
    def get_next_api_key(self):
        current_time = time.time()
        for key in self.api_keys:
            if current_time - self.key_last_used[key] > 3600:
                self.key_usage_count[key] = 0
        best_key = min(self.api_keys, key=lambda k: self.key_usage_count[k])
        if self.key_usage_count[best_key] >= self.max_requests_per_key:
            best_key = self.api_keys[self.current_key_index % len(self.api_keys)]
            self.current_key_index += 1
        self.key_usage_count[best_key] += 1
        self.key_last_used[best_key] = current_time
        return best_key
    def get_key_stats(self):
        return {f"...{key[-4:]}": {"usage_count": self.key_usage_count[key], "last_used": self.key_last_used[key]} for key in self.api_keys}

# --- APP STARTUP & CONFIG ---
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(',')
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "./app/chroma_db"
UPLOAD_DIR = "/tmp/docs"

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing services...")
        app.state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
        )
        app.state.api_key_manager = GroqAPIKeyManager(GROQ_API_KEYS)
        first_key = app.state.api_key_manager.get_next_api_key()
        app.state.groq_client = groq.Groq(api_key=first_key)
        app.state.groq_llm = GroqLLM(groq_client=app.state.groq_client, api_key_manager=app.state.api_key_manager)
        app.state.parsing_service = FastDocumentParserService()
        logger.info("🚀 All services initialized successfully!")
    except Exception as e:
        logger.error(f"FATAL: Could not initialize services. Error: {e}")
        raise e

# --- API MODELS ---
class SubmissionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class SubmissionResponse(BaseModel):
    answers: List[Answer]

# --- API ENDPOINTS ---
@app.post("/hackrx/run", response_model=SubmissionResponse)
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    parsing_service = request.app.state.parsing_service
    session_collection_name = f"hackrx_session_{uuid.uuid4().hex}"
    rag_pipeline = LangChainRAGPipeline(collection_name=session_collection_name, request=request)
    all_chunks = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for doc_url in submission_request.documents:
            try:
                logger.info(f"📥 Downloading document from: {doc_url}")
                response = await client.get(doc_url, follow_redirects=True)
                response.raise_for_status()
                file_name = os.path.basename(doc_url.split('?')[0])
                temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file_name}")
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(temp_file_path, "wb") as f:
                    f.write(response.content)
                
                chunks = parsing_service.process_pdf_ultrafast(temp_file_path)
                all_chunks.extend([chunk.to_dict() for chunk in chunks])
                os.remove(temp_file_path)
                logger.info(f"✅ Processed {len(chunks)} chunks from {file_name}")
            except Exception as e:
                logger.error(f"❌ Failed to process document at {doc_url}: {e}")
                continue
    
    if not all_chunks:
        failed_answers = [Answer(question=q, answer="A valid document could not be processed.") for q in submission_request.questions]
        return SubmissionResponse(answers=failed_answers)

    rag_pipeline.add_documents(all_chunks)

    tasks = [rag_pipeline.answer_question(q) for q in submission_request.questions]
    results = await asyncio.gather(*tasks)
    answers = [Answer(question=q, answer=ans) for q, ans in zip(submission_request.questions, results)]

    logger.info(f"🎯 Successfully processed {len(answers)} questions")
    return SubmissionResponse(answers=answers)

@app.get("/")
def read_root():
    return {"message": "LangChain Enhanced RAG System is running."}