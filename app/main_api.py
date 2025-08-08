# Enhanced main_api.py with LangChain integration

import psutil
import os
import json
import uuid
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from collections import defaultdict

# FastAPI and core dependencies
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain imports - Core components
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

# Additional LangChain utilities
from langchain.schema.retriever import BaseRetriever
from langchain.schema.document import Document as LangChainDocument
from langchain.retrievers.merger_retriever import MergerRetriever

# Embeddings and Vector DB
from sentence_transformers import SentenceTransformer
import chromadb

# LLM Integration
import groq

# ML Libraries for enhanced similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein

# Document processing
from .parser import FastDocumentParserService
import httpx
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRx 6.0 LangChain Enhanced RAG System", version="LANGCHAIN_ENHANCED")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CUSTOM GROQ LLM FOR LANGCHAIN (Corrected) ---
class GroqLLM(LLM):
    """Custom Groq LLM wrapper for LangChain"""
    
    # Declare fields as class attributes
    groq_client: Any
    api_key_manager: Any
    
    class Config:
        """Configuration for this Pydantic model."""
        arbitrary_types_allowed = True
        
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        try:
            # Get next available API key
            api_key = self.api_key_manager.get_next_api_key()
            self.groq_client.api_key = api_key
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=500,
                top_p=0.85,
                stop=stop
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return "Error generating response"
        

# --- ENHANCED DOCUMENT PROCESSOR WITH LANGCHAIN ---
class LangChainDocumentProcessor:
    """Enhanced document processing with LangChain text splitters"""
    
    def __init__(self):
        # Initialize multiple text splitters for different strategies
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=400,  # Token-based chunking
            chunk_overlap=50
        )
        
        # Semantic splitter for better context preservation
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len,
            separators=[
                "\n\n### ",  # Section headers
                "\n\n## ",   # Subsection headers
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence breaks
                " "          # Word breaks
            ]
        )
        
        logger.info("LangChain document processor initialized with multiple splitting strategies")
    
    def process_chunks_to_langchain_docs(self, chunks: List[Dict], split_strategy: str = "semantic") -> List[LangChainDocument]:
        """Convert parsed chunks to LangChain documents with enhanced splitting"""
        langchain_docs = []
        
        for chunk in chunks:
            content = chunk['content'] if isinstance(chunk, dict) else chunk.content
            metadata = chunk['metadata'] if isinstance(chunk, dict) else chunk.metadata
            
            # Create initial document
            doc = LangChainDocument(
                page_content=content,
                metadata=metadata
            )
            
            # Apply splitting strategy
            if split_strategy == "recursive":
                split_docs = self.recursive_splitter.split_documents([doc])
            elif split_strategy == "token":
                split_docs = self.token_splitter.split_documents([doc])
            elif split_strategy == "semantic":
                split_docs = self.semantic_splitter.split_documents([doc])
            else:
                split_docs = [doc]  # No splitting
            
            # Enhance metadata for each split
            for i, split_doc in enumerate(split_docs):
                split_doc.metadata.update({
                    'split_index': i,
                    'total_splits': len(split_docs),
                    'split_strategy': split_strategy,
                    'original_chunk_id': metadata.get('chunk_id', str(uuid.uuid4()))
                })
                langchain_docs.append(split_doc)
        
        logger.info(f"Processed {len(chunks)} chunks into {len(langchain_docs)} LangChain documents using {split_strategy} strategy")
        return langchain_docs

# --- ADVANCED RETRIEVER WITH LANGCHAIN ---
class AdvancedLangChainRetriever:
    """Advanced retriever using LangChain's sophisticated retrieval methods"""
    
    def __init__(self, vectorstore: Chroma, llm: GroqLLM):
        self.vectorstore = vectorstore
        self.llm = llm
        
        # Base retrievers
        self.similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        self.mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 8, "fetch_k": 20}
        )
        
        # Multi-query retriever for query expansion
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.similarity_retriever,
            llm=llm,
            verbose=True
        )
        
        # Ensemble retriever combining multiple strategies
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.similarity_retriever, self.mmr_retriever],
            weights=[0.6, 0.4]  # Weight similarity higher than MMR
        )
        
        # Contextual compression retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm),
            base_retriever=self.ensemble_retriever
        )
        
        logger.info("Advanced LangChain retriever initialized with multiple retrieval strategies")
    
    async def retrieve_documents(self, query: str, strategy: str = "ensemble", k: int = 10) -> List[LangChainDocument]:
        """Retrieve documents using specified strategy"""
        try:
            if strategy == "similarity":
                docs = await asyncio.to_thread(
                    self.similarity_retriever.get_relevant_documents, query
                )
            elif strategy == "mmr":
                docs = await asyncio.to_thread(
                    self.mmr_retriever.get_relevant_documents, query
                )
            elif strategy == "multi_query":
                docs = await asyncio.to_thread(
                    self.multi_query_retriever.get_relevant_documents, query
                )
            elif strategy == "ensemble":
                docs = await asyncio.to_thread(
                    self.ensemble_retriever.get_relevant_documents, query
                )
            elif strategy == "compression":
                docs = await asyncio.to_thread(
                    self.compression_retriever.get_relevant_documents, query
                )
            else:
                docs = await asyncio.to_thread(
                    self.ensemble_retriever.get_relevant_documents, query
                )
            
            # Limit results
            docs = docs[:k]
            
            logger.info(f"Retrieved {len(docs)} documents using {strategy} strategy for query: '{query}'")
            return docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed with strategy {strategy}: {e}")
            # Fallback to simple similarity search
            try:
                docs = await asyncio.to_thread(
                    self.similarity_retriever.get_relevant_documents, query
                )
                return docs[:k]
            except Exception as e2:
                logger.error(f"Fallback retrieval also failed: {e2}")
                return []

# --- INTELLIGENT QA CHAIN WITH LANGCHAIN ---
class IntelligentQAChain:
    """Intelligent QA chain with LangChain prompt engineering"""
    
    def __init__(self, retriever: AdvancedLangChainRetriever, llm: GroqLLM):
        self.retriever = retriever
        self.llm = llm
        
        # Custom prompt template for insurance policy questions
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert insurance policy analyst with deep understanding of policy documents and insurance terminology.

Context Information (ranked by relevance):
{context}

Question: {question}

Instructions:
1. Analyze the provided context carefully, paying attention to the most relevant sections
2. Provide a direct, precise answer based ONLY on the context provided
3. Include specific details like:
   - Exact timeframes (days, months, years)
   - Specific amounts or percentages
   - Conditions and requirements
   - Exclusions or limitations
   - Special circumstances or exceptions

4. If the information is not available in the context, clearly state: "The information is not available in the provided policy document."

5. For complex questions, structure your answer clearly with relevant details
6. Use professional insurance terminology where appropriate
7. Ensure your answer is complete and addresses all parts of the question

Answer:"""
        )
        
        logger.info("Intelligent QA chain initialized with custom insurance prompt template")
    
    async def answer_question(self, question: str) -> str:
        """Answer question using advanced retrieval and LangChain QA"""
        try:
            # Step 1: Retrieve relevant documents using ensemble approach
            relevant_docs = await self.retriever.retrieve_documents(
                question, 
                strategy="ensemble",  # Use ensemble for balanced results
                k=12
            )
            
            if not relevant_docs:
                return "The information is not available in the provided policy document."
            
            # Step 2: Prepare context with relevance scoring
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                # Add document with metadata
                metadata_info = ""
                if doc.metadata:
                    source = doc.metadata.get('source', 'Unknown')
                    metadata_info = f" [Source: {source}]"
                
                context_parts.append(
                    f"--- SECTION {i+1} ---{metadata_info}\n{doc.page_content}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Step 3: Generate answer using custom prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            answer = await asyncio.to_thread(self.llm._call, prompt)
            
            # Step 4: Clean and validate answer
            cleaned_answer = self._clean_answer(answer)
            
            logger.info(f"Generated answer for question: '{question}' (length: {len(cleaned_answer)} chars)")
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"QA chain failed for question '{question}': {e}")
            return "Error: Could not generate an answer due to processing issues."
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer"""
        if not answer or answer.strip() == "":
            return "The information is not available in the provided policy document."
        
        # Remove common artifacts
        cleaned = answer.strip()
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.replace('\\n', ' ')
        cleaned = cleaned.replace('\\t', ' ')
        
        # Ensure proper capitalization
        if cleaned and len(cleaned) > 0:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        # Ensure proper ending
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned

# --- CONFIGURATION & MANAGERS ---
class GroqAPIKeyManager:
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
        return {
            f"...{key[-4:]}": {
                "usage_count": self.key_usage_count[key],
                "last_used": self.key_last_used[key]
            }
            for key in self.api_keys
        }

# Configuration
GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(',')
if not all(GROQ_API_KEYS) or GROQ_API_KEYS == [""]:
    logger.warning("GROQ_API_KEYS not found in .env file. Using placeholder.")
    GROQ_API_KEYS = ["gsk_YourDefaultKeyHere"] 

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "./app/chroma_db"
UPLOAD_DIR = "/tmp/docs"

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing LangChain enhanced services...")
        
        # Initialize embedding model for LangChain
        app.state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ LangChain embeddings initialized")
        
        # Initialize API key manager and Groq client
        app.state.api_key_manager = GroqAPIKeyManager(GROQ_API_KEYS)
        first_key = app.state.api_key_manager.get_next_api_key()
        app.state.groq_client = groq.Groq(api_key=first_key)
        
        # Initialize custom Groq LLM for LangChain
        # Corrected code
        app.state.groq_llm = GroqLLM(groq_client=app.state.groq_client, api_key_manager=app.state.api_key_manager)
        logger.info("✅ Groq LLM wrapper initialized")
        
        # Initialize document processor
        app.state.doc_processor = LangChainDocumentProcessor()
        logger.info("✅ LangChain document processor initialized")
        
        # Initialize parsing service
        app.state.parsing_service = FastDocumentParserService()
        logger.info("✅ Parsing service initialized")
        
        logger.info("🚀 All LangChain enhanced services initialized successfully!")
        
    except Exception as e:
        logger.error(f"FATAL: Could not initialize services. Error: {e}")
        raise e

# Pydantic Models
class SubmissionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class SubmissionResponse(BaseModel):
    answers: List[Answer]

class LangChainRAGPipeline:
    """Complete RAG pipeline using LangChain components"""
    
    def __init__(self, collection_name: str, request: Request):
        self.collection_name = collection_name
        self.request = request
        self.embedding_model = request.app.state.embedding_model
        self.groq_llm = request.app.state.groq_llm
        self.doc_processor = request.app.state.doc_processor
        
        # Initialize Chroma vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        self.retriever = None
        self.qa_chain = None
        
        logger.info(f"LangChain RAG pipeline initialized for collection: {collection_name}")
    
    def add_documents(self, chunks: List[Any]):
        """Add documents to the LangChain vectorstore"""
        if not chunks:
            logger.warning("No chunks provided to add_documents.")
            return
        
        logger.info(f"Processing {len(chunks)} chunks with LangChain...")
        
        # Convert to LangChain documents with enhanced splitting
        langchain_docs = self.doc_processor.process_chunks_to_langchain_docs(
            chunks, 
            split_strategy="semantic"
        )
        
        # Add to vectorstore
        self.vectorstore.add_documents(langchain_docs)
        
        # Initialize retriever and QA chain
        self.retriever = AdvancedLangChainRetriever(self.vectorstore, self.groq_llm)
        self.qa_chain = IntelligentQAChain(self.retriever, self.groq_llm)
        
        logger.info(f"✅ Added {len(langchain_docs)} documents to LangChain vectorstore")
    
    async def answer_question(self, question: str) -> str:
        """Answer question using the LangChain QA chain"""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please add documents first."
        
        logger.info(f"🤔 Answering question with LangChain: {question}")
        answer = await self.qa_chain.answer_question(question)
        return answer

@app.post("/hackrx/run", response_model=SubmissionResponse)
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    
    parsing_service = request.app.state.parsing_service

    # Cleanup old collections
    try:
        import shutil
        old_dirs = [d for d in os.listdir(CHROMA_PERSIST_DIR) if d.startswith("hackrx_session_")]
        for old_dir in old_dirs:
            shutil.rmtree(os.path.join(CHROMA_PERSIST_DIR, old_dir), ignore_errors=True)
    except Exception as e:
        logger.warning(f"Could not clean up old collections: {e}")

    session_collection_name = f"hackrx_session_{uuid.uuid4().hex}"
    rag_pipeline = LangChainRAGPipeline(collection_name=session_collection_name, request=request)
    
    # Process documents
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
                
                # Parse document
                chunks = parsing_service.process_pdf_ultrafast(temp_file_path)
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                all_chunks.extend(chunk_dicts)
                os.remove(temp_file_path)
                
                logger.info(f"✅ Processed {len(chunks)} chunks from {file_name}")

            except Exception as e:
                logger.error(f"❌ Failed to process document at {doc_url}: {e}")
                continue
    
    if not all_chunks:
        failed_answers = [
            Answer(
                question=q, 
                answer="A valid document could not be processed, so an answer could not be found."
            ) 
            for q in submission_request.questions
        ]
        return SubmissionResponse(answers=failed_answers)

    # Add documents to LangChain pipeline
    rag_pipeline.add_documents(all_chunks)

    # Answer all questions using LangChain
    async def process_question(question: str):
        logger.info(f"🔍 Processing: {question}")
        answer_text = await rag_pipeline.answer_question(question)
        return Answer(question=question, answer=answer_text)

    # Process all questions
    tasks = [process_question(q) for q in submission_request.questions]
    answers = await asyncio.gather(*tasks)

    logger.info(f"🎯 Successfully processed {len(answers)} questions")
    return SubmissionResponse(answers=answers)

@app.get("/debug/api-keys")
async def get_api_key_stats(request: Request):
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
    return {"message": "LangChain Enhanced HackRx 6.0 RAG System is running. See /docs for API details."}

@app.get("/memory")
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "memory_usage_mb": mem_info.rss / (1024 * 1024)
    }