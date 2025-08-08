# --- STANDALONE main_api.py with embedded parser ---

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
from pathlib import Path
import gc

# FastAPI and core dependencies
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports (using updated non-deprecated imports)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.document import Document as LangChainDocument

# Document processing imports
import fitz  # PyMuPDF
import pdfplumber
import mammoth
import email
import email.policy
from bs4 import BeautifulSoup

# LLM Integration
import groq

# Other dependencies
import httpx
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Standalone Fixed RAG System", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- EMBEDDED DOCUMENT PARSER ---

class DocumentChunk:
    """Simple data class for document chunks"""
    def __init__(self, content: str, metadata: Dict[str, Any], chunk_id: str):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id

    def to_dict(self):
        return {
            "content": self.content,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }

class EmbeddedDocumentParser:
    """Embedded document parsing service with improved chunking"""
    def __init__(self):
        self.chunk_size = 1500  # Reduced for better semantic chunks
        self.chunk_overlap = 150
        self.max_chunks = 200   # Reduced from 500
        self.table_row_limit = 20
        logger.info("EmbeddedDocumentParser initialized")

    def improved_text_split(self, text: str, source: str) -> List[str]:
        """Improved semantic text splitting that preserves content structure"""
        if not text or len(text) < 100:
            return [text] if text else []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        chunk_count = 0
        
        # Look for section boundaries first
        section_patterns = [
            r'\n\s*\d+\.\s*[A-Z]',  # Numbered sections like "3.1 DEFINITIONS"
            r'\n\s*[A-Z][A-Z\s]{10,}:\s*\n',  # CAPITALIZED HEADINGS
            r'\n\s*SECTION\s+\d+',  # SECTION markers
            r'\n\s*PART\s+\d+',     # PART markers
            r'\n\s*EXCLUSIONS',     # Important sections
            r'\n\s*COVERAGE',
            r'\n\s*BENEFITS',
        ]
        
        while start < len(text) and chunk_count < self.max_chunks:
            end = min(start + self.chunk_size, len(text))
            
            if end < len(text):
                # Try to find section boundary first
                section_break = -1
                search_start = max(start + self.chunk_size // 2, end - 500)
                
                for pattern in section_patterns:
                    matches = list(re.finditer(pattern, text[search_start:end]))
                    if matches:
                        section_break = search_start + matches[-1].start()
                        break
                
                if section_break > start:
                    end = section_break
                else:
                    # Fall back to sentence boundary
                    search_start = max(start, end - 300)
                    period_pos = text.rfind('.', search_start, end)
                    newline_pos = text.rfind('\n', search_start, end)
                    
                    best_pos = max(period_pos, newline_pos)
                    if best_pos > search_start:
                        end = best_pos + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Only keep substantial chunks
                chunks.append(chunk)
                chunk_count += 1
            
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        logger.info(f"Split {source} into {len(chunks)} semantic chunks")
        return chunks

    def extract_tables_fast(self, file_path: str) -> str:
        """Fast table extraction with better filtering"""
        table_text = ""
        table_count = 0
        max_tables = 15  # Reduced from 25

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Process fewer pages more carefully
                if total_pages <= 16:
                    pages_to_process = list(range(total_pages))
                else:
                    pages_to_process = list(range(0, min(total_pages, 20), 2))

                logger.info(f"📊 Processing {len(pages_to_process)} of {total_pages} pages for tables")

                for page_num in pages_to_process:
                    if table_count >= max_tables:
                        break

                    page = pdf.pages[page_num]
                    tables = page.find_tables()

                    for table in tables:
                        if table_count >= max_tables:
                            break

                        try:
                            table_data = table.extract()
                            if table_data and len(table_data) >= 2 and len(table_data[0]) <= 8:
                                
                                # Skip repetitive tables (like ombudsman addresses)
                                first_row = str(table_data[0]).lower()
                                if ('ombudsman' in first_row or 
                                    'lalit bhawan' in first_row or
                                    len(set(str(row) for row in table_data[:5])) < 3):  # Too repetitive
                                    continue
                                
                                limited_data = table_data[:min(20, len(table_data))]
                                header = " | ".join(str(cell or "").strip()[:40] for cell in limited_data[0])
                                separator = " | ".join(["---"] * len(limited_data[0]))
                                
                                rows = []
                                for row in limited_data[1:]:
                                    padded_row = list(row) + [None] * (len(limited_data[0]) - len(row))
                                    row_str = " | ".join(str(cell or "").strip()[:40] for cell in padded_row)
                                    rows.append(row_str)

                                table_md = f"\n**TABLE {table_count + 1} - Page {page_num + 1}**\n"
                                table_md += f"| {header} |\n| {separator} |\n"
                                for row in rows:
                                    table_md += f"| {row} |\n"
                                table_md += "\n"

                                table_text += table_md
                                table_count += 1

                        except Exception as e:
                            logger.warning(f"Skip table on page {page_num + 1}: {e}")

                logger.info(f"🎯 Extracted {table_count} meaningful tables")

        except Exception as e:
            logger.error(f"❌ Table extraction failed: {e}")

        return table_text

    def process_pdf_ultrafast(self, file_path: str) -> List[DocumentChunk]:
        """Improved PDF processing with better content extraction"""
        logger.info(f"🚀 Processing PDF: {os.path.basename(file_path)}")
        start_time = time.time()
        chunks = []

        try:
            # Extract main text content first
            doc = fitz.open(file_path)
            full_text = ""
            total_pages = len(doc)
            
            logger.info("📄 Extracting main content...")
            pages_to_process = list(range(min(total_pages, 20)))  # Process more pages
            
            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    
                    # Get text in reading order
                    text_dict = page.get_text("dict")
                    page_text = ""
                    
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                line_text = ""
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    if text:
                                        line_text += text + " "
                                if line_text.strip():
                                    page_text += line_text.strip() + "\n"
                    
                    # Filter out repetitive content (like ombudsman addresses)
                    lines = page_text.split('\n')
                    filtered_lines = []
                    prev_line = ""
                    repeat_count = 0
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Skip repetitive lines and ombudsman addresses
                        if 'Office of the Insurance Ombudsman' in line:
                            continue
                        if 'Lalit Bhawan' in line:
                            continue
                            
                        if line == prev_line:
                            repeat_count += 1
                            if repeat_count < 2:  # Allow max 2 repeats
                                filtered_lines.append(line)
                        else:
                            repeat_count = 0
                            filtered_lines.append(line)
                            prev_line = line
                    
                    page_text = '\n'.join(filtered_lines)
                    
                    if page_text.strip() and len(page_text) > 100:
                        full_text += f"\n\n=== Page {page_num + 1} ===\n{page_text}"
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")

            doc.close()

            # Extract tables separately
            logger.info("📊 Extracting tables...")
            table_content = self.extract_tables_fast(file_path)

            # Combine content - put main text first
            if table_content:
                full_text += f"\n\n{'='*50}\nIMPORTANT POLICY TABLES\n{'='*50}\n{table_content}"

            # Create better chunks using improved splitting
            logger.info("📦 Creating semantic chunks...")
            text_chunks = self.improved_text_split(full_text, os.path.basename(file_path))

            for idx, chunk_text in enumerate(text_chunks):
                # Add more metadata for better retrieval
                has_tables = "**TABLE" in chunk_text or "POLICY TABLES" in chunk_text
                
                # Extract key terms for better matching
                key_terms = []
                for term in ['premium', 'coverage', 'benefit', 'waiting period', 'exclusion', 'claim', 
                           'deductible', 'copay', 'hospital', 'treatment', 'policy', 'insured']:
                    if term.lower() in chunk_text.lower():
                        key_terms.append(term)

                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "chunk_index": idx,
                        "document_type": "pdf_improved",
                        "has_tables": has_tables,
                        "total_pages": total_pages,
                        "key_terms": key_terms,
                        "content_type": "table" if has_tables else "text",
                        "chunk_length": len(chunk_text)
                    },
                    chunk_id=str(uuid.uuid4())
                ))

            elapsed = time.time() - start_time
            logger.info(f"✅ Processing complete in {elapsed:.2f}s: {len(chunks)} quality chunks")
            return chunks

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return self._emergency_fallback(file_path)

    def _emergency_fallback(self, file_path: str) -> List[DocumentChunk]:
        """Emergency fallback with better text extraction"""
        logger.info("🆘 Emergency fallback")
        try:
            doc = fitz.open(file_path)
            max_pages = min(15, len(doc))
            text_parts = []
            
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Filter out ombudsman addresses
                lines = page_text.split('\n')
                filtered_lines = [line for line in lines 
                                if 'Office of the Insurance Ombudsman' not in line
                                and 'Lalit Bhawan' not in line]
                page_text = '\n'.join(filtered_lines)
                
                if len(page_text) > 5000:
                    page_text = page_text[:5000] + f"\n[Page {page_num + 1} truncated]"
                
                if len(page_text.strip()) > 100:
                    text_parts.append(f"Page {page_num + 1}:\n{page_text}")

            doc.close()
            full_text = "\n\n".join(text_parts)

            chunks = []
            chunk_size = max(1000, len(full_text) // 15)
            
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i + chunk_size]
                if len(chunk_text.strip()) > 50:
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "chunk_index": len(chunks),
                            "document_type": "pdf_emergency_fallback",
                            "has_tables": False,
                            "pages_processed": max_pages
                        },
                        chunk_id=str(uuid.uuid4())
                    ))

            return chunks

        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")
            raise Exception("All processing methods failed")

# --- GROQ LLM WRAPPER ---

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
                temperature=0.1,
                max_tokens=800,
                top_p=0.9,
                stop=stop
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return "Error generating response"

# --- RAG PIPELINE ---

class ImprovedRAGPipeline:
    """Improved RAG pipeline with better retrieval"""
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
        logger.info(f"✅ RAG pipeline initialized: {collection_name}")

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents with improved filtering and indexing"""
        if not chunks:
            logger.error("❌ No chunks provided!")
            return

        logger.info(f"📚 Adding {len(chunks)} chunks to vectorstore...")
        
        # Filter out low-quality chunks
        quality_chunks = []
        for chunk in chunks:
            content = chunk['content']
            
            # Skip chunks that are mostly addresses or repetitive content
            if (len(content) > 150 and 
                'Office of the Insurance Ombudsman' not in content[:300] and
                'Lalit Bhawan' not in content[:300] and
                not re.search(r'(\w+\s+){1,3}\|\s*(\w+\s+){1,3}\|', content[:200])):  # Skip table rows
                quality_chunks.append(chunk)

        logger.info(f"📚 Filtered to {len(quality_chunks)} quality chunks")
        
        # Debug first few chunks
        for i, chunk in enumerate(quality_chunks[:3]):
            logger.info(f"Sample chunk {i}: {chunk['content'][:200]}...")

        langchain_docs = [
            LangChainDocument(page_content=chunk['content'], metadata=chunk['metadata'])
            for chunk in quality_chunks
        ]

        self.vectorstore.add_documents(langchain_docs)
        logger.info(f"✅ Added {len(langchain_docs)} documents to vectorstore")

        # Create retriever with better search parameters
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 8,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        # Create improved prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert insurance policy analyst. Use the following policy document context to answer the question accurately.

Context from policy document:
{context}

Question: {question}

Instructions:
- Provide a clear, specific answer based on the policy document context above
- Include exact details like numbers, percentages, time periods, amounts, etc.
- Quote specific policy text when available
- If you find partial information, provide what you can determine from the context
- Be precise about policy terms, conditions, and benefits
- Only say "Information not available" if absolutely no relevant information exists

Answer:"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.groq_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

        logger.info("✅ QA Chain ready")

    async def answer_question(self, question: str) -> str:
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please add documents first."

        logger.info(f"🤔 Answering: {question}")

        try:
            # Test retrieval first
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.7}
            )
            retrieved_docs = retriever.get_relevant_documents(question)
            
            logger.info(f"🔍 Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs[:3]):  # Show first 3
                logger.info(f"Retrieved {i}: {doc.page_content[:150]}...")

            # Run QA chain
            result = await asyncio.to_thread(self.qa_chain, {"query": question})
            answer = result.get("result", "Failed to get an answer.")
            
            logger.info(f"✅ Answer: {answer[:200]}...")
            return answer

        except Exception as e:
            logger.error(f"❌ Error during QA: {e}")
            return "An error occurred while processing the question."

# --- API KEY MANAGER ---

class GroqAPIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [key.strip() for key in api_keys if key.strip()]
        self.key_usage_count = defaultdict(int)
        self.key_last_used = defaultdict(float)
        self.current_key_index = 0
        self.max_requests_per_key = 45
        logger.info(f"🔑 API Key Manager: {len(self.api_keys)} keys")

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

# --- CONFIGURATION ---

GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(',')
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "/tmp/chroma_db_storage"
UPLOAD_DIR = "/tmp/docs"

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Initializing services...")
        
        app.state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        app.state.api_key_manager = GroqAPIKeyManager(GROQ_API_KEYS)
        first_key = app.state.api_key_manager.get_next_api_key()
        app.state.groq_client = groq.Groq(api_key=first_key)
        app.state.groq_llm = GroqLLM(groq_client=app.state.groq_client, api_key_manager=app.state.api_key_manager)
        
        app.state.parsing_service = EmbeddedDocumentParser()
        
        logger.info("✅ All services initialized!")
    except Exception as e:
        logger.error(f"💥 FATAL: {e}")
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

# --- MAIN ENDPOINT ---

@app.post("/hackrx/run", response_model=SubmissionResponse)
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    logger.info(f"🎯 Processing {len(submission_request.documents)} documents, {len(submission_request.questions)} questions")
    
    parsing_service = request.app.state.parsing_service
    session_collection_name = f"hackrx_session_{uuid.uuid4().hex}"
    rag_pipeline = ImprovedRAGPipeline(collection_name=session_collection_name, request=request)
    
    all_chunks = []

    # Process documents
    async with httpx.AsyncClient(timeout=120.0) as client:
        for doc_idx, doc_url in enumerate(submission_request.documents):
            try:
                logger.info(f"📥 Downloading document {doc_idx + 1}: {doc_url}")
                response = await client.get(doc_url, follow_redirects=True)
                response.raise_for_status()

                file_name = os.path.basename(doc_url.split('?')[0]) or f"document_{doc_idx}.pdf"
                temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file_name}")
                os.makedirs(UPLOAD_DIR, exist_ok=True)

                with open(temp_file_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"📄 Processing {file_name}...")
                chunks = parsing_service.process_pdf_ultrafast(temp_file_path)
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                all_chunks.extend(chunk_dicts)

                os.remove(temp_file_path)
                logger.info(f"✅ Processed {len(chunks)} chunks from {file_name}")

            except Exception as e:
                logger.error(f"❌ Failed to process document: {e}")
                continue

    logger.info(f"📊 Total chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.error("❌ No chunks processed!")
        failed_answers = [Answer(question=q, answer="No valid documents could be processed.") for q in submission_request.questions]
        return SubmissionResponse(answers=failed_answers)

    # Add to RAG pipeline
    rag_pipeline.add_documents(all_chunks)

    # Answer questions
    logger.info(f"❓ Answering questions...")
    tasks = [rag_pipeline.answer_question(q) for q in submission_request.questions]
    results = await asyncio.gather(*tasks)
    
    answers = [Answer(question=q, answer=ans) for q, ans in zip(submission_request.questions, results)]
    
    logger.info("🎉 All questions processed!")
    return SubmissionResponse(answers=answers)

@app.get("/")
def read_root():
    return {"message": "Standalone Fixed RAG System", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Debug endpoint to check content quality
@app.post("/debug/check-content")
async def check_content(request: Request, submission_request: SubmissionRequest = Body(...)):
    """Debug endpoint to check extracted content"""
    parsing_service = request.app.state.parsing_service
    
    # Process first document
    async with httpx.AsyncClient(timeout=120.0) as client:
        doc_url = submission_request.documents[0]
        response = await client.get(doc_url, follow_redirects=True)
        response.raise_for_status()
        
        file_name = f"debug_{uuid.uuid4()}.pdf"
        temp_file_path = os.path.join(UPLOAD_DIR, file_name)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        
        chunks = parsing_service.process_pdf_ultrafast(temp_file_path)
        
        # Return sample of different chunk types
        text_chunks = [c for c in chunks if not c.metadata.get('has_tables', False)]
        table_chunks = [c for c in chunks if c.metadata.get('has_tables', False)]
        
        os.remove(temp_file_path)
        
        return {
            "total_chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "table_chunks": len(table_chunks),
            "sample_text_chunks": [
                {
                    "content": chunk.content[:500] + "...",
                    "metadata": chunk.metadata
                }
                for chunk in text_chunks[:3]
            ],
            "sample_table_chunks": [
                {
                    "content": chunk.content[:500] + "...",
                    "metadata": chunk.metadata
                }
                for chunk in table_chunks[:2]
            ]
        }
