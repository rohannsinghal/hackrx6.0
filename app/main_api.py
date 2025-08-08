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
import numpy as np

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic RAG System", version="2.0.0")

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

class SemanticDocumentParser:
    """Advanced semantic document parsing with intelligent chunking"""
    def __init__(self):
        self.chunk_size = 1200  # Optimized for semantic coherence
        self.chunk_overlap = 200  # Increased overlap for better context preservation
        self.max_chunks = 300   # Increased for better coverage
        self.table_row_limit = 25
        logger.info("SemanticDocumentParser initialized")

    def semantic_text_split(self, text: str, source: str) -> List[str]:
        """Advanced semantic text splitting using multiple strategies"""
        if not text or len(text) < 100:
            return [text] if text else []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # Strategy 1: Split by semantic sections
        section_patterns = [
            r'\n\s*(?:\d+\.)+\s*[A-Z][^.\n]*[.:]',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]{8,}[:\n]',  # ALL CAPS HEADINGS
            r'\n\s*(?:SECTION|PART|CHAPTER)\s+\w+',  # Section markers
            r'\n\s*(?:EXCLUSIONS?|INCLUSIONS?|BENEFITS?|COVERAGE|DEFINITIONS?)',  # Key sections
            r'\n\s*(?:WAITING\s+PERIOD|GRACE\s+PERIOD)',  # Important terms
            r'\n\s*(?:CLAIMS?|PREMIUM|DEDUCTIBLE)',  # Key insurance terms
        ]
        
        # Find all section boundaries
        boundaries = [0]
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                boundaries.append(match.start())
        
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))
        
        # Create semantic chunks
        for i in range(len(boundaries) - 1):
            section_start = boundaries[i]
            section_end = boundaries[i + 1]
            section_text = text[section_start:section_end].strip()
            
            if len(section_text) <= self.chunk_size:
                if section_text and len(section_text) > 50:
                    chunks.append(section_text)
            else:
                # Split large sections using sliding window with semantic boundaries
                sub_chunks = self._split_large_section(section_text)
                chunks.extend(sub_chunks)
        
        # If no semantic boundaries found, use intelligent sliding window
        if len(chunks) == 0:
            chunks = self._intelligent_sliding_window(text)
        
        logger.info(f"Split {source} into {len(chunks)} semantic chunks")
        return chunks[:self.max_chunks]  # Limit total chunks

    def _split_large_section(self, text: str) -> List[str]:
        """Split large sections using semantic boundaries"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            if end < len(text):
                # Find best break point using multiple criteria
                search_start = max(start + self.chunk_size // 2, end - 400)
                
                # Look for sentence endings first
                sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])]
                if sentence_ends:
                    end = search_start + sentence_ends[-1]
                else:
                    # Look for paragraph breaks
                    para_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text[search_start:end])]
                    if para_breaks:
                        end = search_start + para_breaks[-1]
                    else:
                        # Look for any line break
                        line_breaks = [m.start() for m in re.finditer(r'\n', text[search_start:end])]
                        if line_breaks:
                            end = search_start + line_breaks[-1]
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks

    def _intelligent_sliding_window(self, text: str) -> List[str]:
        """Fallback intelligent sliding window approach"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def extract_semantic_tables(self, file_path: str) -> str:
        """Extract tables with semantic understanding"""
        table_text = ""
        table_count = 0
        max_tables = 20

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = list(range(min(total_pages, 25)))  # Process more pages

                logger.info(f"📊 Processing {len(pages_to_process)} pages for semantic table extraction")

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
                            if table_data and len(table_data) >= 2:
                                
                                # Semantic filtering - look for insurance-related content
                                table_text_content = str(table_data).lower()
                                insurance_terms = [
                                    'premium', 'coverage', 'benefit', 'waiting', 'exclusion',
                                    'claim', 'deductible', 'policy', 'insured', 'sum', 'limit',
                                    'medical', 'hospital', 'treatment', 'age', 'amount'
                                ]
                                
                                # Skip if doesn't contain insurance terms
                                if not any(term in table_text_content for term in insurance_terms):
                                    continue
                                
                                # Skip repetitive administrative content
                                if any(admin_term in table_text_content for admin_term in 
                                      ['ombudsman', 'lalit bhawan', 'office address']):
                                    continue
                                
                                # Process meaningful table
                                limited_data = table_data[:min(30, len(table_data))]
                                
                                # Create semantic table representation
                                table_md = f"\n**POLICY TABLE {table_count + 1} (Page {page_num + 1})**\n"
                                
                                # Add table with proper formatting
                                if len(limited_data) > 0 and limited_data[0]:
                                    header = " | ".join(str(cell or "").strip()[:50] for cell in limited_data[0])
                                    table_md += f"| {header} |\n"
                                    table_md += f"| {' | '.join(['---'] * len(limited_data[0]))} |\n"
                                    
                                    for row in limited_data[1:]:
                                        if row:
                                            padded_row = list(row) + [None] * (len(limited_data[0]) - len(row))
                                            row_str = " | ".join(str(cell or "").strip()[:50] for cell in padded_row)
                                            table_md += f"| {row_str} |\n"
                                
                                table_md += "\n"
                                table_text += table_md
                                table_count += 1

                        except Exception as e:
                            logger.warning(f"Skip table on page {page_num + 1}: {e}")

                logger.info(f"🎯 Extracted {table_count} semantic tables")

        except Exception as e:
            logger.error(f"❌ Semantic table extraction failed: {e}")

        return table_text

    def process_pdf_semantically(self, file_path: str) -> List[DocumentChunk]:
        """Semantic PDF processing for any insurance document"""
        logger.info(f"🚀 Processing PDF semantically: {os.path.basename(file_path)}")
        start_time = time.time()
        chunks = []

        try:
            # Extract text with better preservation
            doc = fitz.open(file_path)
            full_text = ""
            total_pages = len(doc)
            
            logger.info("📄 Extracting content with semantic awareness...")
            pages_to_process = list(range(min(total_pages, 30)))  # Process more pages
            
            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    
                    # Get text blocks in reading order
                    blocks = page.get_text("dict")["blocks"]
                    page_text = ""
                    
                    for block in blocks:
                        if "lines" in block:
                            block_text = ""
                            for line in block["lines"]:
                                line_text = ""
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    if text:
                                        line_text += text + " "
                                if line_text.strip():
                                    block_text += line_text.strip() + "\n"
                            
                            # Keep blocks that contain meaningful content
                            if len(block_text.strip()) > 20:
                                page_text += block_text + "\n"
                    
                    # Semantic filtering - remove noise while preserving content
                    lines = page_text.split('\n')
                    filtered_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Skip obvious noise but keep everything else
                        skip_patterns = [
                            r'^office of the insurance ombudsman',
                            r'^lalit bhawan',
                            r'^\d+\s*$',  # Page numbers only
                            r'^page \d+ of \d+$',
                        ]
                        
                        should_skip = any(re.match(pattern, line.lower()) for pattern in skip_patterns)
                        if not should_skip:
                            filtered_lines.append(line)
                    
                    clean_page_text = '\n'.join(filtered_lines)
                    
                    if clean_page_text.strip() and len(clean_page_text) > 50:
                        full_text += f"\n\n=== Page {page_num + 1} ===\n{clean_page_text}"
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")

            doc.close()

            # Extract semantic tables
            logger.info("📊 Extracting semantic tables...")
            table_content = self.extract_semantic_tables(file_path)

            # Combine content strategically
            if table_content:
                full_text += f"\n\n{'='*50}\nIMPORTANT POLICY DATA\n{'='*50}\n{table_content}"

            # Create semantic chunks
            logger.info("📦 Creating semantic chunks...")
            text_chunks = self.semantic_text_split(full_text, os.path.basename(file_path))

            # Create chunks with rich metadata for semantic search
            for idx, chunk_text in enumerate(text_chunks):
                # Analyze chunk content semantically
                chunk_lower = chunk_text.lower()
                
                # Detect content type
                content_indicators = {
                    'definitions': ['means', 'definition', 'shall mean', 'refers to'],
                    'coverage': ['coverage', 'covered', 'benefit', 'sum insured'],
                    'exclusions': ['exclusion', 'excluded', 'not covered', 'shall not'],
                    'waiting_periods': ['waiting period', 'wait', 'months', 'years'],
                    'claims': ['claim', 'settlement', 'reimbursement', 'cashless'],
                    'premium': ['premium', 'payment', 'due', 'grace period'],
                    'medical': ['hospital', 'medical', 'treatment', 'doctor', 'physician'],
                    'tables': ['**table', 'policy table', '|']
                }
                
                detected_types = []
                for content_type, indicators in content_indicators.items():
                    if any(indicator in chunk_lower for indicator in indicators):
                        detected_types.append(content_type)
                
                # Calculate content richness
                sentences = re.split(r'[.!?]+', chunk_text)
                avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "chunk_index": idx,
                        "document_type": "semantic_pdf",
                        "content_types": ", ".join(detected_types),
                        "total_pages": total_pages,
                        "chunk_length": len(chunk_text),
                        "sentence_count": len([s for s in sentences if s.strip()]),
                        "avg_sentence_length": round(avg_sentence_length, 1),
                        "has_tables": "tables" in detected_types,
                        "semantic_score": len(detected_types) * avg_sentence_length
                    },
                    chunk_id=str(uuid.uuid4())
                ))

            elapsed = time.time() - start_time
            logger.info(f"✅ Semantic processing complete in {elapsed:.2f}s: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"❌ Semantic processing failed: {e}")
            return self._emergency_semantic_fallback(file_path)

    def _emergency_semantic_fallback(self, file_path: str) -> List[DocumentChunk]:
        """Emergency semantic fallback"""
        logger.info("🆘 Emergency semantic fallback")
        try:
            doc = fitz.open(file_path)
            text_parts = []
            
            for page_num in range(min(20, len(doc))):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Basic semantic filtering
                if len(page_text.strip()) > 100:
                    # Remove obvious administrative noise
                    lines = [line for line in page_text.split('\n') 
                           if not any(noise in line.lower() for noise in 
                                    ['office of the insurance ombudsman', 'lalit bhawan'])]
                    clean_text = '\n'.join(lines)
                    
                    if len(clean_text.strip()) > 100:
                        text_parts.append(clean_text)

            doc.close()
            full_text = "\n\n".join(text_parts)

            # Create semantic chunks
            chunk_size = max(800, len(full_text) // 20)
            chunks = []
            
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i + chunk_size]
                if len(chunk_text.strip()) > 100:
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "chunk_index": len(chunks),
                            "document_type": "emergency_semantic",
                            "chunk_length": len(chunk_text)
                        },
                        chunk_id=str(uuid.uuid4())
                    ))

            return chunks

        except Exception as e:
            logger.error(f"Emergency semantic fallback failed: {e}")
            raise Exception("All semantic processing methods failed")

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
                max_tokens=1000,  # Increased for more detailed answers
                top_p=0.9,
                stop=stop
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return "Error generating response"

# --- ADVANCED SEMANTIC RAG PIPELINE ---

class AdvancedSemanticRAGPipeline:
    """Advanced semantic RAG pipeline with intelligent retrieval"""
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
        logger.info(f"✅ Advanced semantic RAG pipeline initialized: {collection_name}")

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents with advanced semantic filtering"""
        if not chunks:
            logger.error("❌ No chunks provided!")
            return

        logger.info(f"📚 Adding {len(chunks)} chunks with semantic processing...")
        
        # Advanced semantic filtering
        quality_chunks = []
        for chunk in chunks:
            content = chunk['content']
            
            # Semantic quality assessment
            quality_score = self._assess_chunk_quality(content)
            
            if quality_score > 0.3:  # Threshold for semantic quality
                quality_chunks.append(chunk)

        # Sort by semantic relevance score if available
        quality_chunks.sort(key=lambda x: x['metadata'].get('semantic_score', 0), reverse=True)
        
        # Take top chunks if too many
        if len(quality_chunks) > 150:
            quality_chunks = quality_chunks[:150]

        logger.info(f"📚 Filtered to {len(quality_chunks)} high-quality semantic chunks")
        
        # Debug semantic chunks
        for i, chunk in enumerate(quality_chunks[:3]):
            logger.info(f"Semantic chunk {i}: {chunk['content'][:200]}...")

        langchain_docs = [
            LangChainDocument(page_content=chunk['content'], metadata=chunk['metadata'])
            for chunk in quality_chunks
        ]

        self.vectorstore.add_documents(langchain_docs)
        logger.info(f"✅ Added {len(langchain_docs)} semantic documents to vectorstore")

        # Create advanced retriever with hybrid search
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 15,  # Increased for better coverage
                "fetch_k": 30,  # Increased search space
                "lambda_mult": 0.4  # Balance between relevance and diversity
            }
        )

        # Advanced prompt template for semantic understanding
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert insurance policy analyst with deep semantic understanding. Analyze the following policy document context to answer the question with precision and accuracy.

POLICY DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANALYSIS INSTRUCTIONS:
- Carefully read and understand the semantic meaning of the policy context above
- Extract specific facts, numbers, percentages, time periods, conditions, and requirements
- Look for explicit policy clauses, definitions, and procedural information
- If multiple pieces of information relate to the question, synthesize them comprehensively
- Quote exact policy language when providing specific details
- Distinguish between what is explicitly stated vs. what can be reasonably inferred
- If information is partial or unclear, state what is available and note limitations
- Be precise about conditions, exceptions, and qualifying circumstances
- Only state "Information not available" if absolutely no relevant information exists

SEMANTIC ANALYSIS:
- Consider the broader context and meaning beyond just keyword matching
- Understand the logical relationships between different policy provisions
- Recognize implicit connections between related policy sections

ANSWER FORMAT:
Provide a clear, specific, and complete answer based on your semantic analysis of the policy document.

ANSWER:"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.groq_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

        logger.info("✅ Advanced semantic QA Chain ready")

    def _assess_chunk_quality(self, content: str) -> float:
        """Assess semantic quality of a chunk"""
        content_lower = content.lower()
        
        # Base quality factors
        length_score = min(len(content) / 1000, 1.0)  # Normalize to 1000 chars
        
        # Insurance domain relevance
        insurance_terms = [
            'policy', 'coverage', 'premium', 'benefit', 'claim', 'insured',
            'exclusion', 'waiting period', 'deductible', 'hospital', 'medical',
            'treatment', 'sum insured', 'reimbursement', 'cashless'
        ]
        term_score = sum(1 for term in insurance_terms if term in content_lower) / len(insurance_terms)
        
        # Structural quality
        sentences = re.split(r'[.!?]+', content)
        sentence_score = min(len([s for s in sentences if len(s.strip()) > 10]) / 5, 1.0)
        
        # Information density
        number_score = min(len(re.findall(r'\d+', content)) / 10, 1.0)
        
        # Administrative noise penalty
        noise_penalty = 0
        noise_terms = ['ombudsman', 'lalit bhawan', 'office address']
        if any(term in content_lower for term in noise_terms):
            noise_penalty = 0.5
        
        quality_score = (length_score * 0.2 + term_score * 0.4 + 
                        sentence_score * 0.2 + number_score * 0.2) - noise_penalty
        
        return max(0, quality_score)

    async def answer_question(self, question: str) -> str:
        if not self.qa_chain:
            return "Error: Semantic QA chain not initialized. Please add documents first."

        logger.info(f"🤔 Semantic analysis for: {question}")

        try:
            # Enhanced semantic retrieval
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 15,
                    "fetch_k": 30,
                    "lambda_mult": 0.4
                }
            )
            
            # Get semantically relevant documents
            retrieved_docs = retriever.get_relevant_documents(question)
            
            logger.info(f"🔍 Retrieved {len(retrieved_docs)} semantic documents")
            for i, doc in enumerate(retrieved_docs[:3]):
                logger.info(f"Semantic doc {i}: {doc.page_content[:150]}...")

            # Run advanced semantic QA
            result = await asyncio.to_thread(self.qa_chain, {"query": question})
            answer = result.get("result", "Failed to generate semantic answer.")
            
            logger.info(f"✅ Semantic answer: {answer[:200]}...")
            return answer

        except Exception as e:
            logger.error(f"❌ Error during semantic QA: {e}")
            return "An error occurred while processing the semantic question."

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
        logger.info("🚀 Initializing semantic services...")
        
        app.state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        app.state.api_key_manager = GroqAPIKeyManager(GROQ_API_KEYS)
        first_key = app.state.api_key_manager.get_next_api_key()
        app.state.groq_client = groq.Groq(api_key=first_key)
        app.state.groq_llm = GroqLLM(groq_client=app.state.groq_client, api_key_manager=app.state.api_key_manager)
        
        app.state.parsing_service = SemanticDocumentParser()
        
        logger.info("✅ All semantic services initialized!")
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
    session_collection_name = f"semantic_session_{uuid.uuid4().hex}"
    rag_pipeline = AdvancedSemanticRAGPipeline(collection_name=session_collection_name, request=request)
    
    all_chunks = []

    # Process documents with semantic understanding
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

                logger.info(f"📄 Processing {file_name} with semantic analysis...")
                chunks = parsing_service.process_pdf_semantically(temp_file_path)
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                all_chunks.extend(chunk_dicts)

                os.remove(temp_file_path)
                logger.info(f"✅ Processed {len(chunks)} semantic chunks from {file_name}")

            except Exception as e:
                logger.error(f"❌ Failed to process document: {e}")
                continue

    logger.info(f"📊 Total semantic chunks: {len(all_chunks)}")

    if not all_chunks:
        logger.error("❌ No semantic chunks processed!")
        failed_answers = [Answer(question=q, answer="No valid documents could be processed.") for q in submission_request.questions]
        return SubmissionResponse(answers=failed_answers)

    # Add to semantic RAG pipeline
    rag_pipeline.add_documents(all_chunks)

    # Answer questions with semantic understanding
    logger.info(f"❓ Answering questions with semantic analysis...")
    tasks = [rag_pipeline.answer_question(q) for q in submission_request.questions]
    results = await asyncio.gather(*tasks)
    
    answers = [Answer(question=q, answer=ans) for q, ans in zip(submission_request.questions, results)]
    
    logger.info("🎉 All semantic questions processed!")
    return SubmissionResponse(answers=answers)

@app.get("/")
def read_root():
    return {"message": "Advanced Semantic RAG System", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/debug/semantic-analysis")
async def debug_semantic_analysis(request: Request, submission_request: SubmissionRequest = Body(...)):
    """Debug endpoint for semantic analysis"""
    parsing_service = request.app.state.parsing_service
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        doc_url = submission_request.documents[0]
        response = await client.get(doc_url, follow_redirects=True)
        response.raise_for_status()
        
        file_name = f"semantic_debug_{uuid.uuid4()}.pdf"
        temp_file_path = os.path.join(UPLOAD_DIR, file_name)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        
        chunks = parsing_service.process_pdf_semantically(temp_file_path)
        
        # Analyze semantic distribution
        content_types = {}
        for chunk in chunks:
            types = chunk.metadata.get('content_types', '').split(', ')
            for ctype in types:
                if ctype:
                    content_types[ctype] = content_types.get(ctype, 0) + 1
        
        os.remove(temp_file_path)
        
        return {
            "total_chunks": len(chunks),
            "content_type_distribution": content_types,
            "average_chunk_length": sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0,
            "semantic_samples": [
                {
                    "content": chunk.content[:400] + "...",
                    "metadata": chunk.metadata,
                    "content_types": chunk.metadata.get('content_types', '')
                }
                for chunk in chunks[:5]
            ]
        }
