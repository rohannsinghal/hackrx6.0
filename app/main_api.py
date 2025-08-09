# -*- coding: utf-8 -*-
"""
Ultimate Hackathon Winning RAG System - Multi-format, Multi-LLM, Enhanced Semantic
Version: 4.0.0 - Competition Ready
"""

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
import functools

# FastAPI and core dependencies
from fastapi import FastAPI, Body, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document as LangChainDocument

# Multi-format document processing
import fitz  # PyMuPDF
import pdfplumber
import docx  # python-docx
import openpyxl
import csv
import zipfile
import email
from email.policy import default
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Semantic enhancements
from sentence_transformers import SentenceTransformer, CrossEncoder
import cachetools

# LLM providers
import groq
import openai
import google.generativeai as genai

from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ultimate Hackathon Winning RAG System", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ANTI-JAILBREAK SECURITY SYSTEM ---
class SecurityGuard:
    def __init__(self):
        self.jailbreak_patterns = [
            r'ignore.*previous.*instructions',
            r'act.*as.*different.*character',
            r'generate.*code.*(?:javascript|python|html)',
            r'write.*program',
            r'roleplay.*as',
            r'pretend.*you.*are',
            r'system.*prompt',
            r'override.*settings',
            r'bypass.*restrictions',
            r'admin.*mode',
            r'developer.*mode',
            r'tell.*me.*about.*yourself',
            r'what.*are.*you',
            r'who.*created.*you'
        ]
    
    def detect_jailbreak(self, text: str) -> bool:
        """Detect jailbreak attempts"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.jailbreak_patterns)
    
    def sanitize_response(self, question: str, answer: str) -> str:
        """Sanitize responses against jailbreaks"""
        if self.detect_jailbreak(question):
            return "I can only provide information based on the document content provided. Please ask questions about the document."
        
        # Remove any potential code or script tags
        answer = re.sub(r'<script.*?</script>', '', answer, flags=re.DOTALL | re.IGNORECASE)
        answer = re.sub(r'<.*?>', '', answer)  # Remove HTML tags
        
        return answer

# --- ENHANCED SEMANTIC PROCESSOR (FREE VERSION) ---
class AdvancedSemanticProcessor:
    def __init__(self):
        # Use FREE models for maximum performance
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # FREE
        self.cache = cachetools.TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        
    async def enhance_query_semantically(self, question: str, domain: str = "insurance") -> str:
        """Enhanced semantic query processing"""
        
        # Cache check
        cache_key = hashlib.md5(question.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Insurance domain expansion
        enhanced_query = self._expand_with_domain_knowledge(question, domain)
        
        # Handle "half questions" (R4 requirement)
        enhanced_query = self._handle_incomplete_questions(enhanced_query)
        
        # Cache result
        self.cache[cache_key] = enhanced_query
        return enhanced_query
    
    def _expand_with_domain_knowledge(self, query: str, domain: str) -> str:
        """Expand query with insurance domain knowledge"""
        insurance_expansions = {
            'grace period': ['payment deadline', 'premium due date', 'payment window'],
            'waiting period': ['exclusion time', 'coverage delay', 'qualification period'],
            'pre-existing': ['prior medical condition', 'existing disease', 'medical history'],
            'coverage': ['policy benefits', 'insurance protection', 'covered services'],
            'exclusion': ['policy limitations', 'restrictions', 'non-covered items'],
            'deductible': ['excess amount', 'out-of-pocket cost'],
            'premium': ['insurance cost', 'policy payment', 'monthly fee'],
            'claim': ['insurance claim', 'benefit request', 'reimbursement'],
            'ayush': ['alternative medicine', 'traditional therapy', 'holistic treatment'],
            'hospital': ['healthcare facility', 'medical center', 'treatment center']
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for key_term, synonyms in insurance_expansions.items():
            if key_term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Limit for performance
        
        if expanded_terms:
            enhanced_query = f"{query}. Also consider: {', '.join(expanded_terms[:4])}"
        else:
            enhanced_query = query
            
        return enhanced_query
    
    def _handle_incomplete_questions(self, query: str) -> str:
        """Handle R4's 'half questions' requirement"""
        incomplete_patterns = [
            r'^(what|how|when|where|why)\s*\?*$',
            r'^(yes|no)\s*\?*$',
            r'^\w{1,3}\s*\?*$',  # Very short questions
            r'^(this|that|it)\s*',
        ]
        
        query_lower = query.lower()
        is_incomplete = any(re.search(pattern, query_lower) for pattern in incomplete_patterns)
        
        if is_incomplete:
            # Add context for common incomplete questions
            if len(query.split()) <= 2:
                enhanced_query = f"{query}. Please provide information about insurance policy terms, coverage, exclusions, waiting periods, or benefits."
            else:
                enhanced_query = query
        else:
            enhanced_query = query
            
        return enhanced_query
    
    async def semantic_rerank(self, question: str, documents: List, k: int = 8) -> List:
        """Advanced semantic reranking using CrossEncoder"""
        if not documents or len(documents) <= k:
            return documents
        
        # Prepare pairs for reranking
        pairs = [[question, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = self.reranker.predict(pairs)
        
        # Combine and sort
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:k]]

# --- MULTI-LLM MANAGER (FREE + PAID OPTIONS) ---
class MultiLLMManager:
    def __init__(self):
        # Initialize multiple LLM providers with fallback
        self.providers = ['groq']  # Start with Groq as primary
        
        self.groq_keys = cycle([k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(',') if k.strip()])
        
        # Optional paid providers (if keys available)
        openai_keys = [k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(',') if k.strip()]
        gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(',') if k.strip()]
        
        if openai_keys:
            self.providers.append('openai')
            self.openai_keys = cycle(openai_keys)
            
        if gemini_keys:
            self.providers.append('gemini') 
            self.gemini_keys = cycle(gemini_keys)
        
        self.current_provider_index = 0
        logger.info(f"🔑 Multi-LLM Manager initialized with {len(self.providers)} providers")
    
    async def get_response(self, prompt: str, max_tokens: int = 900) -> str:
        """Get response with automatic fallback between providers"""
        for attempt in range(len(self.providers)):
            try:
                provider = self.providers[self.current_provider_index]
                
                if provider == 'groq':
                    return await self._groq_response(prompt, max_tokens)
                elif provider == 'openai':
                    return await self._openai_response(prompt, max_tokens)
                elif provider == 'gemini':
                    return await self._gemini_response(prompt, max_tokens)
                    
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
    
    async def _openai_response(self, prompt: str, max_tokens: int) -> str:
        key = next(self.openai_keys)
        openai.api_key = key
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    
    async def _gemini_response(self, prompt: str, max_tokens: int) -> str:
        key = next(self.gemini_keys)
        genai.configure(api_key=key)
        
        model = genai.GenerativeModel('gemini-pro')
        response = await model.generate_content_async(prompt)
        return response.text.strip()

# --- UNIVERSAL DOCUMENT PROCESSOR ---
class UniversalDocumentProcessor:
    def __init__(self):
        self.chunk_size = 1200
        self.chunk_overlap = 200
        self.max_chunks = 250
        self.max_pages = 25
        
        # Smart caching system
        self.cache = cachetools.TTLCache(maxsize=100, ttl=3600)
        
        # Supported formats
        self.processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.doc': self.process_doc,
            '.xlsx': self.process_excel,
            '.xls': self.process_excel,
            '.csv': self.process_csv,
            '.txt': self.process_text,
            '.html': self.process_html,
            '.xml': self.process_xml,
            '.eml': self.process_email,
            '.zip': self.process_archive,
            '.json': self.process_json
        }
        
        logger.info("🚀 Universal Document Processor initialized")
    
    def get_file_hash(self, content: bytes) -> str:
        """Generate hash for caching"""
        return hashlib.md5(content).hexdigest()
    
    async def process_document(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process any document format with caching"""
        file_hash = self.get_file_hash(content)
        
        # Check cache first
        if file_hash in self.cache:
            logger.info(f"📦 Cache hit for {os.path.basename(file_path)}")
            return self.cache[file_hash]
        
        # Detect file type
        file_ext = Path(file_path).suffix.lower()
        if not file_ext:
            file_ext = self._detect_file_type(content)
        
        # Process based on file type
        processor = self.processors.get(file_ext, self.process_text)
        
        try:
            chunks = await processor(file_path, content)
            
            # Cache the result
            self.cache[file_hash] = chunks
            
            logger.info(f"✅ Processed {os.path.basename(file_path)}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Processing failed for {file_path}: {e}")
            return self._emergency_text_extraction(content, file_path)
    
    def _detect_file_type(self, content: bytes) -> str:
        """Detect file type from content"""
        if content.startswith(b'%PDF'):
            return '.pdf'
        elif content.startswith(b'PK'):
            return '.docx' if b'word/' in content[:1000] else '.zip'
        elif content.startswith(b'<html') or content.startswith(b'<!DOCTYPE'):
            return '.html'
        elif content.startswith(b'<?xml'):
            return '.xml'
        else:
            return '.txt'
    
    # --- PDF PROCESSING ---
    async def process_pdf(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Enhanced PDF processing"""
        chunks = []
        temp_path = f"/tmp/{uuid.uuid4().hex}.pdf"
        
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        try:
            # Extract text with PyMuPDF
            doc = fitz.open(temp_path)
            full_text = ""
            
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    full_text += f"\n\nPage {page_num + 1}:\n{self._clean_text(text)}"
            
            doc.close()
            
            # Extract tables with pdfplumber
            table_text = await self._extract_pdf_tables(temp_path)
            if table_text:
                full_text += f"\n\n=== TABLES ===\n{table_text}"
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(full_text, file_path, "pdf")
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return chunks
    
    async def _extract_pdf_tables(self, file_path: str) -> str:
        """Extract tables from PDF"""
        table_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:12]):
                    tables = page.find_tables()
                    for i, table in enumerate(tables[:2]):
                        try:
                            table_data = table.extract()
                            if table_data and len(table_data) > 1:
                                table_md = f"\n**Table {i+1} (Page {page_num+1})**\n"
                                for row in table_data[:15]:
                                    if row:
                                        clean_row = [str(cell or "").strip()[:40] for cell in row]
                                        table_md += "| " + " | ".join(clean_row) + " |\n"
                                table_text += table_md + "\n"
                        except:
                            continue
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return table_text
    
    # --- OTHER FORMAT PROCESSORS ---
    async def process_docx(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process DOCX files"""
        temp_path = f"/tmp/{uuid.uuid4().hex}.docx"
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        try:
            doc = docx.Document(temp_path)
            full_text = ""
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text += para.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                table_text = "\n**TABLE**\n"
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    table_text += "| " + " | ".join(row_text) + " |\n"
                full_text += table_text + "\n"
            
            chunks = self._create_semantic_chunks(full_text, file_path, "docx")
            
        except Exception as e:
            logger.error(f"DOCX processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return chunks
    
    async def process_doc(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process DOC files (fallback to text extraction)"""
        return self._emergency_text_extraction(content, file_path)
    
    async def process_excel(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process Excel files"""
        temp_path = f"/tmp/{uuid.uuid4().hex}.xlsx"
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        try:
            workbook = openpyxl.load_workbook(temp_path, read_only=True)
            full_text = ""
            
            for sheet_name in workbook.sheetnames[:3]:
                sheet = workbook[sheet_name]
                full_text += f"\n**Sheet: {sheet_name}**\n"
                
                for row_num, row in enumerate(sheet.iter_rows(max_row=50, values_only=True)):
                    if row_num == 0 or any(cell for cell in row):
                        row_text = [str(cell or "").strip()[:30] for cell in row[:8]]
                        full_text += "| " + " | ".join(row_text) + " |\n"
            
            workbook.close()
            chunks = self._create_semantic_chunks(full_text, file_path, "excel")
            
        except Exception as e:
            logger.error(f"Excel processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return chunks
    
    # --- Other format processors (simplified for brevity) ---
    async def process_csv(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        try:
            text_content = content.decode('utf-8', errors='ignore')
            lines = text_content.split('\n')
            
            full_text = "**CSV DATA**\n"
            for i, line in enumerate(lines[:100]):
                if line.strip():
                    full_text += f"| {line} |\n"
            
            return self._create_semantic_chunks(full_text, file_path, "csv")
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return []
    
    async def process_text(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        try:
            text = content.decode('utf-8', errors='ignore')
            return self._create_semantic_chunks(text, file_path, "text")
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return []
    
    async def process_html(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            return self._create_semantic_chunks(text, file_path, "html")
        except Exception as e:
            logger.error(f"HTML processing error: {e}")
            return []
    
    async def process_xml(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        try:
            root = ET.fromstring(content)
            def extract_text(element, level=0):
                text = ""
                if element.text and element.text.strip():
                    text += f"{'  ' * level}{element.tag}: {element.text.strip()}\n"
                for child in element:
                    text += extract_text(child, level + 1)
                return text
            full_text = extract_text(root)
            return self._create_semantic_chunks(full_text, file_path, "xml")
        except Exception as e:
            logger.error(f"XML processing error: {e}")
            return []
    
    async def process_email(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        try:
            msg = email.message_from_bytes(content, policy=default)
            full_text = f"**EMAIL**\n"
            full_text += f"From: {msg.get('From', 'Unknown')}\n"
            full_text += f"Subject: {msg.get('Subject', 'No Subject')}\n\n"
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_content()
                        full_text += f"Content:\n{body}\n"
            else:
                body = msg.get_content()
                full_text += f"Content:\n{body}\n"
            
            return self._create_semantic_chunks(full_text, file_path, "email")
        except Exception as e:
            logger.error(f"Email processing error: {e}")
            return []
    
    async def process_archive(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        temp_path = f"/tmp/{uuid.uuid4().hex}.zip"
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        chunks = []
        try:
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(temp_path, 'r') as zip_file:
                    for file_info in zip_file.filelist[:5]:
                        try:
                            file_content = zip_file.read(file_info)
                            sub_chunks = await self.process_document(file_info.filename, file_content)
                            chunks.extend(sub_chunks[:20])  # Limit sub-chunks
                        except:
                            continue
        except Exception as e:
            logger.error(f"Archive processing error: {e}")
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return chunks
    
    async def process_json(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        try:
            data = json.loads(content.decode('utf-8'))
            full_text = json.dumps(data, indent=2, ensure_ascii=False)
            return self._create_semantic_chunks(full_text, file_path, "json")
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            return []
    
    # --- UTILITY METHODS ---
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove noise patterns
        noise_patterns = [
            r'Office of.*Insurance Ombudsman.*?\n',
            r'Lalit Bhawan.*?\n',
            r'^\d+\s*$'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _create_semantic_chunks(self, text: str, source: str, doc_type: str) -> List[Dict[str, Any]]:
        """Create semantic chunks from text"""
        text = self._clean_text(text)
        
        if not text or len(text) < 50:
            return []
        
        # Smart sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
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
        
        # Convert to structured chunks
        structured_chunks = []
        for i, chunk_text in enumerate(chunks[:self.max_chunks]):
            structured_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": os.path.basename(source),
                    "chunk_index": i,
                    "document_type": doc_type,
                    "chunk_length": len(chunk_text)
                },
                "chunk_id": str(uuid.uuid4())
            })
        
        return structured_chunks
    
    def _emergency_text_extraction(self, content: bytes, file_path: str) -> List[Dict[str, Any]]:
        """Emergency text extraction for unsupported formats"""
        try:
            text = content.decode('utf-8', errors='ignore')
            if len(text) > 50:
                return self._create_semantic_chunks(text, file_path, "unknown")
        except:
            pass
        
        return [{
            "content": "Failed to extract content from document",
            "metadata": {
                "source": os.path.basename(file_path),
                "chunk_index": 0,
                "document_type": "error",
                "error": True
            },
            "chunk_id": str(uuid.uuid4())
        }]

# --- ULTIMATE RAG PIPELINE WITH SEMANTIC ENHANCEMENTS ---
class UltimateRAGPipeline:
    def __init__(self, collection_name: str, llm_manager: MultiLLMManager):
        self.collection_name = collection_name
        self.llm_manager = llm_manager
        self.security_guard = SecurityGuard()
        self.semantic_processor = AdvancedSemanticProcessor()
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory="/tmp/chroma_ultimate"
        )
        
        logger.info(f"🚀 Ultimate RAG Pipeline initialized: {collection_name}")
    
    async def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents with advanced filtering and processing"""
        if not chunks:
            return
        
        logger.info(f"📚 Processing {len(chunks)} chunks...")
        
        # Advanced quality filtering
        quality_chunks = []
        for chunk in chunks:
            content = chunk['content']
            
            # Skip error chunks
            if chunk['metadata'].get('error'):
                continue
            
            # Quality assessment
            quality_score = 0
            
            # Length factor
            if 100 <= len(content) <= 2000:
                quality_score += 2
            elif len(content) > 50:
                quality_score += 1
            
            # Content richness
            sentences = len(re.split(r'[.!?]+', content))
            if sentences > 3:
                quality_score += 1
            
            # Numerical data (good for policies)
            numbers = len(re.findall(r'\d+', content))
            if numbers > 0:
                quality_score += 1
            
            if quality_score >= 2:
                quality_chunks.append(chunk)
        
        logger.info(f"📚 Filtered to {len(quality_chunks)} quality chunks")
        
        # Convert to LangChain documents
        documents = [
            LangChainDocument(
                page_content=chunk['content'],
                metadata=chunk['metadata']
            )
            for chunk in quality_chunks[:150]  # Limit for performance
        ]
        
        # Add to vector store
        if documents:
            self.vectorstore.add_documents(documents)
            logger.info(f"✅ Added {len(documents)} documents to vector store")
    
    async def answer_question(self, question: str) -> str:
        """Answer question with advanced semantic processing"""
        # Security check
        if self.security_guard.detect_jailbreak(question):
            return self.security_guard.sanitize_response(question, "")
        
        try:
            # Enhanced query processing
            enhanced_question = await self.semantic_processor.enhance_query_semantically(question)
            
            # Initial retrieval (get more candidates)
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 20,  # Get more candidates for reranking
                    "fetch_k": 40,
                    "lambda_mult": 0.5
                }
            )
            
            relevant_docs = retriever.get_relevant_documents(enhanced_question)
            
            if not relevant_docs:
                return "I don't have sufficient information to answer this question based on the provided documents."
            
            # Semantic reranking (GAME CHANGER)
            top_docs = await self.semantic_processor.semantic_rerank(enhanced_question, relevant_docs, k=8)
            
            # Prepare enhanced context
            context = "\n\n".join([doc.page_content for doc in top_docs])
            
            # Create advanced semantic prompt
            prompt = self._create_advanced_prompt(context, question)
            
            # Get response from multi-LLM system
            response = await self.llm_manager.get_response(prompt)
            
            # Final security check and cleaning
            response = self.security_guard.sanitize_response(question, response)
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Question processing failed: {e}")
            return "An error occurred while processing your question."
    
    def _create_advanced_prompt(self, context: str, question: str) -> str:
        """Create advanced semantic-aware prompt"""
        return f"""You are an expert insurance policy analyst with advanced semantic understanding.

CONTEXT ANALYSIS FRAMEWORK:
- Apply deep semantic understanding to connect related concepts across documents
- Recognize implicit relationships and cross-references within policy content
- Understand hierarchical information structures and conditional dependencies
- Synthesize information from multiple sources with semantic coherence

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ADVANCED REASONING APPROACH:
1. SEMANTIC COMPREHENSION: Understand the full meaning and intent behind the question
2. CONTEXTUAL MAPPING: Map question elements to semantically relevant sections
3. RELATIONSHIP INFERENCE: Identify implicit connections between policy components
4. MULTI-SOURCE SYNTHESIS: Combine information while maintaining semantic consistency
5. CONDITIONAL REASONING: Apply logical reasoning to policy exceptions and conditions

RESPONSE REQUIREMENTS:
- Provide semantically rich, contextually grounded answers
- Include specific details: numbers, percentages, timeframes, conditions
- Write in clear, professional language without excessive quotes
- Address both explicit information and reasonable semantic inferences
- Structure information hierarchically when appropriate

ANSWER:"""
    
    def _clean_response(self, response: str) -> str:
        """Enhanced response cleaning"""
        # Remove excessive quotes
        response = re.sub(r'"([^"]{1,50})"', r'\1', response)
        response = re.sub(r'"(\w+)"', r'\1', response)
        response = re.sub(r'"(Rs\.?\s*[\d,]+[/-]*)"', r'\1', response)
        response = re.sub(r'"(\d+%)"', r'\1', response)
        response = re.sub(r'"(\d+\s*(?:days?|months?|years?))"', r'\1', response)
        
        # Clean policy references
        response = re.sub(r'[Aa]s stated in the policy[:\s]*"([^"]+)"', r'As per the policy, \1', response)
        response = re.sub(r'[Aa]ccording to the policy[:\s]*"([^"]+)"', r'According to the policy, \1', response)
        response = re.sub(r'[Tt]he policy states[:\s]*"([^"]+)"', r'The policy states that \1', response)
        
        # Fix spacing and formatting
        response = re.sub(r'\s+', ' ', response)
        response = response.replace(' ,', ',')
        response = response.replace(' .', '.')
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        
        return response.strip()

# --- AUTHENTICATION ---
async def verify_bearer_token(authorization: str = Header(None)):
    """Enhanced authentication with better logging"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    
    if len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    logger.info(f"✅ Authentication successful with token: {token[:10]}...")
    return token

# --- GLOBAL INSTANCES ---
multi_llm = MultiLLMManager()
doc_processor = UniversalDocumentProcessor()

# --- API MODELS ---
class SubmissionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- MAIN HACKATHON ENDPOINT ---
@app.post("/hackrx/run", response_model=SubmissionResponse, dependencies=[Depends(verify_bearer_token)])
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    start_time = time.time()
    logger.info(f"🎯 ULTIMATE PROCESSING: {len(submission_request.documents)} docs, {len(submission_request.questions)} questions")
    
    try:
        # Create unique session
        session_id = f"ultimate_{uuid.uuid4().hex[:8]}"
        rag_pipeline = UltimateRAGPipeline(session_id, multi_llm)
        
        # Process all documents concurrently with optimized performance
        all_chunks = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create semaphore to limit concurrent downloads
            semaphore = asyncio.Semaphore(3)
            
            async def process_single_document(doc_idx: int, doc_url: str):
                async with semaphore:
                    try:
                        logger.info(f"📥 Downloading document {doc_idx + 1}")
                        response = await client.get(doc_url, follow_redirects=True)
                        response.raise_for_status()
                        
                        # Get filename from URL or generate one
                        filename = os.path.basename(doc_url.split('?')[0]) or f"document_{doc_idx}"
                        
                        # Process document with caching
                        chunks = await doc_processor.process_document(filename, response.content)
                        
                        logger.info(f"✅ Document {doc_idx + 1}: {len(chunks)} chunks")
                        return chunks
                        
                    except Exception as e:
                        logger.error(f"❌ Document {doc_idx + 1} failed: {e}")
                        return []
            
            # Process all documents concurrently
            tasks = [
                process_single_document(i, url) 
                for i, url in enumerate(submission_request.documents)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            for chunks in results:
                all_chunks.extend(chunks)
        
        logger.info(f"📊 Total chunks processed: {len(all_chunks)}")
        
        if not all_chunks:
            logger.error("❌ No valid content extracted!")
            return SubmissionResponse(answers=[
                "No valid content could be extracted from the provided documents."
                for _ in submission_request.questions
            ])
        
        # Add to RAG pipeline with advanced processing
        await rag_pipeline.add_documents(all_chunks)
        
        # Answer all questions with semantic intelligence
        logger.info(f"❓ Answering questions with semantic processing...")
        
        # Limit concurrent questions for stability
        semaphore = asyncio.Semaphore(2)
        
        async def answer_single_question(question: str) -> str:
            async with semaphore:
                return await rag_pipeline.answer_question(question)
        
        tasks = [answer_single_question(q) for q in submission_request.questions]
        answers = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 ULTIMATE SUCCESS! Processed in {elapsed:.2f}s")
        
        return SubmissionResponse(answers=answers)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"💥 CRITICAL ERROR after {elapsed:.2f}s: {e}")
        
        return SubmissionResponse(answers=[
            "Processing error occurred. Please try again."
            for _ in submission_request.questions
        ])

# --- HEALTH ENDPOINTS ---
@app.get("/")
def read_root():
    return {
        "message": "🏆 ULTIMATE HACKATHON WINNING RAG SYSTEM",
        "version": "4.0.0",
        "status": "READY TO DOMINATE!",
        "supported_formats": list(doc_processor.processors.keys()),
        "features": [
            "Multi-format document processing (PDF, DOCX, Excel, CSV, HTML, etc.)",
            "Multi-LLM fallback system (Groq, OpenAI, Gemini)",
            "Advanced semantic query enhancement",
            "CrossEncoder reranking for accuracy",
            "Anti-jailbreak security system",
            "Smart caching and concurrent processing",
            "Semantic chunking and context fusion",
            "R4 'half questions' handling",
            "Lightning-fast response times"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "cache_size": len(doc_processor.cache),
        "timestamp": time.time()
    }

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
