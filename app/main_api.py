# --- ULTIMATE HACKATHON WINNING RAG SYSTEM ---

import os
import json
import uuid
import time
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from itertools import cycle
import hashlib
import mimetypes
from pathlib import Path

# FastAPI and core dependencies
from fastapi import FastAPI, Body, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.document import Document as LangChainDocument

# Multi-format document processing
import fitz  # PyMuPDF
import pdfplumber
import docx  # python-docx
import openpyxl
import csv
import zipfile
import rarfile
import email
from email.policy import default
import eml_parser
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Multiple LLM providers
import groq
import openai
import google.generativeai as genai

# Other dependencies
import httpx
from dotenv import load_dotenv
import cachetools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ultimate Hackathon RAG System", version="3.0.0")

# Enhanced CORS for all scenarios
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

# --- MULTI-LLM PROVIDER SYSTEM ---

class MultiLLMManager:
    def __init__(self):
        # Initialize multiple LLM providers
        self.groq_keys = cycle([k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(',') if k.strip()])
        self.openai_keys = cycle([k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(',') if k.strip()])
        self.gemini_keys = cycle([k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(',') if k.strip()])
        
        self.providers = ['groq', 'openai', 'gemini']
        self.current_provider_index = 0
        
        logger.info("🔑 Multi-LLM Manager initialized with fallback support")
    
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
        self.max_pages = 30
        
        # Smart caching system
        self.cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL
        self.security_guard = SecurityGuard()
        
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
            '.rar': self.process_archive,
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
    
    # --- PDF PROCESSING (Enhanced) ---
    async def process_pdf(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Enhanced PDF processing with tables and images"""
        chunks = []
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        try:
            # Extract text with PyMuPDF
            doc = fitz.open(file_path)
            full_text = ""
            
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Extract images as context (if they contain text)
                image_list = page.get_images()
                for img in image_list[:3]:  # Limit images
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        # Could add OCR here if needed
                    except:
                        pass
                
                if text.strip():
                    full_text += f"\n\nPage {page_num + 1}:\n{self._clean_text(text)}"
            
            doc.close()
            
            # Extract tables with pdfplumber
            table_text = await self._extract_pdf_tables(file_path)
            if table_text:
                full_text += f"\n\n=== TABLES ===\n{table_text}"
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(full_text, file_path, "pdf")
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return chunks
    
    async def _extract_pdf_tables(self, file_path: str) -> str:
        """Extract tables from PDF"""
        table_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:15]):
                    tables = page.find_tables()
                    for i, table in enumerate(tables[:3]):
                        try:
                            table_data = table.extract()
                            if table_data and len(table_data) > 1:
                                table_md = f"\n**Table {i+1} (Page {page_num+1})**\n"
                                for row in table_data[:20]:
                                    if row:
                                        clean_row = [str(cell or "").strip()[:50] for cell in row]
                                        table_md += "| " + " | ".join(clean_row) + " |\n"
                                table_text += table_md + "\n"
                        except:
                            continue
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return table_text
    
    # --- DOCX/DOC PROCESSING ---
    async def process_docx(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process DOCX files"""
        with open(file_path, 'wb') as f:
            f.write(content)
        
        try:
            doc = docx.Document(file_path)
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
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return chunks
    
    async def process_doc(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process DOC files (fallback to text extraction)"""
        return self._emergency_text_extraction(content, file_path)
    
    # --- EXCEL PROCESSING ---
    async def process_excel(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process Excel files"""
        with open(file_path, 'wb') as f:
            f.write(content)
        
        try:
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            full_text = ""
            
            for sheet_name in workbook.sheetnames[:5]:  # Max 5 sheets
                sheet = workbook[sheet_name]
                full_text += f"\n**Sheet: {sheet_name}**\n"
                
                # Get data as table
                data = []
                for row in sheet.iter_rows(max_row=min(sheet.max_row, 100), values_only=True):
                    if any(cell for cell in row):  # Skip empty rows
                        data.append([str(cell or "").strip() for cell in row])
                
                if data:
                    # Format as table
                    for row in data:
                        full_text += "| " + " | ".join(row[:10]) + " |\n"  # Max 10 columns
                
                full_text += "\n"
            
            workbook.close()
            chunks = self._create_semantic_chunks(full_text, file_path, "excel")
            
        except Exception as e:
            logger.error(f"Excel processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return chunks
    
    # --- CSV PROCESSING ---
    async def process_csv(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process CSV files"""
        try:
            text_content = content.decode('utf-8', errors='ignore')
            lines = text_content.split('\n')
            
            full_text = "**CSV DATA**\n"
            for i, line in enumerate(lines[:200]):  # Max 200 rows
                if line.strip():
                    # Parse CSV row
                    row_data = next(csv.reader([line]))
                    full_text += "| " + " | ".join(str(cell).strip()[:50] for cell in row_data) + " |\n"
            
            chunks = self._create_semantic_chunks(full_text, file_path, "csv")
            
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        return chunks
    
    # --- EMAIL PROCESSING ---
    async def process_email(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process email files"""
        try:
            # Parse email
            msg = email.message_from_bytes(content, policy=default)
            
            full_text = f"**EMAIL**\n"
            full_text += f"From: {msg.get('From', 'Unknown')}\n"
            full_text += f"To: {msg.get('To', 'Unknown')}\n"
            full_text += f"Subject: {msg.get('Subject', 'No Subject')}\n"
            full_text += f"Date: {msg.get('Date', 'Unknown')}\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_content()
                        full_text += f"Content:\n{body}\n"
            else:
                body = msg.get_content()
                full_text += f"Content:\n{body}\n"
            
            chunks = self._create_semantic_chunks(full_text, file_path, "email")
            
        except Exception as e:
            logger.error(f"Email processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        return chunks
    
    # --- HTML/XML PROCESSING ---
    async def process_html(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process HTML files"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            chunks = self._create_semantic_chunks(text, file_path, "html")
            
        except Exception as e:
            logger.error(f"HTML processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        return chunks
    
    async def process_xml(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process XML files"""
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
            chunks = self._create_semantic_chunks(full_text, file_path, "xml")
            
        except Exception as e:
            logger.error(f"XML processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        return chunks
    
    # --- ARCHIVE PROCESSING ---
    async def process_archive(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process ZIP/RAR archives"""
        with open(file_path, 'wb') as f:
            f.write(content)
        
        chunks = []
        try:
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    for file_info in zip_file.filelist[:10]:  # Max 10 files
                        try:
                            file_content = zip_file.read(file_info)
                            sub_chunks = await self.process_document(file_info.filename, file_content)
                            chunks.extend(sub_chunks)
                        except:
                            continue
            
            # Could add RAR support here if needed
            
        except Exception as e:
            logger.error(f"Archive processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return chunks
    
    # --- JSON PROCESSING ---
    async def process_json(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process JSON files"""
        try:
            data = json.loads(content.decode('utf-8'))
            full_text = json.dumps(data, indent=2, ensure_ascii=False)
            chunks = self._create_semantic_chunks(full_text, file_path, "json")
        except Exception as e:
            logger.error(f"JSON processing error: {e}")
            chunks = self._emergency_text_extraction(content, file_path)
        
        return chunks
    
    # --- TEXT PROCESSING ---
    async def process_text(self, file_path: str, content: bytes) -> List[Dict[str, Any]]:
        """Process plain text files"""
        try:
            text = content.decode('utf-8', errors='ignore')
            chunks = self._create_semantic_chunks(text, file_path, "text")
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            chunks = []
        
        return chunks
    
    # --- UTILITY METHODS ---
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove noise
        noise_patterns = [
            r'Office of the Insurance Ombudsman.*?\n',
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
        
        # Semantic boundary detection
        boundaries = [0]
        
        # Look for section markers
        section_patterns = [
            r'\n\s*(?:\d+\.)+\s*[A-Z]',
            r'\n\s*[A-Z][A-Z\s]{8,}:',
            r'\n\s*(?:TABLE|SECTION|PART)',
            r'\n\s*\*\*[^*]+\*\*'
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))
        
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > self.chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "content": sub_chunk,
                        "metadata": {
                            "source": os.path.basename(source),
                            "chunk_index": len(chunks),
                            "document_type": doc_type,
                            "chunk_length": len(sub_chunk),
                            "is_sub_chunk": True,
                            "parent_chunk": i
                        },
                        "chunk_id": str(uuid.uuid4())
                    })
            elif len(chunk_text) > 100:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": os.path.basename(source),
                        "chunk_index": len(chunks),
                        "document_type": doc_type,
                        "chunk_length": len(chunk_text),
                        "is_sub_chunk": False
                    },
                    "chunk_id": str(uuid.uuid4())
                })
        
        return chunks[:self.max_chunks]
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split large chunks intelligently"""
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
    
    def _emergency_text_extraction(self, content: bytes, file_path: str) -> List[Dict[str, Any]]:
        """Emergency text extraction for unsupported formats"""
        try:
            text = content.decode('utf-8', errors='ignore')
            if len(text) > 50:
                chunks = self._create_semantic_chunks(text, file_path, "unknown")
                return chunks
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

# --- ENHANCED RAG PIPELINE ---

class UltimateRAGPipeline:
    def __init__(self, collection_name: str, llm_manager: MultiLLMManager):
        self.collection_name = collection_name
        self.llm_manager = llm_manager
        self.security_guard = SecurityGuard()
        
        # Initialize embedding model (cached)
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
        """Add documents with advanced filtering"""
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
            for chunk in quality_chunks
        ]
        
        # Add to vector store
        if documents:
            self.vectorstore.add_documents(documents)
            logger.info(f"✅ Added {len(documents)} documents to vector store")
    
    async def answer_question(self, question: str) -> str:
        """Answer question with security and quality checks"""
        # Security check
        if self.security_guard.detect_jailbreak(question):
            return self.security_guard.sanitize_response(question, "")
        
        try:
            # Enhanced retrieval
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 15,  # More documents
                    "fetch_k": 30,
                    "lambda_mult": 0.5
                }
            )
            
            relevant_docs = retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return "I don't have enough information in the provided documents to answer this question."
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(context, question)
            
            # Get response from multi-LLM system
            response = await self.llm_manager.get_response(prompt)
            
            # Final security check
            response = self.security_guard.sanitize_response(question, response)
            
            # Clean formatting
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Question processing failed: {e}")
            return "An error occurred while processing your question."
    
    def _create_enhanced_prompt(self, context: str, question: str) -> str:
        """Create enhanced prompt for better responses"""
        return f"""You are an expert document analyst. Analyze the provided document context to answer the question accurately and professionally.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide accurate answers based ONLY on the document context
- Include specific details: numbers, percentages, dates, amounts, conditions
- Write in clear, professional language without excessive quotes
- If multiple conditions apply, list them clearly
- Be precise about limitations, exceptions, and requirements
- If information is incomplete, state what is available
- Do not make assumptions beyond what is stated in the documents

ANSWER:"""
    
    def _clean_response(self, response: str) -> str:
        """Clean response formatting"""
        # Remove excessive quotes
        response = re.sub(r'"([^"]{1,50})"', r'\1', response)
        response = re.sub(r'"(\w+)"', r'\1', response)
        
        # Fix spacing
        response = re.sub(r'\s+', ' ', response)
        response = response.replace(' ,', ',')
        response = response.replace(' .', '.')
        
        # Clean newlines
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

# Initialize global services
multi_llm = MultiLLMManager()
doc_processor = UniversalDocumentProcessor()

# --- API MODELS ---

class SubmissionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- MAIN ENDPOINT ---

@app.post("/hackrx/run", response_model=SubmissionResponse, dependencies=[Depends(verify_bearer_token)])
async def run_submission(request: Request, submission_request: SubmissionRequest = Body(...)):
    start_time = time.time()
    logger.info(f"🎯 ULTIMATE PROCESSING: {len(submission_request.documents)} docs, {len(submission_request.questions)} questions")
    
    try:
        # Create unique session
        session_id = f"ultimate_{uuid.uuid4().hex[:8]}"
        rag_pipeline = UltimateRAGPipeline(session_id, multi_llm)
        
        # Process all documents concurrently
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
                        
                        # Process document
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
        
        # Add to RAG pipeline
        await rag_pipeline.add_documents(all_chunks)
        
        # Answer all questions concurrently
        logger.info(f"❓ Answering questions...")
        
        # Limit concurrent questions to avoid overwhelming the LLM
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
            f"Processing error occurred. Please try again."
            for _ in submission_request.questions
        ])

# --- HEALTH ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "message": "🏆 ULTIMATE HACKATHON RAG SYSTEM",
        "version": "3.0.0",
        "status": "READY TO WIN!",
        "supported_formats": list(doc_processor.processors.keys()),
        "features": [
            "Multi-format document processing",
            "Multi-LLM fallback system", 
            "Anti-jailbreak security",
            "Smart caching",
            "Concurrent processing",
            "Semantic chunking"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "cache_size": len(doc_processor.cache),
        "timestamp": time.time()
    }

# --- TESTING ENDPOINT ---

@app.post("/test")
async def test_endpoint(request: dict):
    """Test endpoint for validation"""
    return {
        "status": "success",
        "message": "Ultimate RAG system is operational",
        "processed_request": request
    }
