# --- OPTIMIZED SEMANTIC RAG SYSTEM ---

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

# LLM Integration
import groq
import httpx
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized Semantic RAG", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- OPTIMIZED SEMANTIC DOCUMENT PARSER ---

class DocumentChunk:
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

class OptimizedSemanticParser:
    def __init__(self):
        # Optimized parameters - balanced between quality and performance
        self.chunk_size = 1200
        self.chunk_overlap = 180
        self.max_chunks = 200  # Sweet spot for memory vs coverage
        self.max_pages = 20    # Reduced from 30
        logger.info("OptimizedSemanticParser initialized")

    def semantic_text_split(self, text: str, source: str) -> List[str]:
        """Optimized semantic text splitting - keeps intelligence while being efficient"""
        if not text or len(text) < 100:
            return [text] if text else []
        
        chunks = []
        
        # Semantic boundary patterns (optimized list)
        semantic_patterns = [
            r'\n\s*(?:\d+\.)+\s*[A-Z][^.\n]*[.:]',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]{8,}[:\n]',  # ALL CAPS HEADINGS
            r'\n\s*(?:EXCLUSIONS?|BENEFITS?|COVERAGE|DEFINITIONS?)',  # Key sections
            r'\n\s*(?:WAITING\s+PERIOD|GRACE\s+PERIOD|CLAIMS?)',  # Important terms
        ]
        
        # Find semantic boundaries efficiently
        boundaries = [0]
        for pattern in semantic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            boundaries.extend(match.start() for match in matches)
        
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))
        
        # Create semantic chunks
        for i in range(len(boundaries) - 1):
            section_start = boundaries[i]
            section_end = boundaries[i + 1]
            section_text = text[section_start:section_end].strip()
            
            if len(section_text) <= self.chunk_size:
                if section_text and len(section_text) > 80:
                    chunks.append(section_text)
            else:
                # Split large sections intelligently
                sub_chunks = self._split_section_smartly(section_text)
                chunks.extend(sub_chunks)
        
        # Fallback to sentence-based splitting if no boundaries found
        if len(chunks) == 0:
            chunks = self._fallback_sentence_split(text)
        
        # Limit total chunks for memory management
        chunks = chunks[:self.max_chunks]
        logger.info(f"Split {source} into {len(chunks)} semantic chunks")
        return chunks

    def _split_section_smartly(self, text: str) -> List[str]:
        """Smart splitting for large sections"""
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

    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback intelligent sentence-based splitting"""
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
        """Optimized semantic table extraction"""
        table_text = ""
        table_count = 0
        max_tables = 12  # Balanced number

        try:
            with pdfplumber.open(file_path) as pdf:
                # Process key pages only
                pages_to_process = list(range(min(len(pdf.pages), 18)))
                
                for page_num in pages_to_process:
                    if table_count >= max_tables:
                        break
                        
                    page = pdf.pages[page_num]
                    tables = page.find_tables()

                    for table in tables[:2]:  # Max 2 tables per page for efficiency
                        if table_count >= max_tables:
                            break

                        try:
                            table_data = table.extract()
                            if table_data and len(table_data) >= 2:
                                
                                # Semantic relevance check (optimized)
                                table_str = str(table_data).lower()
                                insurance_keywords = ['premium', 'coverage', 'benefit', 'waiting', 'exclusion', 
                                                    'claim', 'limit', 'sum', 'medical', 'hospital']
                                
                                if any(keyword in table_str for keyword in insurance_keywords):
                                    # Skip administrative tables
                                    if not any(admin in table_str for admin in ['ombudsman', 'lalit bhawan']):
                                        
                                        # Format table efficiently
                                        table_md = f"\n**POLICY TABLE {table_count + 1} (Page {page_num + 1})**\n"
                                        
                                        # Limit rows for memory efficiency
                                        limited_data = table_data[:min(15, len(table_data))]
                                        
                                        for row in limited_data:
                                            if row:
                                                row_str = " | ".join(str(cell or "")[:40] for cell in row)
                                                table_md += f"| {row_str} |\n"
                                        
                                        table_text += table_md + "\n"
                                        table_count += 1

                        except Exception:
                            continue

                logger.info(f"Extracted {table_count} semantic tables")

        except Exception as e:
            logger.warning(f"Semantic table extraction failed: {e}")

        return table_text

    def process_pdf_optimized_semantic(self, file_path: str) -> List[DocumentChunk]:
        """Optimized semantic PDF processing - keeps intelligence while being memory efficient"""
        logger.info(f"🚀 Processing PDF with optimized semantics: {os.path.basename(file_path)}")
        start_time = time.time()
        chunks = []

        try:
            # Efficient text extraction
            doc = fitz.open(file_path)
            full_text = ""
            total_pages = len(doc)
            
            # Process optimized number of pages
            pages_to_process = list(range(min(total_pages, self.max_pages)))
            
            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    # Intelligent content filtering
                    lines = page_text.split('\n')
                    filtered_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if (line and len(line) > 15 and 
                            not any(noise in line.lower() for noise in 
                                  ['ombudsman', 'lalit bhawan', 'page ']) and
                            not re.match(r'^\d+\s*$', line)):  # Skip page numbers
                            filtered_lines.append(line)
                    
                    clean_text = '\n'.join(filtered_lines)
                    if clean_text and len(clean_text) > 100:
                        full_text += f"\n\nPage {page_num + 1}:\n{clean_text}"
                        
                except Exception:
                    continue

            doc.close()

            # Add semantic tables
            table_content = self.extract_semantic_tables(file_path)
            if table_content:
                full_text += f"\n\n{'='*40}\nKEY POLICY TABLES\n{'='*40}\n{table_content}"

            # Create semantic chunks
            text_chunks = self.semantic_text_split(full_text, os.path.basename(file_path))

            # Create chunks with optimized semantic metadata
            for idx, chunk_text in enumerate(text_chunks):
                # Lightweight semantic analysis
                chunk_lower = chunk_text.lower()
                
                # Detect content types efficiently
                content_types = []
                type_indicators = {
                    'definitions': ['means', 'definition', 'shall mean'],
                    'coverage': ['coverage', 'covered', 'benefit'],
                    'exclusions': ['exclusion', 'excluded', 'not covered'],
                    'waiting_periods': ['waiting period', 'wait'],
                    'claims': ['claim', 'settlement'],
                    'premium': ['premium', 'payment', 'grace period'],
                    'medical': ['hospital', 'medical', 'treatment']
                }
                
                for content_type, indicators in type_indicators.items():
                    if any(indicator in chunk_lower for indicator in indicators):
                        content_types.append(content_type)
                
                # Calculate simple relevance score
                insurance_terms = ['policy', 'coverage', 'benefit', 'exclusion', 'claim', 'premium']
                relevance_score = sum(1 for term in insurance_terms if term in chunk_lower)

                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "chunk_index": idx,
                        "document_type": "optimized_semantic",
                        "content_types": ", ".join(content_types) if content_types else "general",
                        "total_pages": total_pages,
                        "chunk_length": len(chunk_text),
                        "relevance_score": relevance_score,
                        "has_tables": "table" in chunk_text.lower()
                    },
                    chunk_id=str(uuid.uuid4())
                ))

            elapsed = time.time() - start_time
            logger.info(f"✅ Optimized semantic processing complete in {elapsed:.2f}s: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"❌ Optimized semantic processing failed: {e}")
            return self._emergency_fallback(file_path)

    def _emergency_fallback(self, file_path: str) -> List[DocumentChunk]:
        """Emergency fallback that still maintains some intelligence"""
        logger.info("🆘 Emergency fallback with basic semantics")
        try:
            doc = fitz.open(file_path)
            text_parts = []
            
            for page_num in range(min(15, len(doc))):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Basic semantic filtering
                if (len(page_text.strip()) > 100 and
                    'ombudsman' not in page_text.lower()):
                    text_parts.append(page_text)

            doc.close()
            full_text = "\n\n".join(text_parts)

            # Simple but effective chunking
            chunks = []
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= 1000:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(DocumentChunk(
                            content=current_chunk.strip(),
                            metadata={
                                "source": os.path.basename(file_path),
                                "chunk_index": len(chunks),
                                "document_type": "emergency_fallback"
                            },
                            chunk_id=str(uuid.uuid4())
                        ))
                    current_chunk = sentence + " "
            
            if current_chunk.strip():
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    metadata={
                        "source": os.path.basename(file_path),
                        "chunk_index": len(chunks),
                        "document_type": "emergency_fallback"
                    },
                    chunk_id=str(uuid.uuid4())
                ))

            return chunks[:100]  # Limit for safety

        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")
            raise Exception("All processing methods failed")

# --- GROQ LLM WRAPPER ---

class GroqLLM(LLM):
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
                max_tokens=900,
                top_p=0.9,
                stop=stop
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq LLM call failed: {e}")
            return "Error generating response"

# --- OPTIMIZED SEMANTIC RAG PIPELINE ---

class OptimizedSemanticRAGPipeline:
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
        logger.info(f"✅ Optimized semantic RAG pipeline initialized: {collection_name}")

    def clean_response(self, answer: str) -> str:
        """Clean up the response formatting for better readability"""
        if not answer:
            return answer
        
        # Remove excessive newlines
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Multiple newlines to double
        answer = re.sub(r'\n\s*\n', '\n\n', answer)  # Ensure consistent double newlines for paragraphs
        
        # Remove quotes around single words and short phrases
        answer = re.sub(r'"([A-Z\s]{2,20})"', r'\1', answer)  # Remove quotes from short caps phrases
        answer = re.sub(r'"(\w+)"', r'\1', answer)  # Remove quotes from single words
        answer = re.sub(r'"(Rs\. [\d,]+[/-]*)"', r'\1', answer)  # Remove quotes from amounts
        answer = re.sub(r'"(\d+%)"', r'\1', answer)  # Remove quotes from percentages
        answer = re.sub(r'"(\d+ (?:days?|months?|years?))"', r'\1', answer)  # Remove quotes from time periods
        
        # Clean up policy references - keep important quotes but make them flow better
        answer = re.sub(r'As stated in the policy: "([^"]+)"', r'The policy states that \1', answer)
        answer = re.sub(r'According to the policy document: "([^"]+)"', r'According to the policy document, \1', answer)
        answer = re.sub(r'The policy states: "([^"]+)"', r'The policy states that \1', answer)
        answer = re.sub(r'As per the policy: "([^"]+)"', r'As per the policy, \1', answer)
        
        # Fix spacing and formatting
        answer = re.sub(r'\s+', ' ', answer)  # Multiple spaces to single space
        answer = answer.replace(' ,', ',')  # Fix spacing before commas
        answer = answer.replace(' .', '.')  # Fix spacing before periods
        answer = answer.strip()  # Remove leading/trailing whitespace
        
        # Clean up excessive line breaks in the middle of sentences
        answer = re.sub(r'([a-z,])\s*\n\s*([a-z])', r'\1 \2', answer)
        
        return answer

    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            logger.error("❌ No chunks provided!")
            return

        logger.info(f"📚 Adding {len(chunks)} chunks with optimized semantic processing...")
        
        # Optimized semantic filtering
        quality_chunks = []
        for chunk in chunks:
            content = chunk['content']
            metadata = chunk.get('metadata', {})
            
            # Multi-factor quality assessment
            quality_factors = []
            
            # Length factor
            if len(content) > 120:
                quality_factors.append(1)
            
            # Insurance relevance factor
            insurance_terms = ['policy', 'coverage', 'benefit', 'exclusion', 'claim', 'premium', 
                             'hospital', 'medical', 'treatment', 'waiting', 'insured']
            term_count = sum(1 for term in insurance_terms if term in content.lower())
            if term_count >= 2:
                quality_factors.append(2)
            
            # Content type factor
            content_types = metadata.get('content_types', '')
            if content_types and content_types != 'general':
                quality_factors.append(1)
            
            # Noise penalty
            if any(noise in content.lower() for noise in ['ombudsman', 'lalit bhawan']):
                quality_factors.append(-2)
            
            # Calculate final quality score
            quality_score = sum(quality_factors)
            
            if quality_score > 0:
                quality_chunks.append(chunk)

        # Sort by relevance score if available
        quality_chunks.sort(key=lambda x: x['metadata'].get('relevance_score', 0), reverse=True)
        
        # Limit for memory efficiency while keeping quality
        if len(quality_chunks) > 120:
            quality_chunks = quality_chunks[:120]

        logger.info(f"📚 Filtered to {len(quality_chunks)} high-quality semantic chunks")

        langchain_docs = [
            LangChainDocument(page_content=chunk['content'], metadata=chunk['metadata'])
            for chunk in quality_chunks
        ]

        self.vectorstore.add_documents(langchain_docs)
        logger.info(f"✅ Added {len(langchain_docs)} semantic documents to vectorstore")

        # Optimized retriever with semantic search
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Keep MMR for diversity
            search_kwargs={
                "k": 10,  # Balanced retrieval
                "fetch_k": 20,  # Reasonable search space
                "lambda_mult": 0.6  # Balance relevance vs diversity
            }
        )

        # Enhanced semantic prompt template with better formatting
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert insurance policy analyst with semantic understanding capabilities. Analyze the policy document context to provide accurate, detailed answers.

POLICY DOCUMENT CONTEXT:
{context}

QUESTION: {question}

SEMANTIC ANALYSIS INSTRUCTIONS:
- Carefully analyze the semantic meaning and relationships in the policy context
- Extract specific facts: numbers, percentages, time periods, conditions, and requirements
- Understand implicit connections between different policy sections
- Quote exact policy language when providing specific details, but format quotes naturally
- Synthesize information from multiple context sections when relevant
- Distinguish between explicit statements and reasonable inferences
- If information is partial, provide what's available and note limitations
- Be precise about conditions, exceptions, and qualifying circumstances

FORMATTING GUIDELINES:
- Write in clear, professional paragraphs without unnecessary line breaks
- When quoting policy text, integrate quotes smoothly into sentences
- Use bullet points or numbered lists only when listing multiple related items
- Avoid excessive quotation marks around single words or short phrases
- Write numbers and percentages directly (e.g., 30 days, 5%, Rs. 10,000) without quotes
- Make the response flow naturally and be easy to read

ANSWER FORMAT:
Provide a comprehensive, well-formatted answer based on your semantic analysis of the policy document context.

ANSWER:"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.groq_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

        logger.info("✅ Optimized semantic QA Chain ready")

    async def answer_question(self, question: str) -> str:
        if not self.qa_chain:
            return "Error: Semantic QA chain not initialized."

        logger.info(f"🤔 Semantic analysis for: {question}")

        try:
            # Retrieve with semantic understanding
            result = await asyncio.to_thread(self.qa_chain, {"query": question})
            raw_answer = result.get("result", "Failed to generate semantic answer.")
            
            # Clean up the response formatting
            clean_answer = self.clean_response(raw_answer)
            
            logger.info(f"✅ Semantic answer generated: {len(clean_answer)} characters")
            return clean_answer

        except Exception as e:
            logger.error(f"❌ Error during semantic QA: {e}")
            return "An error occurred while processing the semantic question."

# --- API KEY MANAGER ---

class GroqAPIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [key.strip() for key in api_keys if key.strip()]
        self.key_usage_count = defaultdict(int)
        self.current_key_index = 0
        logger.info(f"🔑 API Key Manager: {len(self.api_keys)} keys")

    def get_next_api_key(self):
        if not self.api_keys:
            raise ValueError("No API keys available")
        
        key = self.api_keys[self.current_key_index % len(self.api_keys)]
        self.current_key_index += 1
        return key

# --- CONFIGURATION ---

GROQ_API_KEYS = os.getenv("GROQ_API_KEYS", "").split(',')
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "/tmp/chroma_db_storage"
UPLOAD_DIR = "/tmp/docs"

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Initializing optimized semantic services...")
        
        app.state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        app.state.api_key_manager = GroqAPIKeyManager(GROQ_API_KEYS)
        first_key = app.state.api_key_manager.get_next_api_key()
        app.state.groq_client = groq.Groq(api_key=first_key)
        app.state.groq_llm = GroqLLM(groq_client=app.state.groq_client, api_key_manager=app.state.api_key_manager)
        
        app.state.parsing_service = OptimizedSemanticParser()
        
        logger.info("✅ All optimized semantic services initialized!")
        
    except Exception as e:
        logger.error(f"💥 FATAL ERROR: {e}")
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
    try:
        logger.info(f"🎯 Processing {len(submission_request.documents)} documents, {len(submission_request.questions)} questions")
        
        parsing_service = request.app.state.parsing_service
        session_collection_name = f"opt_semantic_{uuid.uuid4().hex[:8]}"
        rag_pipeline = OptimizedSemanticRAGPipeline(collection_name=session_collection_name, request=request)
        
        all_chunks = []

        # Process documents with optimized semantics
        async with httpx.AsyncClient(timeout=90.0) as client:
            for doc_idx, doc_url in enumerate(submission_request.documents):
                try:
                    logger.info(f"📥 Downloading document {doc_idx + 1}")
                    response = await client.get(doc_url, follow_redirects=True)
                    response.raise_for_status()

                    file_name = f"doc_{doc_idx}_{uuid.uuid4().hex[:8]}.pdf"
                    temp_file_path = os.path.join(UPLOAD_DIR, file_name)
                    os.makedirs(UPLOAD_DIR, exist_ok=True)

                    with open(temp_file_path, "wb") as f:
                        f.write(response.content)

                    logger.info(f"📄 Processing with optimized semantics...")
                    chunks = parsing_service.process_pdf_optimized_semantic(temp_file_path)
                    chunk_dicts = [chunk.to_dict() for chunk in chunks]
                    all_chunks.extend(chunk_dicts)

                    os.remove(temp_file_path)
                    logger.info(f"✅ Processed {len(chunks)} semantic chunks")

                except Exception as e:
                    logger.error(f"❌ Document processing failed: {e}")
                    continue

        logger.info(f"📊 Total semantic chunks: {len(all_chunks)}")

        if not all_chunks:
            logger.error("❌ No chunks processed!")
            return SubmissionResponse(answers=[
                Answer(question=q, answer="Document processing failed.")
                for q in submission_request.questions
            ])

        # Add to semantic RAG pipeline
        rag_pipeline.add_documents(all_chunks)

        # Answer questions with semantic understanding
        logger.info(f"❓ Answering questions with optimized semantics...")
        answers = []
        
        for question in submission_request.questions:
            try:
                answer = await rag_pipeline.answer_question(question)
                answers.append(Answer(question=question, answer=answer))
            except Exception as e:
                logger.error(f"❌ Question failed: {e}")
                answers.append(Answer(question=question, answer="Failed to process question."))
        
        logger.info("🎉 All semantic questions processed!")
        return SubmissionResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {e}")
        return SubmissionResponse(answers=[
            Answer(question=q, answer=f"System error: {str(e)}")
            for q in submission_request.questions
        ])

@app.get("/")
def read_root():
    return {"message": "Optimized Semantic RAG System", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.1.0"}
