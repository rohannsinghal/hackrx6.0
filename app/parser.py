# FAST Document Parser - Optimized for Speed and Large Documents

import os
import json
import uuid
import logging
import uvicorn
import gc
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException

# Minimal dependencies for speed
import fitz  # PyMuPDF - faster than Unstructured
import pdfplumber  # Only for tables
import mammoth
import email
import email.policy
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class FastDocumentParserService:
    """Ultra-fast document parsing service"""
    
    def __init__(self):
        self.chunk_size = 2000  # Larger chunks = fewer chunks
        self.chunk_overlap = 200  # Minimal overlap
        self.max_chunks = 500  # Hard limit on total chunks
        self.table_row_limit = 20  # Max rows per table
        logger.info("FastDocumentParserService initialized with speed optimizations")
    
    def fast_text_split(self, text: str, source: str) -> List[str]:
        """Super fast text splitting with hard limits"""
        if not text or len(text) < 100:
            return [text] if text else []
        
        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        chunk_count = 0
        
        while start < len(text) and chunk_count < self.max_chunks:
            end = min(start + self.chunk_size, len(text))
            
            # Quick sentence boundary check (no complex searching)
            if end < len(text):
                # Look back max 200 chars for period
                search_start = max(start, end - 200)
                period_pos = text.rfind('.', search_start, end)
                if period_pos > search_start:
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
            
            start = end - self.chunk_overlap
            
            # Safety break for infinite loops
            if start <= 0:
                start = end
        
        logger.info(f"Split {source} into {len(chunks)} chunks (limit: {self.max_chunks})")
        return chunks[:self.max_chunks]  # Hard limit

    def extract_tables_fast(self, file_path: str) -> str:
        """Fast table extraction with smart limits"""
        table_text = ""
        table_count = 0
        max_tables = 25  # Increased for better coverage
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Better sampling strategy
                if total_pages <= 20:
                    step = 1  # Process ALL pages for small docs
                elif total_pages <= 40:
                    step = 2  # Process every 2nd page for medium docs
                else:
                    step = 3  # Process every 3rd page for large docs
                    
                pages_to_process = list(range(0, min(total_pages, 50), step))  # Increased to 50 pages max
                
                logger.info(f"📊 Smart table scan: processing {len(pages_to_process)} of {total_pages} pages (step={step})")
                
                for page_num in pages_to_process:
                    if table_count >= max_tables:
                        break
                        
                    page = pdf.pages[page_num]
                    tables = page.find_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table_count >= max_tables:
                            break
                            
                        try:
                            table_data = table.extract()
                            if table_data and len(table_data) >= 2:
                                # Better table processing
                                limited_data = table_data[:min(30, len(table_data))]  # Up to 30 rows
                                
                                # Smart markdown conversion with better formatting
                                if len(limited_data[0]) <= 6:  # Reasonable number of columns
                                    header = " | ".join(str(cell or "").strip()[:60] for cell in limited_data[0])  # 60 chars per cell
                                    separator = " | ".join(["---"] * len(limited_data[0]))
                                    
                                    rows = []
                                    for row in limited_data[1:]:
                                        # Pad row to match header length
                                        padded_row = list(row) + [None] * (len(limited_data[0]) - len(row))
                                        row_str = " | ".join(str(cell or "").strip()[:60] for cell in padded_row)
                                        rows.append(row_str)
                                    
                                    table_md = f"\n**TABLE {table_count + 1} - Page {page_num + 1}**\n"
                                    table_md += f"*{len(limited_data)} rows × {len(limited_data[0])} columns*\n\n"
                                    table_md += f"| {header} |\n| {separator} |\n"
                                    for row in rows:
                                        table_md += f"| {row} |\n"
                                    table_md += "\n"
                                    
                                    table_text += table_md
                                    table_count += 1
                                    logger.info(f"⚡ Table {table_count}: {len(limited_data)}×{len(limited_data[0])} from page {page_num + 1}")
                                else:
                                    logger.info(f"⚠️ Skipped wide table ({len(limited_data[0])} cols) on page {page_num + 1}")
                        
                        except Exception as e:
                            logger.warning(f"⚠️ Skip table on page {page_num + 1}: {e}")
                
                logger.info(f"🎯 Extracted {table_count} tables in fast mode")
                
        except Exception as e:
            logger.error(f"❌ Fast table extraction failed: {e}")
        
        return table_text

    def process_pdf_ultrafast(self, file_path: str) -> List[DocumentChunk]:
        """Ultra-fast PDF processing - under 1 minute target"""
        logger.info(f"🚀 ULTRA-FAST PDF processing: {os.path.basename(file_path)}")
        start_time = __import__('time').time()
        
        chunks = []
        
        try:
            # STEP 1: Fast table extraction (parallel to text extraction)
            logger.info("📊 Fast table extraction...")
            table_content = self.extract_tables_fast(file_path)
            
            # STEP 2: Fast text extraction with PyMuPDF
            logger.info("📄 Fast text extraction with PyMuPDF...")
            doc = fitz.open(file_path)
            
            full_text = ""
            total_pages = len(doc)
            
            # Process pages in chunks for large documents
            if total_pages > 40:
                # For very large docs, process every 2nd page
                pages_to_process = list(range(0, min(total_pages, 60), 2))
                logger.info(f"📑 Large document: processing {len(pages_to_process)} of {total_pages} pages")
            else:
                pages_to_process = list(range(total_pages))
            
            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    # Clean and limit page text
                    page_text = page_text.strip()
                    if len(page_text) > 10000:  # Limit page size
                        page_text = page_text[:10000] + f"\n[Page {page_num + 1} truncated for speed]"
                    
                    full_text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing page {page_num + 1}: {e}")
            
            doc.close()
            
            # STEP 3: Append tables at the end
            if table_content:
                full_text += f"\n\n{'='*50}\nEXTRACTED TABLES\n{'='*50}\n{table_content}"
            
            # STEP 4: Fast chunking with hard limits
            logger.info("📦 Creating chunks...")
            text_chunks = self.fast_text_split(full_text, os.path.basename(file_path))
            
            # STEP 5: Create DocumentChunk objects
            for idx, chunk_text in enumerate(text_chunks):
                has_tables = "**TABLE" in chunk_text or "EXTRACTED TABLES" in chunk_text
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "chunk_index": idx,
                        "document_type": "pdf_ultrafast",
                        "has_tables": has_tables,
                        "total_pages": total_pages,
                        "pages_processed": len(pages_to_process),
                        "processing_method": "ultrafast_pymupdf"
                    },
                    chunk_id=str(uuid.uuid4())
                ))
            
            elapsed = __import__('time').time() - start_time
            logger.info(f"✅ ULTRA-FAST processing complete in {elapsed:.2f}s: {len(chunks)} chunks")
            
            if elapsed > 90:  # 1.5 minutes
                logger.warning(f"⚠️ Processing took {elapsed:.2f}s - consider reducing document size")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast processing failed: {e}")
            return self._emergency_fallback(file_path)

    def _emergency_fallback(self, file_path: str) -> List[DocumentChunk]:
        """Emergency fallback - text only, no tables"""
        logger.info("🆘 Emergency fallback: text-only extraction")
        
        try:
            doc = fitz.open(file_path)
            
            # Process only first 10 pages
            max_pages = min(10, len(doc))
            text_parts = []
            
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = page.get_text()
                if len(page_text) > 5000:
                    page_text = page_text[:5000] + f"\n[Page {page_num + 1} truncated]"
                text_parts.append(f"Page {page_num + 1}:\n{page_text}")
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            chunks = []
            
            # Create max 10 chunks
            chunk_size = len(full_text) // 10 + 1
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i + chunk_size]
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

    def process_word_doc_fast(self, file_path: str) -> List[DocumentChunk]:
        """Fast Word document processing"""
        chunks = []
        
        try:
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                soup = BeautifulSoup(result.html, 'html.parser')
                
                # Quick table conversion
                tables = soup.find_all('table')
                for idx, table in enumerate(tables[:10]):  # Max 10 tables
                    rows = table.find_all('tr')[:15]  # Max 15 rows per table
                    table_md = f"\n**TABLE {idx + 1}**\n"
                    
                    for row in rows:
                        cells = [cell.get_text(strip=True)[:30] for cell in row.find_all(['td', 'th'])]
                        table_md += "| " + " | ".join(cells) + " |\n"
                    
                    table.replace_with(table_md)
                
                text_content = soup.get_text()
                text_chunks = self.fast_text_split(text_content, os.path.basename(file_path))
                
                for idx, chunk in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        content=chunk,
                        metadata={
                            "source": os.path.basename(file_path),
                            "chunk_index": idx,
                            "document_type": "docx_fast",
                            "has_tables": "**TABLE" in chunk
                        },
                        chunk_id=str(uuid.uuid4())
                    ))
                    
        except Exception as e:
            logger.error(f"Fast Word processing failed: {e}")
            raise Exception(f"Word processing failed: {e}")
        
        return chunks
    
    def process_email_fast(self, file_path: str) -> List[DocumentChunk]:
        """Fast email processing"""
        chunks = []
        
        try:
            with open(file_path, 'rb') as email_file:
                msg = email.message_from_bytes(email_file.read(), policy=email.policy.default)
                
                subject = msg.get('Subject', 'No Subject')
                sender = msg.get('From', 'Unknown Sender')
                date = msg.get('Date', 'Unknown Date')
                
                # Get body content quickly
                body_content = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            content = part.get_content()[:5000]  # Limit size
                            body_content += content
                            break  # Take first text part only
                else:
                    body_content = msg.get_content()[:5000]
                
                email_content = f"EMAIL: {subject}\nFrom: {sender}\nDate: {date}\n\n{body_content}"
                text_chunks = self.fast_text_split(email_content, os.path.basename(file_path))
                
                for idx, chunk in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        content=chunk,
                        metadata={
                            "source": os.path.basename(file_path),
                            "chunk_index": idx,
                            "document_type": "email_fast",
                            "subject": subject
                        },
                        chunk_id=str(uuid.uuid4())
                    ))
                    
        except Exception as e:
            logger.error(f"Fast email processing failed: {e}")
            raise Exception(f"Email processing failed: {e}")
        
        return chunks


# Create the fast parser service
parser_service = FastDocumentParserService()

# FastAPI app
app = FastAPI(title="Ultra-Fast Document Parser", version="3.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Ultra-fast document parser running"}

@app.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    """Ultra-fast file parsing - target < 60 seconds"""
    temp_file_path = None
    start_time = __import__('time').time()
    
    try:
        gc.collect()  # Clean start
        
        temp_file_path = f"./temp_{uuid.uuid4()}_{file.filename}"
        
        # Fast file write
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_extension = Path(file.filename).suffix.lower()
        logger.info(f"⚡ FAST processing: {file.filename} ({file_extension})")
        
        # Route to appropriate fast processor
        if file_extension == '.pdf':
            chunks = parser_service.process_pdf_ultrafast(temp_file_path)
        elif file_extension in ['.docx', '.doc']:
            chunks = parser_service.process_word_doc_fast(temp_file_path)
        elif file_extension in ['.eml', '.msg']:
            chunks = parser_service.process_email_fast(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Convert to response format
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        
        elapsed = __import__('time').time() - start_time
        
        # Save minimal debug info
        try:
            with open("./_fast_parsed_output.json", "w") as f:
                json.dump({
                    "filename": file.filename,
                    "total_chunks": len(chunks),
                    "processing_time_seconds": elapsed,
                    "first_chunk_preview": chunks[0].content[:200] if chunks else "No chunks"
                }, f, indent=2)
        except:
            pass
        
        logger.info(f"🎯 COMPLETED {file.filename} in {elapsed:.2f}s: {len(chunks)} chunks")
        
        return {
            "filename": file.filename,
            "status": "success",
            "chunks": chunk_dicts,
            "total_chunks": len(chunks),
            "processing_time_seconds": round(elapsed, 2),
            "processing_method": "ultrafast"
        }
        
    except Exception as e:
        elapsed = __import__('time').time() - start_time
        logger.error(f"❌ Processing failed after {elapsed:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        gc.collect()

if __name__ == "__main__":
    logger.info("🚀 Starting Ultra-Fast Document Parser...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")