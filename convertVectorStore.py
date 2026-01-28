import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import re
from dotenv import load_dotenv
import os
import time
import torch

load_dotenv()

class HierarchicalVectorStore:
    def __init__(self, db_config: dict, device: str = None):
        """
        Initialize dengan database config
        db_config format: {
            'host': 'localhost',
            'database': 'your_db',
            'user': 'your_user',
            'password': 'your_password',
            'port': 5432
        }
        device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.db_config = db_config
        
        # Deteksi dan setup device (GPU/CPU)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load Qwen embedding model (1024 dimensions) ke GPU
        print("Loading Qwen3-Embedding-0.6B model...")
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=self.device)
        self.embedding_dim = 1024  # Qwen3-Embedding-0.6B menghasilkan 1024 dimensi
        
        print(f"Model loaded successfully on {self.device}!")
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Verify model is on GPU
        if self.device == 'cuda':
            print(f"Model device verification: {next(self.model.parameters()).device}")

    def connect_db(self):
        """Koneksi ke database"""
        return psycopg2.connect(**self.db_config)

    def count_words(self, text: str) -> int:
        """Hitung jumlah kata dalam teks"""
        if not text:
            return 0
        return len(text.split())

    def split_abstract(self, abstract: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """
        Split abstract jika lebih dari 1000 kata
        chunk_size: 600 kata per chunk
        overlap: 100 kata overlap
        """
        words = abstract.split()
        word_count = len(words)
        
        # Jika kurang dari 1000 kata, return as is
        if word_count <= 1000:
            return [abstract]
        
        chunks = []
        start = 0
        
        while start < word_count:
            end = min(start + chunk_size, word_count)
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Jika sudah sampai akhir, break
            if end >= word_count:
                break
            
            # Move start dengan mempertimbangkan overlap
            start = end - overlap
        
        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector dari text menggunakan Qwen3-Embedding-0.6B"""
        if not text or text.strip() == '':
            # Return zero vector jika text kosong
            return [0.0] * self.embedding_dim
        
        # Model akan otomatis menggunakan GPU karena sudah di-load ke GPU
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def insert_vector(self, conn, paper_id: str, chunk_type: str, content_chunk: str, 
                     chunk_index: int, embedding: List[float]):
        """Insert vector ke database"""
        cursor = conn.cursor()
        
        # Convert list to PostgreSQL array format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        query = """
            INSERT INTO bot.vector_store 
            (paper_id, chunk_type, content_chunk, chunk_index, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
        """
        
        cursor.execute(query, (paper_id, chunk_type, content_chunk, chunk_index, embedding_str))
        cursor.close()

    def is_valid_text(self, text) -> bool:
        """Check apakah text valid (tidak None, tidak kosong setelah strip)"""
        if text is None:
            return False
        if isinstance(text, str) and text.strip() == '':
            return False
        return True

    def format_judul_keyword(self, title: str, keywords: str) -> str:
        """
        Format gabungan judul dan keyword dengan struktur yang jelas
        Format: 
        Judul: [title]
        Kata Kunci: [keywords atau "Tidak tersedia"]
        """
        # Handle title
        title_text = title.strip() if self.is_valid_text(title) else "Tidak tersedia"
        
        # Handle keywords
        if self.is_valid_text(keywords):
            keywords_text = keywords.strip()
        else:
            keywords_text = "Tidak tersedia"
        
        # Format dengan struktur yang jelas
        formatted_text = f"Judul: {title_text}\nKata Kunci: {keywords_text}"
        return formatted_text

    def process_paper(self, paper_id: str, title: str, abstract: str, keywords: str, 
                     max_retries: int = 3, include_judul_keyword: bool = True):
        """
        Process satu paper menjadi embeddings dengan retry mechanism
        include_judul_keyword: jika True, generate chunk_type judul_keyword
        """
        conn = self.connect_db()
        processed_fields = []
        failed_fields = []
        
        try:
            # 1. Process TITLE
            if self.is_valid_text(title):
                print(f"  → Processing title for paper {paper_id}")
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        title_embedding = self.generate_embedding(title)
                        self.insert_vector(conn, paper_id, 'title', title, 0, title_embedding)
                        processed_fields.append('title')
                        print(f"    ✓ Title processed successfully")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"    ⚠ Error processing title (attempt {retry_count}/{max_retries}): {str(e)}")
                            print(f"    ↻ Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            print(f"    ✗ Failed to process title after {max_retries} attempts: {str(e)}")
                            failed_fields.append(('title', str(e)))
            else:
                print(f"  ⊘ Skipping title (empty or null)")

            # 2. Process KEYWORDS
            if self.is_valid_text(keywords):
                print(f"  → Processing keywords for paper {paper_id}")
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        keywords_embedding = self.generate_embedding(keywords)
                        self.insert_vector(conn, paper_id, 'keywords', keywords, 0, keywords_embedding)
                        processed_fields.append('keywords')
                        print(f"    ✓ Keywords processed successfully")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"    ⚠ Error processing keywords (attempt {retry_count}/{max_retries}): {str(e)}")
                            print(f"    ↻ Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            print(f"    ✗ Failed to process keywords after {max_retries} attempts: {str(e)}")
                            failed_fields.append(('keywords', str(e)))
            else:
                print(f"  ⊘ Skipping keywords (empty or null)")

            # 3. Process ABSTRACT (dengan splitting jika perlu)
            if self.is_valid_text(abstract):
                abstract_chunks = self.split_abstract(abstract)
                print(f"  → Processing abstract for paper {paper_id} - {len(abstract_chunks)} chunk(s)")
                
                abstract_success = 0
                abstract_failed = 0
                
                for idx, chunk in enumerate(abstract_chunks):
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            chunk_embedding = self.generate_embedding(chunk)
                            self.insert_vector(conn, paper_id, 'abstract', chunk, idx, chunk_embedding)
                            abstract_success += 1
                            print(f"    ✓ Abstract chunk {idx+1}/{len(abstract_chunks)} processed")
                            break
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"    ⚠ Error processing abstract chunk {idx+1} (attempt {retry_count}/{max_retries}): {str(e)}")
                                print(f"    ↻ Retrying in 2 seconds...")
                                time.sleep(2)
                            else:
                                print(f"    ✗ Failed to process abstract chunk {idx+1} after {max_retries} attempts: {str(e)}")
                                abstract_failed += 1
                                failed_fields.append((f'abstract_chunk_{idx}', str(e)))
                
                if abstract_success > 0:
                    processed_fields.append(f'abstract ({abstract_success}/{len(abstract_chunks)} chunks)')
            else:
                print(f"  ⊘ Skipping abstract (empty or null)")

            # 4. Process JUDUL_KEYWORD (gabungan title dan keywords)
            if include_judul_keyword:
                # Generate judul_keyword bahkan jika salah satu field kosong
                print(f"  → Processing judul_keyword for paper {paper_id}")
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        judul_keyword_text = self.format_judul_keyword(title, keywords)
                        judul_keyword_embedding = self.generate_embedding(judul_keyword_text)
                        self.insert_vector(conn, paper_id, 'judul_keyword', judul_keyword_text, 0, judul_keyword_embedding)
                        processed_fields.append('judul_keyword')
                        print(f"    ✓ Judul_keyword processed successfully")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"    ⚠ Error processing judul_keyword (attempt {retry_count}/{max_retries}): {str(e)}")
                            print(f"    ↻ Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            print(f"    ✗ Failed to process judul_keyword after {max_retries} attempts: {str(e)}")
                            failed_fields.append(('judul_keyword', str(e)))

            conn.commit()
            
            # Print summary
            print(f"\n  {'='*60}")
            if processed_fields:
                print(f"  ✓ Successfully processed: {', '.join(processed_fields)}")
            if failed_fields:
                print(f"  ✗ Failed to process:")
                for field, error in failed_fields:
                    print(f"    - {field}: {error}")
            
            if not processed_fields:
                print(f"  ⚠ WARNING: No fields were processed for paper {paper_id}")
            
            print(f"  {'='*60}\n")
            
            return len(failed_fields) == 0  # Return True jika tidak ada yang gagal
            
        except Exception as e:
            conn.rollback()
            print(f"\n  ✗ CRITICAL ERROR processing paper {paper_id}: {str(e)}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            print(f"  Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            conn.close()

    def process_all_papers(self, limit: int = None, regenerate: bool = False):
        """
        Process semua papers dari database
        limit: batasi jumlah paper yang diproses (optional)
        regenerate: jika True, hapus dan generate ulang semua vectors
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Query untuk ambil semua paper
        query = """
            SELECT item_uuid, title, abstract, keywords 
            FROM bot.paper_metadata
        """
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        papers = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"\nFound {len(papers)} papers to process")
        if regenerate:
            print("MODE: REGENERATE (akan hapus dan buat ulang semua vectors)")
        else:
            print("MODE: UPDATE (hanya process paper baru)")
        print("="*80)
        
        success_count = 0
        partial_success_count = 0
        error_count = 0
        
        for i, (item_uuid, title, abstract, keywords) in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing paper: {item_uuid}")
            try:
                if regenerate:
                    # Hapus vectors lama terlebih dahulu
                    self.clear_vectors_for_paper(item_uuid)
                
                fully_successful = self.process_paper(item_uuid, title, abstract, keywords)
                
                if fully_successful:
                    success_count += 1
                else:
                    partial_success_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"  ✗ Critical error for paper {item_uuid}, continuing to next paper...")
                continue
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
        print(f"Total papers: {len(papers)}")
        print(f"✓ Fully successful: {success_count}")
        print(f"⚠ Partially successful (some fields failed): {partial_success_count}")
        print(f"✗ Critical errors: {error_count}")
        print("="*80)

    def process_new_papers_only(self, limit: int = None):
        """
        Process hanya papers yang belum punya vectors
        Efisien untuk update incremental
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Query untuk ambil paper yang belum ada di vector_store
        query = """
            SELECT pm.item_uuid, pm.title, pm.abstract, pm.keywords
            FROM bot.paper_metadata pm
            LEFT JOIN bot.vector_store vs ON pm.item_uuid = vs.paper_id
            WHERE vs.paper_id IS NULL
        """
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        new_papers = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(new_papers) == 0:
            print("\nNo new papers found. All papers already have vectors!")
            return
        
        print(f"\nFound {len(new_papers)} NEW papers to process")
        print("="*80)
        
        success_count = 0
        partial_success_count = 0
        error_count = 0
        
        for i, (item_uuid, title, abstract, keywords) in enumerate(new_papers, 1):
            print(f"\n[{i}/{len(new_papers)}] Processing new paper: {item_uuid}")
            try:
                fully_successful = self.process_paper(item_uuid, title, abstract, keywords)
                
                if fully_successful:
                    success_count += 1
                else:
                    partial_success_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"  ✗ Critical error for paper {item_uuid}, continuing to next paper...")
                continue
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)
        print(f"Total new papers: {len(new_papers)}")
        print(f"✓ Fully successful: {success_count}")
        print(f"⚠ Partially successful (some fields failed): {partial_success_count}")
        print(f"✗ Critical errors: {error_count}")
        print("="*80)

    def get_statistics(self):
        """Dapatkan statistik dari vector store"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        # Total papers di metadata
        cursor.execute("SELECT COUNT(*) FROM bot.paper_metadata")
        total_papers = cursor.fetchone()[0]
        
        # Total papers yang sudah punya vectors
        cursor.execute("SELECT COUNT(DISTINCT paper_id) FROM bot.vector_store")
        vectorized_papers = cursor.fetchone()[0]
        
        # Total vectors
        cursor.execute("SELECT COUNT(*) FROM bot.vector_store")
        total_vectors = cursor.fetchone()[0]
        
        # Breakdown by chunk_type
        cursor.execute("""
            SELECT chunk_type, COUNT(*) 
            FROM bot.vector_store 
            GROUP BY chunk_type
        """)
        breakdown = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*50)
        print("VECTOR STORE STATISTICS")
        print("="*50)
        print(f"Device: {self.device}")
        print(f"Model: Qwen/Qwen3-Embedding-0.6B")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Total papers in database: {total_papers}")
        print(f"Papers with vectors: {vectorized_papers}")
        print(f"Papers without vectors: {total_papers - vectorized_papers}")
        print(f"Total vector entries: {total_vectors}")
        print("\nBreakdown by type:")
        for chunk_type, count in breakdown:
            print(f"  - {chunk_type}: {count}")
        print("="*50)

    def process_judul_keyword_for_all_papers(self, limit: int = None, regenerate: bool = False):
        """
        Generate judul_keyword untuk semua papers atau yang belum punya
        limit: batasi jumlah paper yang diproses (optional)
        regenerate: jika True, hapus dan generate ulang judul_keyword untuk semua papers
        """
        conn = self.connect_db()
        cursor = conn.cursor()
        
        if regenerate:
            # Ambil semua papers
            query = """
                SELECT item_uuid, title, keywords 
                FROM bot.paper_metadata
            """
            if limit:
                query += f" LIMIT {limit}"
        else:
            # Ambil papers yang belum punya judul_keyword
            query = """
                SELECT pm.item_uuid, pm.title, pm.keywords
                FROM bot.paper_metadata pm
                LEFT JOIN bot.vector_store vs ON pm.item_uuid = vs.paper_id 
                    AND vs.chunk_type = 'judul_keyword'
                WHERE vs.paper_id IS NULL
            """
            if limit:
                query += f" LIMIT {limit}"
        
        cursor.execute(query)
        papers = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if len(papers) == 0:
            print("\nTidak ada papers yang perlu diproses untuk judul_keyword!")
            return
        
        print(f"\nFound {len(papers)} papers untuk generate judul_keyword")
        if regenerate:
            print("MODE: REGENERATE (akan hapus dan buat ulang judul_keyword)")
        else:
            print("MODE: UPDATE (hanya papers yang belum punya judul_keyword)")
        print("="*80)
        
        success_count = 0
        error_count = 0
        
        for i, (item_uuid, title, keywords) in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing judul_keyword for paper: {item_uuid}")
            
            try:
                conn = self.connect_db()
                
                # Hapus judul_keyword lama jika regenerate
                if regenerate:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM bot.vector_store WHERE paper_id = %s AND chunk_type = 'judul_keyword'",
                        (item_uuid,)
                    )
                    cursor.close()
                    print(f"  ↻ Cleared old judul_keyword")
                
                # Generate dan insert judul_keyword
                judul_keyword_text = self.format_judul_keyword(title, keywords)
                judul_keyword_embedding = self.generate_embedding(judul_keyword_text)
                self.insert_vector(conn, item_uuid, 'judul_keyword', judul_keyword_text, 0, judul_keyword_embedding)
                
                conn.commit()
                conn.close()
                
                success_count += 1
                print(f"  ✓ Judul_keyword generated successfully")
                print(f"  Content preview: {judul_keyword_text[:100]}...")
                
            except Exception as e:
                error_count += 1
                print(f"  ✗ Error processing paper {item_uuid}: {str(e)}")
                if 'conn' in locals():
                    conn.rollback()
                    conn.close()
                continue
        
        print("\n" + "="*80)
        print("JUDUL_KEYWORD GENERATION COMPLETE!")
        print("="*80)
        print(f"Total papers processed: {len(papers)}")
        print(f"✓ Success: {success_count}")
        print(f"✗ Errors: {error_count}")
        print("="*80)

    def clear_vectors_for_paper(self, paper_id: str):
        """Hapus semua vectors untuk paper tertentu (jika perlu re-process)"""
        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bot.vector_store WHERE paper_id = %s", (paper_id,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"  ↻ Cleared all vectors for paper {paper_id}")

    def clear_gpu_cache(self):
        """Bersihkan GPU cache jika menggunakan CUDA"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("GPU cache cleared")


# ============= USAGE EXAMPLE =============
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', 5432)
    }
    
    # Initialize vector store dengan Qwen3-Embedding-0.6B
    # device='cuda' untuk force GPU, 'cpu' untuk CPU, None untuk auto-detect
    vector_store = HierarchicalVectorStore(db_config, device='cuda')
    
    # ===== SCENARIO 1: FIRST TIME SETUP =====
    # Process semua papers untuk pertama kali
    # vector_store.process_all_papers()
    
    # ===== SCENARIO 2: UPDATE - HANYA DATA BARU =====
    # Process hanya papers yang belum punya vectors (RECOMMENDED untuk update rutin)
    # vector_store.process_new_papers_only()
    
    # ===== SCENARIO 3: REGENERATE SEMUA =====
    # Hapus dan generate ulang SEMUA vectors dengan Qwen model
    # PENTING: Gunakan ini jika sebelumnya pakai model lain (misal all-MiniLM-L6-v2)
    # vector_store.process_all_papers(regenerate=True)
    
    # ===== SCENARIO 4: TESTING =====
    # Process dengan limit untuk testing
    # vector_store.process_all_papers(limit=5)
    # vector_store.process_new_papers_only(limit=5)
    
    # ===== SCENARIO 5: SINGLE PAPER =====
    # Process satu paper spesifik
    # vector_store.process_paper(
    #     paper_id='uuid-123',
    #     title='Your Paper Title',
    #     abstract='Your abstract here...',
    #     keywords='keyword1, keyword2, keyword3'
    # )
    
    # ===== SCENARIO 6: RE-PROCESS SINGLE PAPER =====
    # Hapus dan re-generate vectors untuk satu paper tertentu
    # vector_store.clear_vectors_for_paper('uuid-123')
    # vector_store.process_paper('uuid-123', 'title', 'abstract', 'keywords')
    
    # ===== SCENARIO 7: GENERATE JUDUL_KEYWORD UNTUK SEMUA PAPERS =====
    # Generate judul_keyword untuk papers yang belum punya
    # vector_store.process_judul_keyword_for_all_papers()
    
    # ===== SCENARIO 8: REGENERATE SEMUA JUDUL_KEYWORD =====
    # Hapus dan generate ulang SEMUA judul_keyword
    # vector_store.process_judul_keyword_for_all_papers(regenerate=True)
    
    # ===== SCENARIO 9: TESTING JUDUL_KEYWORD =====
    # Generate judul_keyword dengan limit untuk testing
    # vector_store.process_judul_keyword_for_all_papers(limit=5)
    
    # ===== SCENARIO 10: CHECK STATISTICS =====
    # Lihat statistik vector store (termasuk judul_keyword)
    vector_store.get_statistics()
    
    # ===== SCENARIO 11: CLEAR GPU CACHE =====
    # Bersihkan GPU cache setelah selesai processing
    # vector_store.clear_gpu_cache()