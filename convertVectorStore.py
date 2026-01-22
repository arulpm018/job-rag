import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import re
from dotenv import load_dotenv
import os

load_dotenv()

class HierarchicalVectorStore:
    def __init__(self, db_config: dict):
        """
        Initialize dengan database config
        db_config format:
        {
            'host': 'localhost',
            'database': 'your_db',
            'user': 'your_user',
            'password': 'your_password',
            'port': 5432
        }
        """
        self.db_config = db_config
        # Load Qwen embedding model (1024 dimensions)
        print("Loading Qwen3-Embedding-0.6B model...")
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        self.embedding_dim = 1024  # Qwen3-Embedding-0.6B menghasilkan 1024 dimensi
        print(f"Model loaded successfully! Embedding dimension: {self.embedding_dim}")
        
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
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def insert_vector(self, conn, paper_id: str, chunk_type: str, 
                     content_chunk: str, chunk_index: int, embedding: List[float]):
        """Insert vector ke database"""
        cursor = conn.cursor()
        
        # Convert list to PostgreSQL array format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        query = """
        INSERT INTO bot.vector_store (paper_id, chunk_type, content_chunk, chunk_index, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """
        
        cursor.execute(query, (paper_id, chunk_type, content_chunk, chunk_index, embedding_str))
        cursor.close()
    
    def process_paper(self, paper_id: str, title: str, abstract: str, keywords: str):
        """
        Process satu paper menjadi embeddings
        """
        conn = self.connect_db()
        
        try:
            # 1. Process TITLE
            if title and title.strip():
                print(f"Processing title for paper {paper_id}")
                title_embedding = self.generate_embedding(title)
                self.insert_vector(conn, paper_id, 'title', title, 0, title_embedding)
            
            # 2. Process KEYWORDS
            if keywords and keywords.strip():
                print(f"Processing keywords for paper {paper_id}")
                keywords_embedding = self.generate_embedding(keywords)
                self.insert_vector(conn, paper_id, 'keywords', keywords, 0, keywords_embedding)
            
            # 3. Process ABSTRACT (dengan splitting jika perlu)
            if abstract and abstract.strip():
                abstract_chunks = self.split_abstract(abstract)
                print(f"Processing abstract for paper {paper_id} - {len(abstract_chunks)} chunk(s)")
                
                for idx, chunk in enumerate(abstract_chunks):
                    chunk_embedding = self.generate_embedding(chunk)
                    self.insert_vector(conn, paper_id, 'abstract', chunk, idx, chunk_embedding)
            
            conn.commit()
            print(f"✓ Successfully processed paper {paper_id}")
            
        except Exception as e:
            conn.rollback()
            print(f"✗ Error processing paper {paper_id}: {str(e)}")
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
        print("="*50)
        
        for i, (item_uuid, title, abstract, keywords) in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing paper: {item_uuid}")
            try:
                if regenerate:
                    # Hapus vectors lama terlebih dahulu
                    self.clear_vectors_for_paper(item_uuid)
                self.process_paper(item_uuid, title, abstract, keywords)
            except Exception as e:
                print(f"Skipping paper {item_uuid} due to error")
                continue
        
        print("\n" + "="*50)
        print("Processing complete!")
    
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
        print("="*50)
        
        for i, (item_uuid, title, abstract, keywords) in enumerate(new_papers, 1):
            print(f"\n[{i}/{len(new_papers)}] Processing new paper: {item_uuid}")
            try:
                self.process_paper(item_uuid, title, abstract, keywords)
            except Exception as e:
                print(f"Skipping paper {item_uuid} due to error")
                continue
        
        print("\n" + "="*50)
        print("Processing complete!")
    
    def get_statistics(self):
        """
        Dapatkan statistik dari vector store
        """
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
    
    def clear_vectors_for_paper(self, paper_id: str):
        """Hapus semua vectors untuk paper tertentu (jika perlu re-process)"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM bot.vector_store WHERE paper_id = %s", (paper_id,))
        conn.commit()
        
        cursor.close()
        conn.close()
        print(f"Cleared all vectors for paper {paper_id}")


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
    vector_store = HierarchicalVectorStore(db_config)
    
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
    
    # ===== SCENARIO 7: CHECK STATISTICS =====
    # Lihat statistik vector store
    vector_store.get_statistics()