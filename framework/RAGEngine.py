import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, persist_directory="./rag_db", model_name="all-MiniLM-L6-v2"):
        print("--- Initializing RAG (ChromaDB + CPU Embeddings) ---")
        
        # 1. Force Embedding Model to CPU (Saves 8GB VRAM for Qwen)
        # This model is small (~80MB) and runs fast on system RAM.
        self.embed_model = SentenceTransformer(model_name, device='cpu')
        
        # 2. Initialize ChromaDB (Persistent)
        self.client = chromadb.PersistentClient(path=persist_directory, settings=chromadb.config.Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="physics_knowledge")

    def _custom_text_splitter(self, text, chunk_size=1000, overlap=100):
        """
        A simple sliding window splitter to replace LangChain.
        Splits text into chunks of `chunk_size` characters with `overlap`.
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Calculate end index
            end = start + chunk_size
            
            # Slice the text
            chunk = text[start:end]
            
            # Only add if the chunk has substantial content
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
            
            # Move the window forward, but step back by overlap
            start += (chunk_size - overlap)
            
            # Safety break for edge cases
            if start >= text_len:
                break
                
        return chunks

    def ingest_files(self, articles_dir="articles"):
        import pymupdf4llm 
        
        if not os.path.exists(articles_dir):
            print(f"Directory {articles_dir} not found.")
            return

        print(f"--- Indexing documents in {articles_dir} ---")
        
        for filename in os.listdir(articles_dir):
            # Check if file is already in DB to save time
            existing = self.collection.get(where={"source": filename})
            if existing['ids']:
                continue # Skip already indexed files

            file_path = os.path.join(articles_dir, filename)
            try:
                print(f"Processing {filename}...")
                # 1. Extract Text
                text_content = pymupdf4llm.to_markdown(file_path)
                
                # 2. Split Text (Custom function)
                chunks = self._custom_text_splitter(text_content)
                
                # 3. Generate Embeddings on CPU
                # converting to list is required for Chroma
                embeddings = self.embed_model.encode(chunks, convert_to_numpy=True).tolist()
                
                # 4. Store in Chroma
                ids = [str(uuid.uuid4()) for _ in chunks]
                metadatas = [{"source": filename} for _ in chunks]
                
                self.collection.add(
                    documents=chunks,
                    embeddings=embeddings, # We provide CPU embeddings explicitly
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Indexed {filename}: {len(chunks)} chunks.")
                
            except Exception as e:
                print(f"Failed to index {filename}: {e}")

    def query(self, query_text, n_results=3):
        # 1. Embed the user query on CPU
        query_embedding = self.embed_model.encode([query_text], convert_to_numpy=True).tolist()
        
        # 2. Search Chroma
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 3. Format results for the LLM
        retrieved_text = ""
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                source = results['metadatas'][0][i]['source']
                retrieved_text += f"\n--- Context from {source} ---\n{doc}\n"
        
        return retrieved_text if retrieved_text else "No relevant information found in knowledge base."