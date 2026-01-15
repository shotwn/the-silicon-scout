import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
import argparse
import torch

class RAGEngine:
    def __init__(self, persist_directory="./rag_db", model_name="nomic-ai/nomic-embed-text-v1.5"):
        print("--- Initializing RAG (ChromaDB) ---")
        
        # Pick device to run embeddings on
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        print(f"Using device for embeddings: {device}")
        # This model is small (~80MB) and runs fast on system RAM.
        self.embed_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        # Initialize ChromaDB (Persistent)
        self.client = chromadb.PersistentClient(path=persist_directory, settings=chromadb.config.Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="physics_knowledge")

    def _custom_text_splitter(self, text, chunk_size=2000, overlap=200):
        """
        A simple custom text splitter that splits text into chunks of specified size with overlap.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap (last 100 chars of previous)
                current_chunk = current_chunk[-overlap:] + para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def ingest_files(self, articles_dir="knowledge_base", tag="knowledge"):
        import pymupdf4llm 
        
        if not os.path.exists(articles_dir):
            print(f"Directory {articles_dir} not found.")
            return

        print(f"--- Indexing documents in {articles_dir} ---")
        
        for filename in os.listdir(articles_dir):
            # Check if file is already in DB to save time
            file_path = os.path.join(articles_dir, filename)
            current_mtime = os.path.getmtime(file_path)

            # Check if it is a folder
            if os.path.isdir(file_path):
                print(f"Skipping directory {filename}.")
                continue

            existing = self.collection.get(where={"source": filename})
            
            # Check if file is new OR modified since last index
            if existing['ids']:
                stored_mtime = existing['metadatas'][0].get('mtime', 0)
                if current_mtime <= stored_mtime:
                    continue # Truly skip only if unchanged
                else:
                    print(f"File {filename} modified. Re-indexing...")
                    self.collection.delete(where={"source": filename}) # Clear old chunks
            
            try:
                print(f"Processing {filename}...")
                # Extract Text
                if filename.lower().endswith(('.txt', '.md')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                elif filename.lower().endswith(('.pdf', '.docx')):
                    text_content = pymupdf4llm.to_markdown(file_path)
                else:
                    print(f"Unsupported file format for {filename}. Skipping.")
                    continue
                
                # Split Text (Custom function)
                chunks = self._custom_text_splitter(text_content)

                # Nomic requires this prefix for the embedding model
                prefixed_chunks = ["search_document: " + c for c in chunks]
                
                # Encode the PREFIXED text
                embeddings = self.embed_model.encode(
                    prefixed_chunks, 
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
                
                # Store in Chroma
                # Documents should be the ORIGINAL (clean) text: chunks
                ids = [str(uuid.uuid4()) for _ in chunks]
                metadatas = [{"source": filename, "mtime": os.path.getmtime(file_path), "tag": tag} for _ in chunks]
                
                self.collection.add(
                    documents=chunks,     
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Indexed {filename}: {len(chunks)} chunks.")
                
            except Exception as e:
                print(f"Failed to index {filename}: {e}")

    def query(self, query_text, n_results=3, exclude_tags=[]):
        # Embed the user query on the selected device
        search_query = f"search_query: {query_text}"
        
        query_embedding = self.embed_model.encode(
            [search_query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).tolist()

        # Apply tag filtering if needed
        if exclude_tags:
            where_filter = {"tag": {"$nin": exclude_tags}}
        else:
            where_filter = None
        
        # Search Chroma
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter
        )

        # Filter weak matches
        valid_docs = []
        for i, dist in enumerate(results['distances'][0]):
            print(f"Distance for doc {i}: {dist}")
            # Threshold depends on model/metric (L2 vs Cosine)
            # With normalized embeddings, L2 distance ranges from 0 (identical) to ~1.414 (opposite)
            if dist < 0.8: 
                valid_docs.append(results['documents'][0][i])
        
        if not valid_docs:
            return "No confident matches found in knowledge base."
        
        # Format results for the LLM
        retrieved_text = ""
        if valid_docs:
            for i, doc in enumerate(valid_docs):
                source = results['metadatas'][0][i]['source']
                retrieved_text += f"\n ## Context from {source}\n{doc}\n"
        
        return retrieved_text if retrieved_text else "No relevant information found in knowledge base."
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine Ingestion and Querying")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from the articles directory")
    parser.add_argument("--query", type=str, help="Query string to search the knowledge base")
    args = parser.parse_args()
    rag_engine = RAGEngine()
    if args.ingest:
        rag_engine.ingest_files()  # Ingest documents from 'articles' directory

        if os.path.exists("knowledge_base/gemma"):
            rag_engine.ingest_files(articles_dir="knowledge_base/gemma", tag="gemma")

    if args.query:
        response = rag_engine.query(args.query, exclude_tags=["gemma"])
        print(f"<tool_result>\n{response}\n</tool_result>")