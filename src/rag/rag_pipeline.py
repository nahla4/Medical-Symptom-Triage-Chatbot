"""
RAG (Retrieval-Augmented Generation) Pipeline Module for Medical Triage Chatbot.
Loads the preprocessed medical knowledge, builds a vector index of diseases using SentenceTransformers,
and retrieves relevant conditions with metadata (descriptions, precautions, and ICD-10 codes).
Supports both ChromaDB and a pure Python/numpy memory vector store fallback.
"""
import os
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """A pure Python/numpy/scikit-learn memory-based vector database.
    Used as a fallback if ChromaDB fails to build or load under Windows.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = []

    def add_documents(self, texts, metadatas):
        for text, meta in zip(texts, metadatas):
            embedding = self.embedding_model.encode(text)
            self.documents.append({"text": text, "metadata": meta})
            self.embeddings.append(embedding)
        if self.embeddings:
            self.embeddings_matrix = np.array(self.embeddings)

    def similarity_search(self, query, k=5):
        if not self.embeddings:
            return []
        query_vector = self.embedding_model.encode(query)
        # Compute cosine similarity: (A . B) / (||A|| * ||B||)
        dot_product = np.dot(self.embeddings_matrix, query_vector)
        matrix_norms = np.linalg.norm(self.embeddings_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Avoid division by zero
        norms = matrix_norms * query_norm
        norms[norms == 0] = 1e-10
        
        scores = dot_product / norms
        
        # Sort indices by descending score
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx])
            })
        return results

class MedicalRAGPipeline:
    def __init__(self):
        self.knowledge_path = os.path.join("data", "processed", "medical_knowledge.json")
        self.db = None
        self.embedding_model = None
        self.use_chroma = False
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Load knowledge base and populate vector store
        self._init_vector_store()

    def _init_embeddings(self):
        """Attempts to load sentence-transformers embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 model...")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("[SUCCESS] SentenceTransformer loaded.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            # Write a very simple fallback character-based or word-overlap embedder if sentence-transformers is missing
            self.embedding_model = None

    def _init_vector_store(self):
        """Initializes ChromaDB or falls back to SimpleVectorStore."""
        # Load knowledge data
        if not os.path.exists(self.knowledge_path):
            logger.error(f"Processed knowledge base not found at {self.knowledge_path}. Please run preprocess.py first.")
            self.knowledge = {}
            return
            
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            self.knowledge = json.load(f)

        # Prepare documents for indexing
        texts = []
        metadatas = []
        for disease, info in self.knowledge.items():
            # Build indexable text describing the disease and its symptom list
            symptoms_str = ", ".join(info["symptoms"]).replace("_", " ")
            document_text = f"Disease: {disease}. Symptoms: {symptoms_str}. Description: {info['description']}"
            texts.append(document_text)
            metadatas.append(info)

        # Try to use ChromaDB first
        try:
            import chromadb
            from chromadb.config import Settings
            
            logger.info("Initializing ChromaDB vector store...")
            # Persistent client in the workspace
            persist_dir = os.path.join("data", "chroma")
            os.makedirs(persist_dir, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            
            # Custom Chroma Embedding Function wrapper (ChromaDB 1.5+ requires a 'name' attribute)
            class ChromaEmbeddingFunction:
                name = "sentence-transformers-all-MiniLM-L6-v2"

                def __init__(self, model):
                    self.model = model
                def __call__(self, input_texts):
                    # Chroma expects a list of embeddings as list of floats
                    if self.model:
                        embeddings = self.model.encode(input_texts)
                        return [e.tolist() for e in embeddings]
                    else:
                        # Dummy zero embeddings as double-fallback
                        return [[0.0] * 384 for _ in input_texts]

            emb_fn = ChromaEmbeddingFunction(self.embedding_model)
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_knowledge",
                embedding_function=emb_fn
            )
            
            # Add documents in batches (delete collection first to avoid duplicates in dev)
            if self.collection.count() > 0:
                logger.info("Re-populating database with fresh medical data...")
                self.chroma_client.delete_collection(name="medical_knowledge")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="medical_knowledge",
                    embedding_function=emb_fn
                )

            ids = [f"id_{i}" for i in range(len(texts))]
            # Stringify precautions list for Chroma metadata compatibility (it only accepts primitive types)
            chroma_metadatas = []
            for meta in metadatas:
                c_meta = meta.copy()
                c_meta["precautions"] = json.dumps(meta["precautions"])
                c_meta["symptoms"] = json.dumps(meta["symptoms"])
                chroma_metadatas.append(c_meta)
                
            self.collection.add(
                documents=texts,
                metadatas=chroma_metadatas,
                ids=ids
            )
            self.use_chroma = True
            logger.info(f"[SUCCESS] ChromaDB initialized with {len(texts)} entries.")
            
        except Exception as e:
            logger.warning(f"Could not load or initialize ChromaDB: {e}. Falling back to SimpleVectorStore.")
            self.use_chroma = False
            
        # Initialize SimpleVectorStore if ChromaDB is not used
        if not self.use_chroma:
            # If embedding model is also missing, build a mock one
            if self.embedding_model is None:
                class MockEmbeddingModel:
                    def encode(self, texts):
                        # Returns deterministic lists based on character hashing
                        if isinstance(texts, str):
                            texts = [texts]
                        res = []
                        for txt in texts:
                            v = np.zeros(384)
                            for char in txt:
                                v[ord(char) % 384] += 1
                            res.append(v)
                        return res[0] if len(res) == 1 else res
                self.embedding_model = MockEmbeddingModel()
                
            self.db = SimpleVectorStore(self.embedding_model)
            self.db.add_documents(texts, metadatas)
            logger.info(f"[SUCCESS] SimpleVectorStore fallback initialized with {len(texts)} entries.")

    def search_conditions(self, query: str, extracted_symptoms: list = None, k: int = 3) -> list:
        """
        Queries the vector store using a combined query of symptoms and text.
        Returns a list of matching conditions sorted by confidence score.
        """
        # Create a hybrid query
        symptoms_str = " ".join(extracted_symptoms or []).replace("_", " ")
        combined_query = f"{query} {symptoms_str}".strip()
        
        logger.info(f"Querying vector store with: '{combined_query}'")
        
        results = []
        
        if self.use_chroma:
            try:
                raw_results = self.collection.query(
                    query_texts=[combined_query],
                    n_results=k
                )
                
                # Format Chroma query results
                if raw_results and raw_results["documents"]:
                    documents = raw_results["documents"][0]
                    metadatas = raw_results["metadatas"][0]
                    distances = raw_results["distances"][0] if "distances" in raw_results else [0.0]*len(documents)
                    
                    for doc, meta, dist in zip(documents, metadatas, distances):
                        # Convert stringified lists back
                        precautions = json.loads(meta["precautions"]) if isinstance(meta["precautions"], str) else meta["precautions"]
                        symptoms = json.loads(meta["symptoms"]) if isinstance(meta["symptoms"], str) else meta["symptoms"]
                        
                        # Chroma distance can be converted to similarity score (e.g. 1 / (1 + dist))
                        similarity = float(1 / (1 + dist)) if dist is not None else 0.8
                        
                        results.append({
                            "disease_name": meta["disease_name"],
                            "icd10": meta["icd10"],
                            "description": meta["description"],
                            "precautions": precautions,
                            "symptoms": symptoms,
                            "score": round(similarity, 3)
                        })
            except Exception as e:
                logger.error(f"Error querying ChromaDB: {e}. Attempting SimpleVectorStore fallback query.")
                # Temporary dynamic fallback
                self.use_chroma = False
                self._init_vector_store()
                
        if not self.use_chroma:
            # Query SimpleVectorStore
            raw_results = self.db.similarity_search(combined_query, k=k)
            for item in raw_results:
                doc = item["document"]
                results.append({
                    "disease_name": doc["metadata"]["disease_name"],
                    "icd10": doc["metadata"]["icd10"],
                    "description": doc["metadata"]["description"],
                    "precautions": doc["metadata"]["precautions"],
                    "symptoms": doc["metadata"]["symptoms"],
                    "score": round(item["score"], 3)
                })
                
        # Sort results by score descending
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results

if __name__ == "__main__":
    # Test execution
    pipeline = MedicalRAGPipeline()
    res = pipeline.search_conditions("vomiting and high fever", ["vomiting", "high_fever"])
    print(json.dumps(res, indent=2))
