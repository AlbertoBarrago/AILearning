from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from src.utils.llm.model_handler import LLMHandler


class RAGEngine:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the RAG engine with embedding model and vector store
        Args:
            embedding_model (str): Name of the sentence transformer model
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.llm = LLMHandler()
        self.index = None
        self.documents = []
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2

    def initialize_index(self):
        """
        Initialize the FAISS index for vector storage
        """
        self.index = faiss.IndexFlatL2(self.dimension)

    def add_documents(self, documents: List[str]):
        """
        Add documents to the RAG system
        Args:
            documents (List[str]): List of document texts to add
        """
        if self.index is None:
            self.initialize_index()

        # Generate embeddings for documents
        embeddings = self.embedding_model.encode(documents)

        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(documents)

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on the query
        Args:
            query (str): Search a query
            k (int): Number of results to return
        Returns:
            List[Dict]: List of relevant documents with scores
        """
        if self.index is None or len(self.documents) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search in FAISS index
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)

        results = []
        for i, idx in enumerate(indices[0]):
            # Convert idx to int to ensure proper comparison
            idx_int = int(idx)
            if idx_int < len(self.documents):
                results.append({
                    "document": self.documents[idx_int],
                    "score": float(distances[0][i])
                })

        return results

    def query(self, question: str, k: int = 3) -> str:
        """
        Perform RAG query: retrieve relevant documents and generate answer
        Args:
            question (str): User question
            k (int): Number of documents to retrieve
        Returns:
            str: Generated answer
        """
        # Retrieve relevant documents
        relevant_docs = self.search(question, k)

        # Process and structure retrieved context
        formatted_context = "\n".join([f"Reference {i + 1}:\n{doc['document']}" for i, doc in enumerate(relevant_docs)])

        # Construct enhanced prompt with better instructions
        prompt = f"""Based on the following references, provide a comprehensive and accurate answer.

        References:
        {formatted_context}
        
        Question: {question}
        
        Provide a detailed answer that:
        1. Directly addresses the question
        2. Uses information from the references
        3. Explains concepts clearly and thoroughly
        
        Answer:"""

        # Generate answer using LLM
        response = self.llm.generate_response(prompt)

        return response

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system
        Returns:
            Dict: Statistics about the system
        """
        return {
            "document_count": len(self.documents),
            "embedding_model": self.dimension,
            "index_initialized": self.index is not None
        }
