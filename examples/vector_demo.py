from src.vectorstore.vector_store import VectorStore
from src.utils.text_processor import TextProcessor
import numpy as np

def demonstrate_vector_embeddings():
    print("\n=== Vector Embeddings Demo ===")
    # Initialize vector store and text processor
    vector_store = VectorStore()
    text_processor = TextProcessor()
    
    # Sample texts for demonstration
    texts = [
        "Artificial Intelligence is revolutionizing technology",
        "Machine learning algorithms learn from data",
        "Neural networks process information like human brains",
        "Deep learning is a subset of machine learning"
    ]
    
    print("Processing sample texts...")
    # Process and embed texts
    processed_texts = [text_processor.preprocess(text) for text in texts]
    print("\nPreprocessed texts:")
    for original, processed in zip(texts, processed_texts):
        print(f"Original: {original}")
        print(f"Processed: {processed}")
        # Show vector representation
        vector = vector_store.get_embedding(processed)
        print(f"Vector (shape={vector.shape}, first 5 dimensions): {vector[:5]}\n")
    
    # Add texts to vector store and show vector details
    print("Converting texts to vectors...")
    vector_store.add_texts(texts)
    
    # Demonstrate vector similarity search
    query = "How do machines learn?"
    print(f"\nPerforming similarity search for: '{query}'")
    query_vector = vector_store.get_embedding(query)
    print(f"Query vector shape: {query_vector.shape}")
    print(f"Query vector first 5 dimensions: {query_vector[:5]}\n")
    
    similar_texts = vector_store.similarity_search(query, k=2)
    print("Most similar texts:")
    for i, text in enumerate(similar_texts, 1):
        similarity_vector = vector_store.get_embedding(text)
        similarity = np.dot(query_vector, similarity_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(similarity_vector))
        print(f"{i}. {text} (similarity: {similarity:.4f})")
    
    # Demonstrate vector operations
    print("\n=== Vector Operations Demo ===")
    # Get embeddings for specific texts
    text1 = "Artificial Intelligence"
    text2 = "Machine Learning"
    
    vec1 = vector_store.get_embedding(text1)
    vec2 = vector_store.get_embedding(text2)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"\nCosine similarity between '{text1}' and '{text2}': {similarity:.4f}")

def main():
    print("Vector Embeddings and Text Processing Demo\n")
    demonstrate_vector_embeddings()

if __name__ == "__main__":
    main()