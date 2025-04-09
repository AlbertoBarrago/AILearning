from src.llm.model_handler import LLMHandler
from src.rag.rag_engine import RAGEngine

def demonstrate_llm():
    print("\n=== LLM Demo ===")
    # Initialize LLM
    llm = LLMHandler()
    
    # Show model information
    print("Model Info:", llm.get_model_info())
    
    # Generate a response
    prompt = "Explain what is artificial intelligence in simple terms."
    print("\nPrompt:", prompt)
    response = llm.generate_response(prompt)
    print("Response:", response)

def demonstrate_rag():
    print("\n=== RAG Demo ===")
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Add some sample documents
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        "Machine Learning is a subset of AI that enables systems to learn from data.",
        "Deep Learning is a type of machine learning based on artificial neural networks."
    ]
    
    print("Adding sample documents...")
    rag.add_documents(documents)
    
    # Show system statistics
    print("RAG System Stats:", rag.get_stats())
    
    # Perform a query
    question = "What is the relationship between AI and machine learning?"
    print("\nQuestion:", question)
    answer = rag.query(question)
    print("Answer:", answer)

def main():
    print("AI Learning Project Demo\n")
    
    # Demonstrate LLM capabilities
    demonstrate_llm()
    
    # Demonstrate RAG capabilities
    demonstrate_rag()

if __name__ == "__main__":
    main()