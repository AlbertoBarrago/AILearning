from src.llm.model_handler import LLMHandler
from src.rag.rag_engine import RAGEngine

import os



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
    
    # Read documents from the data folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(('.txt', '.md', '.pdf')):  # Add more extensions if needed
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    print(f"Loading {len(documents)} documents from data folder...")
    
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