from src.vectorstore.vector_store import VectorStore
from src.utils.text_processor import TextProcessor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

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
        print(f"Vector (shape={vector.shape[0]}, first 5 dimensions): {vector[:5]}\n")
    
    # Add texts to vector store and show vector details
    print("Converting texts to vectors...")
    vector_store.add_texts(texts)
    
    # Demonstrate vector similarity search
    query = "How do machines learn?"
    print(f"\nPerforming similarity search for: '{query}'")
    query_vector = vector_store.get_embedding(query)
    print(f"Query vector shape: {query_vector.shape[0]}")
    print(f"Query vector first 5 dimensions: {query_vector[:5]}\n")
    
    similar_texts = vector_store.similarity_search(query, k=2)
    print("Most similar texts:")
    for i, text in enumerate(similar_texts, 1):
        similarity_vector = vector_store.get_embedding(text)
        similarity = np.dot(query_vector, similarity_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(similarity_vector))
        print(f"{i}. {text} (similarity: {similarity:.4f})")
    
    # Visualize the embeddings
    visualize_embeddings(vector_store, texts, query)

def visualize_embeddings(vector_store, texts, query):
    """
    Create an animated visualization of vector embeddings
    """
    print("\n=== Visualizing Vector Embeddings ===")

    # Get all embeddings including the query
    all_texts = texts + [query]
    all_embeddings = [vector_store.get_embedding(text) for text in all_texts]

    # Use PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for different points
    colors = ['blue', 'blue', 'blue', 'blue', 'red']

    # Create animation
    def init():
        ax.clear()
        ax.set_title('Vector Embeddings Visualization')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        return []

    def animate(i):
        ax.clear()
        ax.set_title('Vector Embeddings in 2D Space')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')

        # Plot all points
        for j, (x, y) in enumerate(embeddings_2d):
            if j < len(texts):
                ax.scatter(x, y, color=colors[j], alpha=0.7)
                ax.annotate(texts[j][:20] + "...", (x, y), fontsize=9)
            else:
                # This is the query point
                if i > 0:  # Only show query after first frame
                    ax.scatter(x, y, color=colors[j], s=100, alpha=0.9)
                    ax.annotate(f"QUERY: {query}", (x, y), fontsize=10, weight='bold')

        # Draw lines from query to other points if query is visible
        if i > 1:
            query_point = embeddings_2d[-1]
            for j, point in enumerate(embeddings_2d[:-1]):
                # Calculate similarity for line thickness
                similarity = np.dot(all_embeddings[-1], all_embeddings[j]) / (
                    np.linalg.norm(all_embeddings[-1]) * np.linalg.norm(all_embeddings[j]))

                # Only draw lines to the two most similar texts
                if j < 2:  # Assuming the first two are the most similar
                    ax.plot([query_point[0], point[0]], [query_point[1], point[1]],
                            'r-', alpha=min(similarity, 1.0),
                            linewidth=similarity*5)
                    ax.annotate(f"{similarity:.4f}",
                                ((query_point[0] + point[0])/2, (query_point[1] + point[1])/2),
                                fontsize=9, color='red')

        return []

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=4, init_func=init, interval=3000, blit=True)

    # Save animation
    ani.save('vector_visualization.gif', writer='pillow', fps=1)
    print("Animation saved as 'vector_visualization.gif'")

    # Show the plot
    plt.tight_layout()
    plt.show()

def main():
    print("Vector Embeddings and Text Processing Demo\n")
    demonstrate_vector_embeddings()

if __name__ == "__main__":
    main()