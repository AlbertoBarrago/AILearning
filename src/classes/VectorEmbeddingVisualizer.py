from src.utils.vectorstore.vector_store import VectorStore
from src.utils.text_processor import TextProcessor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.animation as animation
from sklearn.manifold import TSNE

class VectorEmbeddingVisualizer:
    """
    Class to visualize vector embeddings of textual data using various methods.

    This class offers functionality to process text data, compute their vector
    representations (embeddings), perform similarity searches, and create visualizations.
    It is particularly useful for exploring relationships between text data and understanding the
    embedding space through dimensionality reduction techniques such as PCA and t-SNE.

    :ivar vector_store: Instance of VectorStore to handle vector
        embeddings and similarity computations.
    :type vector_store: VectorStore
    :ivar text_processor: Instance of TextProcessor for text preprocessing.
    :type text_processor: TextProcessor
    :ivar sample_texts: List of sample input texts to demonstrate embeddings
        and visualizations.
    :type sample_texts: list of str
    :ivar query: Input query text used for similarity search.
    :type query: str
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.text_processor = TextProcessor()
        self.sample_texts = [
            "Artificial Intelligence is revolutionizing technology",
            "Machine learning algorithms learn from data",
            "Neural networks process information like human brains",
            "Deep learning is a subset of machine learning",
            "Python is a programming language",
            "Natural language processing understands text",
            "Computer vision analyzes images and videos",
            "The silence is gold",
            "My name is Jhon"
        ]
        self.query = "How do machines learn?"

    def demonstrate_vector_embeddings(self):
        """
        Demonstrate vector embeddings and text processing.
        """
        print("\n=== Vector Embeddings Demo ===")
        self._process_and_display_texts()
        self.perform_similarity_search()

        # Create separate figures for animations and heatmap
        plt.ion()  # Turn on interactive mode
        self.visualize_embeddings_multi()
        self.visualize_similarity_matrix()
        plt.ioff()  # Turn off interactive mode

        # Show all plots
        plt.show(block=True)

    def _process_and_display_texts(self):
        """Process sample texts and display their embeddings."""
        print("Processing sample texts...")
        processed_texts = [self.text_processor.preprocess(text) for text in self.sample_texts]
        print("\nPreprocessed texts:")
        for original, processed in zip(self.sample_texts, processed_texts):
            print(f"Original: {original}")
            print(f"Processed: {processed}")
            vector = self.vector_store.get_embedding(processed)
            print(f"Vector (shape={vector.shape[0]}, first 5 dimensions): {vector[:5]}\n")

        print("Converting texts to vectors...")
        self.vector_store.add_texts(self.sample_texts)

    def perform_similarity_search(self) -> None:
        """
        Perform similarity search for the query text and display results.

        This method:
        1. Retrieve the vector embedding
        2. Performs similarity search to find most similar texts
        3. Calculates and displays cosine similarity scores
        """
        print(f"\nPerforming similarity search for: '{self.query}'")

        # Get and display query vector information
        query_vector = self.vector_store.get_embedding(self.query)
        self._display_vector_info(query_vector)

        # Perform similarity search
        similar_texts = self.vector_store.similarity_search(self.query, k=2)
        print("Most similar texts:")

        # Display results with similarity scores
        for i, text in enumerate(similar_texts, 1):
            similarity_vector = self.vector_store.get_embedding(text)
            similarity_score = self._calculate_cosine_similarity(query_vector, similarity_vector)
            print(f"{i}. {text} (similarity: {similarity_score:.4f})")

    @staticmethod
    def _calculate_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    @staticmethod
    def _display_vector_info(vector: np.ndarray) -> None:
        """
        Display basic information about a vector.

        Args:
            vector: Vector to display information about
        """
        print(f"Query vector shape: {vector.shape[0]}")
        print(f"Query vector first 5 dimensions: {vector[:5]}\n")

    def visualize_embeddings_multi(self):
        """Create multiple animated visualizations using different techniques."""
        print("\n=== Multiple Visualization Methods ===")

        # Prepare data
        all_texts = self.sample_texts + [self.query]
        all_embeddings = [self.vector_store.get_embedding(text) for text in all_texts]
        all_embeddings_array = np.vstack(all_embeddings)

        # Setup visualization
        fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 4])

        self._add_description(fig, gs)
        ax1, ax2 = self._setup_visualization_axes(fig, gs)

        # Generate visualizations
        pca_result = self._get_pca_visualization(all_embeddings_array)
        tsne_result = self._get_tsne_visualization(all_embeddings_array)

        # Create animation
        ani = self._create_animation(fig, ax1, ax2, pca_result, tsne_result, all_embeddings)
        ani.save('vector_visualization_multi.gif', writer='pillow', fps=1)
        print("Animation saved as 'vector_visualization_multi.gif'")
        plt.show()

    def _get_tsne_visualization(self, embeddings_array):
        """Generate t-SNE visualization."""
        n_samples = len(self.sample_texts) + 1
        perplexity = min(n_samples - 1, 5)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        return tsne.fit_transform(embeddings_array)

    def _create_animation(self, fig, ax1, ax2, pca_result, tsne_result, all_embeddings):
        """Create animation for the visualizations."""

        def init():
            ax1.clear()
            ax2.clear()
            ax1.set_title('PCA Visualization\n(Global Structure)')
            ax2.set_title('t-SNE Visualization\n(Local Relationships)')
            return []

        def animate(frame):
            return self._animate_frame(frame, ax1, ax2, pca_result, tsne_result, all_embeddings)

        return animation.FuncAnimation(fig, animate, frames=4, init_func=init,
                                       interval=2000, blit=True, repeat=False)

    def _plot_frame_data(self, frame, ax, data, all_embeddings):
        """Plot data for a single frame."""
        # Plot text points
        ax.scatter(data[:-1, 0], data[:-1, 1], c='blue', alpha=0.7, label='Texts')

        # Add text labels
        for i, txt in enumerate(self.sample_texts):
            ax.annotate(txt[:20] + "...", (data[i, 0], data[i, 1]), fontsize=8)

        # Animate query point appearance
        if frame > 0:
            ax.scatter(data[-1, 0], data[-1, 1], c='red', s=100, label='Query')
            ax.annotate(f"QUERY: {self.query}", (data[-1, 0], data[-1, 1]),
                        fontsize=9, color='red', weight='bold')

        # Draw similarity lines
        if frame > 1:
            query_point = data[-1]
            for j, point in enumerate(data[:-1]):
                similarity = np.dot(all_embeddings[-1], all_embeddings[j]) / (
                        np.linalg.norm(all_embeddings[-1]) * np.linalg.norm(all_embeddings[j]))

                # Only draw lines to the two most similar texts
                if j < 2:
                    ax.plot([query_point[0], point[0]], [query_point[1], point[1]],
                            'r-', alpha=min(similarity, 1.0), linewidth=similarity * 3)
                    ax.annotate(f"{similarity:.2f}",
                                ((query_point[0] + point[0]) / 2, (query_point[1] + point[1]) / 2),
                                fontsize=8, color='red')

        ax.legend()

    def _animate_frame(self, frame, ax1, ax2, pca_result, tsne_result, all_embeddings):
        """Animate a single frame."""
        self._clear_and_set_titles(ax1, ax2)

        for data, ax in [(pca_result, ax1), (tsne_result, ax2)]:
            self._plot_frame_data(frame, ax, data, all_embeddings)

        plt.tight_layout()
        return []

    def visualize_similarity_matrix(self):
        """Create a heatmap of similarities between texts."""
        embeddings = [self.vector_store.get_embedding(text) for text in self.sample_texts]
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        self._plot_similarity_heatmap(similarity_matrix)

    def _calculate_similarity_matrix(self, embeddings):
        """Calculate similarity matrix between embeddings."""
        n_texts = len(self.sample_texts)
        similarity_matrix = np.zeros((n_texts, n_texts))

        for i in range(n_texts):
            for j in range(n_texts):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                similarity_matrix[i, j] = similarity

        return similarity_matrix

    def _plot_similarity_heatmap(self, similarity_matrix):
        """Plot similarity matrix heatmap using matplotlib."""
        # Create a new figure with a specific number
        fig = plt.figure(num='Similarity Matrix', figsize=(10, 8))
        im = plt.imshow(similarity_matrix, cmap='YlOrRd')

        # Add colorbar
        plt.colorbar(im)

        # Add text annotations
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                ha='center', va='center')

        # Set labels
        plt.xticks(range(len(self.sample_texts)),
                   [t[:20] + "..." for t in self.sample_texts],
                   rotation=45, ha='right')
        plt.yticks(range(len(self.sample_texts)),
                   [t[:20] + "..." for t in self.sample_texts])

        plt.title('Text Similarity Matrix')
        plt.tight_layout()

    @staticmethod
    def _add_description(fig, gs):
        """Add description text to the visualization."""
        desc_ax = fig.add_subplot(gs[0, :])
        desc_ax.axis('off')
        description = """Visualization of Text Embeddings:
        - Left: PCA reduces high-dimensional data to 2D while preserving global structure
        - Right: t-SNE focuses on preserving local relationships between similar texts
        - Blue dots: Original text embeddings
        - Red dot: Query text embedding
        - Red lines: Connections to most similar texts (thickness indicates similarity)
        - Numbers on lines: Cosine similarity scores (1.0 = identical, 0.0 = unrelated)"""
        desc_ax.text(0, 0.5, description, fontsize=10, va='center', ha='left', wrap=True)

    @staticmethod
    def _setup_visualization_axes(fig, gs):
        """Set up the visualization axes."""
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        return ax1, ax2

    @staticmethod
    def _get_pca_visualization(embeddings_array):
        """Generate PCA visualization."""
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings_array)

    @staticmethod
    def _clear_and_set_titles(ax1, ax2):
        """Clear axes and set titles and labels."""
        ax1.clear()
        ax2.clear()
        ax1.set_title('PCA Visualization\n(Global Structure)')
        ax2.set_title('t-SNE Visualization\n(Local Relationships)')
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')