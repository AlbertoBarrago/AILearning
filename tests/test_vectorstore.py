import unittest
import numpy as np
from src.utils import VectorStore
from unittest.mock import patch, MagicMock


class TestVectorStore(unittest.TestCase):

    @patch('src.utils.vectorstore.vector_store.SentenceTransformer')
    def setUp(self, mock_transformer):
        # Setup mock for SentenceTransformer
        self.mock_model = MagicMock()
        mock_transformer.return_value = self.mock_model

        # Initialize VectorStore
        self.vector_store = VectorStore()

    def test_init(self):
        # Test initialization
        self.assertEqual(self.vector_store.texts, [])
        self.assertEqual(self.vector_store.embeddings, [])

    def test_add_texts(self):
        # Setup mock for encode method
        self.mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Add texts
        texts = ["text1", "text2"]
        self.vector_store.add_texts(texts)

        # Assertions
        self.assertEqual(self.vector_store.texts, texts)
        self.mock_model.encode.assert_called_once_with(texts)
        self.assertEqual(len(self.vector_store.embeddings), 2)

    def test_get_embedding(self):
        # Setup mock for encode method
        self.mock_model.encode.return_value = np.array([[0.1, 0.2]])

        # Get embedding
        embedding = self.vector_store.get_embedding("test text")

        # Assertions
        self.mock_model.encode.assert_called_with(["test text"])
        self.assertTrue(isinstance(embedding, np.ndarray))

    @patch('numpy.dot')
    @patch('numpy.linalg.norm')
    def test_similarity_search(self, mock_norm, mock_dot):
        # Setup mocks
        self.mock_model.encode.return_value = np.array([[0.5, 0.5]])
        mock_dot.return_value = 0.75
        mock_norm.return_value = 1.0

        # Add sample texts and embeddings
        self.vector_store.texts = ["text1", "text2", "text3"]
        self.vector_store.embeddings = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6])
        ]

        # Perform similarity search
        results = self.vector_store.similarity_search("query text", k=2)

        # Assertions
        self.assertEqual(len(results), 2)
        self.assertIn(results[0], self.vector_store.texts)
        self.assertIn(results[1], self.vector_store.texts)


if __name__ == '__main__':
    unittest.main()
