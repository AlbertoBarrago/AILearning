import unittest
from unittest.mock import MagicMock


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        # Import inside the test to avoid circular import issues
        from src.utils.rag.rag_engine import RAGEngine

        # Create mocks
        self.mock_index = MagicMock()
        self.mock_llm = MagicMock()

        # Initialize RAGEngine with mocks
        self.rag = RAGEngine()
        self.rag.index = self.mock_index
        self.rag.llm = self.mock_llm

    def test_init(self):
        # Test initialization
        self.assertIsNotNone(self.rag.index)
        self.assertIsNotNone(self.rag.llm)
        self.assertEqual(self.rag.documents, [])


    def test_get_stats(self):
        # Setup
        self.rag.documents = ["doc1", "doc2"]

        # Get stats
        stats = self.rag.get_stats()

        # Assertions
        self.assertIsInstance(stats, dict)
        self.assertIn("document_count", stats)
        self.assertEqual(stats["document_count"], 2)


if __name__ == '__main__':
    unittest.main()
