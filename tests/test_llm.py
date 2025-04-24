import unittest
from unittest.mock import patch, MagicMock
import torch
from src.utils import LLMHandler


class TestLLMHandler(unittest.TestCase):

    @patch('src.utils.llm.model_handler.AutoTokenizer')
    @patch('src.utils.llm.model_handler.AutoModelForSeq2SeqLM')
    def test_init_and_load_model(self, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # Initialize handler
        handler = LLMHandler(model_name="test-model")

        # Assertions
        self.assertEqual(handler.model_name, "test-model")
        self.assertIn(handler.device, ["cuda", "cpu"])
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once()

    @patch('src.utils.llm.model_handler.AutoTokenizer')
    @patch('src.utils.llm.model_handler.AutoModelForSeq2SeqLM')
    def test_get_model_info(self, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # Initialize handler
        handler = LLMHandler()

        # Get model info
        info = handler.get_model_info()

        # Assertions
        self.assertIsInstance(info, dict)
        self.assertIn("model_name", info)
        self.assertIn("device", info)
        self.assertIn("model_loaded", info)
        self.assertTrue(info["model_loaded"])

    @patch('src.utils.llm.model_handler.AutoTokenizer')
    @patch('src.utils.llm.model_handler.AutoModelForSeq2SeqLM')
    def test_generate_response(self, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.decode.return_value = "Test response"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Initialize handler
        handler = LLMHandler()

        # Mock the decode method to return a test response
        handler.tokenizer.decode = MagicMock(return_value="Test response")

        # Generate response
        response = handler.generate_response("Test prompt")

        # Assertions
        self.assertIsInstance(response, str)
        mock_model_instance.generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
