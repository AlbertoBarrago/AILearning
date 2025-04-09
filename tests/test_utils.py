import unittest
import re
from src.utils.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = TextProcessor()

    def test_preprocess(self):
        # Test cases
        test_cases = [
            # Input, Expected Output
            ("Hello World!", "hello world!"),
            ("  Extra  Spaces  ", "extra spaces"),
            ("Special@#$Characters", "specialcharacters"),
            ("Mixed CASE text", "mixed case text"),
            ("", "")
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                result = self.processor.preprocess(input_text)
                self.assertEqual(result, expected_output)

    def test_tokenize(self):
        # Test cases
        test_cases = [
            # Input, Expected Output
            ("hello world", ["hello", "world"]),
            ("this is a test", ["this", "is", "a", "test"]),
            ("single", ["single"]),
            ("", [""]),
            ("multiple   spaces", ["multiple", "", "", "spaces"])
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                result = self.processor.tokenize(input_text)
                self.assertEqual(result, expected_output)


if __name__ == '__main__':
    unittest.main()
