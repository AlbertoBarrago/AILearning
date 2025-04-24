from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# Set tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMHandler:
    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initialize the LLM handler with a specified model.
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load the model and tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def generate_response(self, prompt, max_length=150):
        """
        Generate a response for the given prompt
        Args:
            prompt (str): Input text
            max_length (int): Maximum length of generated response
        Returns:
            str: Generated response
        """
        # Enhanced system prompt for better response quality
        system_prompt = (
            "You are a highly knowledgeable AI assistant that provides detailed, accurate, and well-structured answers. "
            "Your responses should be comprehensive yet clear, using examples where appropriate. "
            "Always maintain a professional and informative tone."
        )
        enhanced_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nProvide a detailed, well-organized answer with relevant examples where applicable:"
        
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return "Error: Model not loaded"

        try:
            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(
                input_ids=inputs.input_ids, # Added input_ids
                attention_mask=inputs.attention_mask, # Added attention mask
                max_length=max(500, max_length),  # Further increased for more comprehensive responses
                min_length=50,  # Ensure minimum response length
                num_return_sequences=1, # Generate a single response
                temperature=0.7,  # Adjusted for a better balance between creativity and coherence
                top_p=0.95,  # Slightly increased nucleus sampling for better quality
                top_k=50,  # Add top-k sampling for more focused responses
                do_sample=True, # Enable sampling for more diverse responses
                repetition_penalty=1.3,  # Increased to further prevent repetition
                no_repeat_ngram_size=3  # Prevent repetition of 3-grams
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_model_info(self):
        """
        Get information about the loaded model
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None
        }