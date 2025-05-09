# AI Learning Project

[![Run Tests](https://github.com/AlbertoBarrago/AI_Learning/actions/workflows/run-tests.yml/badge.svg)](https://github.com/AlbertoBarrago/AI_Learning/actions/workflows/run-tests.yml)

This project is designed to help you learn fundamental concepts of Artificial Intelligence,
focusing on Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).
The project provides hands-on experience with popular AI tools and frameworks.

## Features

- LLM Integration using Hugging Face Transformers
- RAG Implementation for document retrieval and Q&A
- Vector Store management
- Document processing capabilities
- Neural Heat-Map 

## Screenshots
![img.png](img.png)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or setup.py install

   ```bash
   pip setup.py install
   ```

## Learning Path

1. **LLM Basics**
   - Understanding transformer architecture
   - Using pre-trained models
   - Fine-tuning for specific tasks

2. **RAG Implementation**
   - Document processing
   - Vector embeddings
   - Retrieval mechanisms
   - Question answering

3. **Advanced Topics**
   - Custom model training
   - Performance optimization

## Usage

Check the `examples/` directory for practical demonstrations of each concept.

## Testing

Run unit tests:
```bash
python -m unittest discover tests/
```

## Contributing

Feel free to contribute by:
- Adding new examples
- Improving documentation
- Fixing bugs
- Suggesting enhancements

## License

MIT License