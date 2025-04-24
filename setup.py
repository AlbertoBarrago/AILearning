from setuptools import setup, find_packages

setup(
    name="ai_learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers==4.48.0",
        "torch>=2.1.2",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "langchain>=0.1.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.2",
        "pandas>=2.1.4",
        "fastapi>=0.109.0",
        "uvicorn>=0.25.0",
        "pyPDF2>=3.0.1",
        "docx2txt>=0.8",
        "matplotlib~=3.10.1",
        "scikit-learn~=1.6.1",
        "setuptools~=79.0.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
)