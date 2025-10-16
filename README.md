# RAGify

A comprehensive RAG (Retrieval-Augmented Generation) toolkit built with Streamlit and LangChain for text processing, tokenization, and document splitting.

## Features

- **Tokenizer**: Text tokenization tools
- **Text Splitters**: Multiple text splitting strategies including:
  - Character Text Splitter
  - Recursive Character Text Splitter  
  - Markdown Text Splitter
  - PDF processing with Unstructured.io

## Local Development

### Using Docker Compose (Recommended)

```bash
# Build and run the application
docker-compose up --build

# Access the application at http://localhost:7860
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Deployment

### Hugging Face Spaces

This project is configured for deployment to Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Select "Docker" as the SDK
3. Upload your code to the Space
4. The Space will automatically build and deploy using the provided Dockerfile

### Docker Build

```bash
# Build the Docker image
docker build -t ragify .

# Run the container
docker run -p 7860:7860 ragify
```

## Project Structure

```
RAGify/
├── app.py                 # Main Streamlit application
├── BaseCharSplitter.py    # Base character splitter class
├── commondata.py         # Common data and constants
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
├── pages/
│   ├── 1_tokenizer.py   # Tokenizer page
│   ├── splitters/       # Text splitter implementations
│   └── chunk/           # Chunking utilities
└── README.md
```

## Dependencies

- Streamlit for the web interface
- LangChain for text processing
- ChromaDB for vector storage
- Transformers for NLP models
- Unstructured for document processing

## License

This project is open source and available under the MIT License.
