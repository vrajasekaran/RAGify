---
title: RAGify
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: A comprehensive RAG toolkit for text processing and document splitting
---

# RAGify

A comprehensive RAG (Retrieval-Augmented Generation) toolkit built with Streamlit and LangChain for text processing, tokenization, and document splitting.

## Features

- **Tokenizer**: Text tokenization tools
- **Text Splitters**: Multiple text splitting strategies including:
  - Character Text Splitter
  - Recursive Character Text Splitter  
  - Markdown Text Splitter
  - PDF processing with Unstructured.io

## Usage

1. Navigate through the different pages using the sidebar
2. Input your text in the text area
3. Adjust chunk size and overlap parameters
4. View the processed chunks in real-time

## Local Development

```bash
# Using Docker Compose
docker-compose up --build

# Manual setup
pip install -r requirements.txt
streamlit run app.py
```

## License

MIT License
