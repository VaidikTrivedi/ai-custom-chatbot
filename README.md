# Document-based Chatbot with NLP

This is a custom machine learning model that can read documents and act as a chatbot, answering questions based on the provided documents using Natural Language Processing (NLP).

## Features

- Document processing and storage
- Weblink processing
- Natural Language Understanding
- Question-Answering capabilities
- Semantic search for relevant context
- Command-line interface for interaction

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- NLTK
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Add data source:
- Provide path to the documents that you want to process (it works with txt and pdf files)
- Provide web-links to the articles that you want to add (it might now work if website has a bot detection)

2. Run the chat interface:
```bash
python chat_interface.py
```
3. Ask questions

## How it Works

The chatbot uses state-of-the-art NLP models:
- RoBERTa for question answering
- Sentence transformers for semantic search
- FAISS for efficient similarity search
- NLTK for text processing

The system processes documents by:
1. Breaking them into sentences
2. Creating embeddings for semantic search
3. Using relevant context to answer questions
4. Providing confidence scores for answers

## Notes

- The first run will download the required models
- Performance depends on the quality and relevance of provided documents
- Confidence scores indicate the reliability of answers 