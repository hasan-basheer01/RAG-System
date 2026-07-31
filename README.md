# RAG System

A simple Retrieval-Augmented Generation (RAG) project that uses a PDF document as a knowledge source and answers user questions using OpenAI embeddings and FAISS vector search.

## Project Overview

This project loads a PDF document, splits it into chunks, creates embeddings, stores them in a FAISS vector database, and then uses those chunks to answer questions based on the document content.

## Features

- Loads a PDF using LangChain
- Splits the document into meaningful chunks
- Creates vector embeddings with OpenAI
- Stores and retrieves relevant chunks using FAISS
- Answers questions using the retrieved context
- Supports LangSmith tracing for observability

## Files

- `app.py` - Main application logic
- `Electric_Circuit_Analysis_Complete_Notes.pdf` - Source PDF used for retrieval
- `.env` - Environment variables for API keys
- `.gitignore` - Ignores sensitive environment files

## Setup

1. Create and activate a virtual environment.
2. Install the required dependencies:
   ```bash
   pip install python-dotenv langchain-community langchain-text-splitters langchain-openai faiss-cpu openai langsmith
   ```
3. Create a `.env` file and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Usage

When the app starts, it will:
- Load the PDF
- Build the FAISS index
- Prompt you to ask questions

Type `exit` to quit the program.

## Notes

- The app expects the PDF file to be present in the project folder.
- Make sure your OpenAI API key is available in the environment before running the app.
- LangSmith tracing is enabled through the installed wrappers.
