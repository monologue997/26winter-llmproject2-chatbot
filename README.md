# Streamlit AI Chatbot

A conversational chatbot built with Streamlit and OpenAI GPT-3.5-turbo for UW 596A Mini Project 2.

## Project Structure
```text
├── app.py                          # Part 2: Streamlit chatbot
├── solution.ipynb                  # Part 1: RAG pipeline implementation
├── Mini Project 2 Part 1 and 2.ipynb  # Assignment notebook
├── report.md                       # Part 2 test report
├── docs/
│   └── machine-learning.pdf        # Reference document for RAG
├── screenshots/
│   ├── test_screenshoot.png        # Chatbot conversation test
│   └── test_screenshoot_2.png      # Domain knowledge query test
└── .gitignore
```
## Features

- **Part 1**: RAG (Retrieval-Augmented Generation) pipeline using Pinecone vector database and OpenAI embeddings
- **Part 2**: Interactive chatbot with conversation history and context memory

## Setup

1. Install dependencies:
   ```bash
   pip install streamlit openai
2. Create .streamlit/secrets.toml:
   ```bash
   OPENAI_API_KEY = "sk-your-key-here"
3. Run:
   ```bash
   streamlit run app.py

