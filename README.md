# LangChain AI Chatbot

## Features
- LLM-based customer support chatbot
- Retrieval-Augmented Generation (RAG) with FAISS
- Backend: FastAPI
- LLM: LLaMA3 via Ollama
- Embeddings: HuggingFace Embeddings

## How to Run
1. Clone repo
2. Create virtual env: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run server: `uvicorn app:app --reload`
5. Test API at `/docs`
