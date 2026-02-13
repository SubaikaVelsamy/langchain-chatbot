from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# Load LLM
llm = Ollama(model="llama3")

# Load documents
loader = TextLoader("faq.txt")
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(question: Question):
    result = qa_chain.run(question.question)
    return {"response": result}

@app.get("/")
def home():
    return {"message": "AI Chatbot API is running"}