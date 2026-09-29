import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DeterministicFakeEmbedding

class TelecomRAG:
    """
    Telecom Knowledge Base Retrieval-Augmented Generation module using FAISS.
    """
    def __init__(self, doc_dir: str = "rag/docs"):
        self.doc_dir = doc_dir
        self.vector_store = None
        self.embeddings = DeterministicFakeEmbedding(size=384)

    def build_vector_store(self):
        os.makedirs(self.doc_dir, exist_ok=True)
        documents = []
        
        sample_kb = os.path.join(self.doc_dir, "default_sop.txt")
        if not os.path.exists(sample_kb):
            with open(sample_kb, "w") as f:
                f.write("""
                Telecom Operations SOP & Troubleshooting Guide:
                1. Rectifier Alarm: Check AC mains supply, verify module load sharing, inspect blown fuses, and test controller communication.
                2. DG Overheating: Inspect coolant level, radiator blockage, fan belt tension, and oil filters. Avoid continuous load above 80% if ambient temp > 45C.
                3. Battery Replacement: Disconnect load circuit breakers first, isolate bank, verify string voltage parity before re-energizing.
                4. Low Fuel Mitigation: Refill tank, prime fuel lines to eliminate air locks, clear water separators.
                """)

        for file in os.listdir(self.doc_dir):
            file_path = os.path.join(self.doc_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(splits, self.embeddings)

    def query(self, question: str, k: int = 2) -> str:
        if not self.vector_store:
            self.build_vector_store()
            
        docs = self.vector_store.similarity_search(question, k=k)
        if docs:
            context = "\n---\n".join([d.page_content for d in docs])
            return context
        return "No relevant telecom standard operating procedures found."
