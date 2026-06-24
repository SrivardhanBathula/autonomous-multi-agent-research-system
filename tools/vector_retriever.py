from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, embedding_model: str = "text-embedding-3-small", chunk_size: int = 512):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=64)
        self.vectorstore: Optional[FAISS] = None

    def index_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas or [{}] * len(texts))]
        chunks = self.splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info(f"Indexed {len(chunks)} chunks from {len(docs)} documents")

    def retrieve(self, query: str, k: int = 8) -> List[str]:
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def save(self, path: str):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load(self, path: str):
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
