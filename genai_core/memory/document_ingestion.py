from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class DocumentIngestion:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None   # Lazy initialization


    def load(self, file_path):
        """Load a document from disk and ingest it into FAISS."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = Document(page_content=text)

        # First document → initialize FAISS
        if self.store is None:
            self.store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.store.add_documents([doc])

        return f"Document '{file_path}' ingested."


    def search(self, query, k=3):
        """Retrieve semantically similar document chunks."""
        if self.store is None:
            return []

        return self.store.similarity_search(query, k=k)
