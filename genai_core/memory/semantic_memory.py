from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class SemanticMemory:
    def __init__(self, persist_dir="memory/semantic_store"):
        # Local embedding model — no API key required.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Try loading existing FAISS store
        if (self.persist_dir / "index.faiss").exists():
            self.store = FAISS.load_local(
                folder_path=str(self.persist_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.store = None

    def add(self, text: str):
        """Add text to semantic memory."""
        doc = Document(page_content=text)

        # Initialize FAISS with the first document.
        if self.store is None:
            self.store = FAISS.from_documents(
                [doc],
                self.embeddings,
            )
        else:
            self.store.add_documents([doc])

        # Persist after every update
        self.save()

    def search(self, query: str, k: int = 3):
        """Retrieve semantically similar memory chunks."""
        if self.store is None:
            return []

        results = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def save(self):
        """Persist FAISS index + docstore to disk."""
        if self.store is not None:
            self.store.save_local(str(self.persist_dir))
