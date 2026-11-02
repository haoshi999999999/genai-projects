from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class SemanticMemory:
    def __init__(self, persist_dir=None):
        # Local embedding model — no API key required.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Resolve persistence directory relative to this Python file.
        if persist_dir is None:
            self.persist_dir = Path(__file__).resolve().parent / "semantic_store"
        else:
            self.persist_dir = Path(persist_dir)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.persist_dir / "index.faiss"
        self.pickle_file = self.persist_dir / "index.pkl"

        self.store = None

        self._load_store()

    def _load_store(self):
        """Load an existing FAISS store if valid."""

        if not self.index_file.exists() and not self.pickle_file.exists():
            print("No existing semantic memory found. Starting with an empty store.")
            return

        if not self.index_file.exists() or not self.pickle_file.exists():
            raise RuntimeError(
                f"Incomplete FAISS store found at {self.persist_dir}. "
                "Both index.faiss and index.pkl are required. "
                "Back up the directory and rebuild the index."
            )

        try:
            self.store = FAISS.load_local(
                folder_path=str(self.persist_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("Existing FAISS semantic memory loaded successfully.")

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAISS semantic memory from "
                f"{self.persist_dir}. The index may be corrupted. "
                "Back up the directory before rebuilding."
            ) from exc

    def add(self, text: str):
        """Add text to semantic memory."""

        if not text or not text.strip():
            return

        doc = Document(page_content=text)

        if self.store is None:
            self.store = FAISS.from_documents(
                [doc],
                self.embeddings
            )
        else:
            self.store.add_documents([doc])

        self.save()

    def search(self, query: str, k: int = 3):
        """Retrieve semantically similar memory chunks."""

        if self.store is None:
            return []

        results = self.store.similarity_search(query, k=k)

        return [doc.page_content for doc in results]

    def save(self):
        """Persist FAISS index and document store."""

        if self.store is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.persist_dir))

