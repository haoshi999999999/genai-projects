from pathlib import Path

#from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIngestion:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None

        # Keep the document vector index separate from conversational memory.
        self.persist_directory = Path(__file__).resolve().parent / "document_store"

        # Split documents into smaller, overlapping chunks.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Load an existing FAISS index if one is available.
        self._load_existing_store()

    def _load_existing_store(self):
        index_file = self.persist_directory / "index.faiss"

        if index_file.exists():
            self.store = FAISS.load_local(
                str(self.persist_directory),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def load(self, file_path):
        """Load a TXT or PDF document, split it, and add chunks to FAISS."""

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load document content based on file extension.
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            raise ValueError("Only PDF and TXT files are supported.")

        documents = loader.load()

        # Attach source metadata to every document.
        for document in documents:
            document.metadata["source"] = file_path.name

        # Split documents into smaller chunks.
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            return f"No text content found in '{file_path.name}'."

        # Initialize FAISS or add to the existing index.
        if self.store is None:
            self.store = FAISS.from_documents(
                chunks,
                self.embeddings
            )
        else:
            self.store.add_documents(chunks)

        # Persist the index to disk.
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.store.save_local(str(self.persist_directory))

        return (
            f"Document '{file_path.name}' ingested successfully. "
            f"{len(chunks)} chunks added."
        )

    def search(self, query, k=3):
        if self.store is None:
            return []

        results = self.store.similarity_search(query, k=k)

        print(f"FAISS index size: {self.store.index.ntotal}")

        if len(results) >= 2:
            print(
                "Chunks identical:",
                results[0].page_content == results[1].page_content
            )
            print(
                "Metadata identical:",
                results[0].metadata == results[1].metadata
            )

        print(f"\nFAISS SEARCH: {query}")
        print(f"Retrieved chunks: {len(results)}")

        for i, doc in enumerate(results, start=1):
            print(f"\nChunk {i}")
            print(f"Metadata: {doc.metadata}")
            print(f"Content length: {len(doc.page_content)}")
            print(f"Content preview: {doc.page_content[:200]}")

        return results