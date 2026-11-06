from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentIngestion:

    def __init__(self, embeddings):
        self.embeddings = embeddings

        # Root directory for all document indexes.
        self.persist_directory = (
            Path(__file__).resolve().parent / "document_store"
        )

        # Maintain separate FAISS stores in memory.
        self.shared_store = None
        self.user_stores = {}

        # Split documents into smaller, overlapping chunks.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Load the shared index when the application starts.
        self.shared_store = self._load_store(
            self.persist_directory / "shared"
        )

    # --------------------------------------------------
    # 1. Generate a safe directory name for each user
    # --------------------------------------------------

    def _get_user_directory(self, user_id):

        if not user_id:
            raise ValueError("user_id is required.")

        # Hash the user ID to avoid unsafe directory names.
        user_hash = hashlib.sha256(
            str(user_id).encode("utf-8")
        ).hexdigest()

        return self.persist_directory / "users" / user_hash

    # --------------------------------------------------
    # 2. Load an existing FAISS index
    # --------------------------------------------------

    def _load_store(self, directory):

        index_file = directory / "index.faiss"

        if not index_file.exists():
            return None

        return FAISS.load_local(
            str(directory),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    # --------------------------------------------------
    # 3. Get the correct FAISS store
    # --------------------------------------------------

    def _get_store(self, user_id=None, is_shared=False):

        # Shared document index.
        if is_shared:
            return self.shared_store

        if not user_id:
            raise ValueError(
                "user_id is required for private documents."
            )

        # Return an already-loaded user index.
        if user_id in self.user_stores:
            return self.user_stores[user_id]

        # Load the user's index if it already exists.
        directory = self._get_user_directory(user_id)

        store = self._load_store(directory)

        self.user_stores[user_id] = store

        return store

    # --------------------------------------------------
    # 4. Load and ingest a document
    # --------------------------------------------------

    def load(self, file_path, user_id=None, is_shared=False):

        """
        Load a PDF or TXT document and add its chunks
        to either the shared or user-specific FAISS index.
        """

        if not is_shared and not user_id:
            raise ValueError(
                "user_id is required for private documents."
            )

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # Load document content based on file extension.
        if file_path.suffix.lower() == ".pdf":

            loader = PyPDFLoader(str(file_path))

        elif file_path.suffix.lower() == ".txt":

            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        else:

            raise ValueError(
                "Only PDF and TXT files are supported."
            )

        documents = loader.load()

        # Attach metadata to every document.
        for document in documents:

            document.metadata["source"] = file_path.name

            document.metadata["user_id"] = (
                None if is_shared else str(user_id)
            )

            document.metadata["is_shared"] = is_shared

        # Split documents into smaller chunks.
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            return (
                f"No text content found in '{file_path.name}'."
            )

        # Determine the correct storage location.
        if is_shared:

            directory = self.persist_directory / "shared"
            store = self.shared_store

        else:

            directory = self._get_user_directory(user_id)
            store = self._get_store(user_id=user_id)

        # Initialize FAISS or add to the existing index.
        if store is None:

            store = FAISS.from_documents(
                chunks,
                self.embeddings
            )

        else:

            store.add_documents(chunks)

        # Persist the index to disk.
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        store.save_local(str(directory))

        # Update the in-memory reference.
        if is_shared:

            self.shared_store = store

        else:

            self.user_stores[user_id] = store

        document_type = (
            "shared" if is_shared else "private"
        )

        return (
            f"Document '{file_path.name}' ingested successfully. "
            f"{len(chunks)} chunks added. "
            f"Document type: {document_type}."
        )

    # --------------------------------------------------
    # 5. Search shared and user-specific documents
    # --------------------------------------------------

    def search(self, query, user_id, k=3, source=None):

        """
        Search both shared documents and the requesting
        user's private documents.

        Results from both indexes are combined and ranked
        by FAISS distance.
        """

        if not user_id:
            raise ValueError(
                "user_id is required for document retrieval."
            )

        results = []

        # Search the shared document index.
        stores_to_search = [
            ("shared", self.shared_store)
        ]

        # Search the requesting user's private index.
        user_store = self._get_store(user_id=user_id)

        stores_to_search.append(
            ("private", user_store)
        )

        # Retrieve results from both indexes.
        for store_type, store in stores_to_search:

            if store is None:
                continue

            search_filter = (
                {"source": source} if source else None
            )

            matches = store.similarity_search_with_score(
                query,
                k=k,
                filter=search_filter
            )

            for document, score in matches:

                results.append({
                    "document": document,
                    "score": float(score),
                    "store_type": store_type
                })

        # Sort by FAISS distance.
        results.sort(
            key=lambda item: item["score"]
        )

        # Return the best K results across both indexes.
        final_results = results[:k]

        print(f"\nFAISS SEARCH: {query}")
        print(f"User ID: {user_id}")
        print(f"Source filter: {source or 'All documents'}")
        print(f"Retrieved chunks: {len(final_results)}")

        for i, item in enumerate(final_results, start=1):

            document = item["document"]

            print(f"\nChunk {i}")
            print(f"Store: {item['store_type']}")
            print(f"Score: {item['score']}")
            print(f"Metadata: {document.metadata}")
            print(f"Content length: {len(document.page_content)}")
            print(f"Content preview: {document.page_content[:200]}")

        # Return documents in the format expected by
        # your existing retrieve_documents() method.
        return [
            item["document"]
            for item in final_results
        ]

    # --------------------------------------------------
    # 6. List available document sources
    # --------------------------------------------------

    def list_sources(self, user_id=None):

        """
        List shared document sources and, optionally,
        the requesting user's private document sources.
        """

        stores_to_check = [
            ("shared", self.shared_store)
        ]

        if user_id:

            user_store = self._get_store(
                user_id=user_id
            )

            stores_to_check.append(
                ("private", user_store)
            )

        sources = {}

        for store_type, store in stores_to_check:

            if store is None:
                continue

            for doc_id in store.index_to_docstore_id.values():

                document = store.docstore.search(doc_id)

                if isinstance(document, str):
                    continue

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                key = (store_type, source)

                if key not in sources:
                    sources[key] = 0

                sources[key] += 1

        return [
            {
                "source": source,
                "type": store_type,
                "chunks": count
            }
            for (store_type, source), count
            in sorted(sources.items())
        ]