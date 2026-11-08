from pathlib import Path
import hashlib
import logging

from datetime import datetime, timezone

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class DocumentIngestion:

    def __init__(self, embeddings):

        self.embeddings = embeddings

        # --------------------------------------------------
        # 1. Storage configuration
        # --------------------------------------------------

        # FAISS storage location:
        # genai_core/memory/document_store/

        self.persist_directory = (
            Path(__file__).resolve().parent / "document_store"
        )

        self.shared_store = None
        self.user_stores = {}

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Create storage directory if it doesn't exist.
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # Create user storage directory.
        (self.persist_directory / "users").mkdir(
            parents=True,
            exist_ok=True
        )

        # Create shared storage directory.
        (self.persist_directory / "shared").mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # 2. Load existing shared FAISS index
        # --------------------------------------------------

        self.shared_store = self._load_store(
            self.persist_directory / "shared"
        )

        # --------------------------------------------------
        # 3. Log storage configuration
        # --------------------------------------------------

        logger.info(
            "FAISS storage directory: %s",
            self.persist_directory
        )

        logger.info(
            "Shared FAISS index loaded: %s",
            self.shared_store is not None
        )

        logger.info(
            "User FAISS storage directory: %s",
            self.persist_directory / "users"
        )

    # --------------------------------------------------
    # 2. Generate a safe directory name for each user
    # --------------------------------------------------

    def _get_user_directory(self, user_id):

        # --------------------------------------------------
        # Validate user ID
        # --------------------------------------------------

        if not user_id:

            raise ValueError(
                "user_id is required."
            )

        user_id = str(user_id).strip().lower()

        if not user_id:

            raise ValueError(
                "user_id cannot be empty."
            )

        # --------------------------------------------------
        # Generate SHA-256 hash
        # --------------------------------------------------

        user_hash = hashlib.sha256(
            user_id.encode("utf-8")
        ).hexdigest()

        # --------------------------------------------------
        # Construct private FAISS directory
        # --------------------------------------------------

        user_directory = (
            self.persist_directory
            / "users"
            / user_hash
        )

        # Create directory if necessary.
        user_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.debug(
            "Resolved private FAISS directory for user hash %s: %s",
            user_hash[:12],
            user_directory
        )

        return user_directory

    # --------------------------------------------------
    # 3. Calculate document hash
    # --------------------------------------------------

    def _get_file_hash(self, file_path):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    # --------------------------------------------------
    # 4. Load an existing FAISS index
    # --------------------------------------------------

    def _load_store(self, directory):

        directory = Path(directory)

        index_file = directory / "index.faiss"

        if not index_file.exists():

            logger.info(
                "No existing FAISS index found: %s",
                directory
            )

            return None

        try:

            store = FAISS.load_local(
                str(directory),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info(
                "FAISS index loaded successfully: %s",
                directory
            )

            return store

        except Exception:

            logger.exception(
                "Failed to load FAISS index: %s",
                directory
            )

            raise

    # --------------------------------------------------
    # 5. Get the correct FAISS store
    # --------------------------------------------------

    def _get_store(self, user_id=None, is_shared=False):

        if is_shared:
            return self.shared_store

        if not user_id:
            raise ValueError(
                "user_id is required for private documents."
            )

        user_id = str(user_id)

        if user_id in self.user_stores:
            return self.user_stores[user_id]

        directory = self._get_user_directory(user_id)

        store = self._load_store(directory)

        self.user_stores[user_id] = store

        return store

    # --------------------------------------------------
    # 6. Validate document paths and file types
    # --------------------------------------------------

    def _validate_file(self, file_path):

        if not file_path:
            raise ValueError(
                "A document file path is required."
            )

        file_path = Path(file_path).expanduser()

        try:

            file_path = file_path.resolve(strict=True)

        except FileNotFoundError:

            raise FileNotFoundError(
                f"Document file not found: {file_path}"
            )

        if not file_path.is_file():

            raise ValueError(
                f"The provided path is not a file: {file_path}"
            )

        supported_extensions = {
            ".pdf",
            ".txt"
        }

        if file_path.suffix.lower() not in supported_extensions:

            raise ValueError(
                "Unsupported document format. "
                "Only PDF and TXT files are supported."
            )

        return file_path

    # --------------------------------------------------
    # 7. Load and ingest a document
    # --------------------------------------------------

    def load(
        self,
        file_path,
        user_id=None,
        is_shared=False,
        source_name=None
    ):

        """
        Load a PDF or TXT document and add its chunks
        to either the shared or user-specific FAISS index.

        Args:
            file_path: Local path to the document.
            user_id: Authenticated user's ID or email.
            is_shared: Whether the document is shared.
            source_name: Original uploaded filename.

        Returns:
            A message describing the ingestion result.
        """

        if not is_shared and not user_id:

            raise ValueError(
                "user_id is required for private documents."
            )

        if user_id is not None:
            user_id = str(user_id)

        file_path = self._validate_file(file_path)

        # Preserve the original uploaded filename.
        if source_name:

            source_name = Path(source_name).name

        else:

            source_name = file_path.name

        # Calculate document hash.
        file_hash = self._get_file_hash(file_path)

        logger.info(
            "Starting document ingestion: %s",
            source_name
        )

        # --------------------------------------------------
        # Load document content
        # --------------------------------------------------

        try:

            if file_path.suffix.lower() == ".pdf":

                loader = PyPDFLoader(str(file_path))

            else:

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True
                )

            documents = loader.load()

        except Exception as exc:

            logger.exception(
                "Failed to load document: %s",
                source_name
            )

            raise RuntimeError(
                f"Unable to read document '{source_name}': {exc}"
            ) from exc

        if not documents:

            return (
                f"No readable content found in '{source_name}'."
            )

        # --------------------------------------------------
        # Attach metadata
        # --------------------------------------------------

        for document in documents:

            document.metadata["source"] = source_name

            document.metadata["file_hash"] = file_hash

            document.metadata["user_id"] = (
                None if is_shared else user_id
            )

            document.metadata["is_shared"] = bool(is_shared)

        # --------------------------------------------------
        # Split into chunks
        # --------------------------------------------------

        chunks = self.text_splitter.split_documents(
            documents
        )

        if not chunks:

            return (
                f"No text content found in '{source_name}'."
            )

        logger.info(
            "Document split into %s chunks: %s",
            len(chunks),
            source_name
        )

        # --------------------------------------------------
        # Determine storage location
        # --------------------------------------------------

        if is_shared:

            directory = (
                self.persist_directory / "shared"
            )

            store = self.shared_store

        else:

            directory = self._get_user_directory(user_id)

            store = self._get_store(
                user_id=user_id
            )

        # --------------------------------------------------
        # Initialize FAISS or update existing index
        # --------------------------------------------------

        try:

            if store is None:

                store = FAISS.from_documents(
                    chunks,
                    self.embeddings
                )

            else:

                store.add_documents(chunks)

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

            store.save_local(str(directory))

        except Exception as exc:

            logger.exception(
                "Failed to create or update FAISS index."
            )

            raise RuntimeError(
                f"Failed to index document '{source_name}': {exc}"
            ) from exc

        # --------------------------------------------------
        # Update in-memory reference
        # --------------------------------------------------

        if is_shared:

            self.shared_store = store

        else:

            self.user_stores[user_id] = store

        document_type = (
            "shared" if is_shared else "private"
        )

        logger.info(
            "Document ingestion completed: %s, chunks=%s, "
            "type=%s, hash=%s",
            source_name,
            len(chunks),
            document_type,
            file_hash
        )

        return (
            f"Document '{source_name}' ingested successfully. "
            f"{len(chunks)} chunks added. "
            f"Document type: {document_type}."
        )

    # --------------------------------------------------
    # 8. Search shared and user-specific documents
    # --------------------------------------------------

    def search(
        self,
        query,
        user_id,
        k=5,
        source=None
    ):

        """
        Search shared and requesting user's private indexes.

        Returns the best K document chunks ranked by FAISS
        L2 distance.
        """

        if not user_id:

            raise ValueError(
                "user_id is required for document retrieval."
            )

        if not query or not query.strip():

            raise ValueError(
                "A search query is required."
            )

        if k < 1:

            raise ValueError(
                "k must be greater than zero."
            )

        user_id = str(user_id)

        results = []

        # --------------------------------------------------
        # Identify indexes to search
        # --------------------------------------------------

        stores_to_search = [
            ("shared", self.shared_store)
        ]

        user_store = self._get_store(
            user_id=user_id
        )

        stores_to_search.append(
            ("private", user_store)
        )

        # --------------------------------------------------
        # Search each index
        # --------------------------------------------------

        for store_type, store in stores_to_search:

            if store is None:
                continue

            search_filter = (
                {"source": source} if source else None
            )

            try:

                matches = store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=search_filter
                )

            except Exception:

                logger.exception(
                    "FAISS search failed for %s index.",
                    store_type
                )

                raise

            for document, score in matches:

                results.append({

                    "document": document,

                    "score": float(score),

                    "store_type": store_type

                })

        # --------------------------------------------------
        # Rank results
        # --------------------------------------------------

        results.sort(
            key=lambda item: item["score"]
        )

        final_results = results[:k]

        # --------------------------------------------------
        # Retrieval diagnostics
        # --------------------------------------------------

        logger.info(
            "FAISS search completed. User=%s, "
            "results=%s, query=%s",
            user_id,
            len(final_results),
            query
        )

        for i, item in enumerate(final_results, start=1):

            document = item["document"]

            logger.info(
                "Retrieved chunk %s | Store=%s | "
                "Distance=%.4f | Source=%s | "
                "Content length=%s",
                i,
                item["store_type"],
                item["score"],
                document.metadata.get("source", "Unknown"),
                len(document.page_content)
            )

        # Keep the return format compatible with agent.py.
        return [
            item["document"]
            for item in final_results
        ]

    # --------------------------------------------------
    # 9. List available document sources
    # --------------------------------------------------

    def list_sources(self, user_id=None):

        """
        List shared and optionally user-specific sources.
        """

        stores_to_check = [
            ("shared", self.shared_store)
        ]

        if user_id:

            user_store = self._get_store(
                user_id=str(user_id)
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

