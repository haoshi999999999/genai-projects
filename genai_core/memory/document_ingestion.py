from pathlib import Path
import hashlib
import logging

from datetime import datetime, timezone, date

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class DocumentIngestion:

    ALLOWED_AUTHORITIES = {
        "Official Policy",
        "Employee Handbook",
        "Reference Material",
        "Informational"
    }

    def __init__(
        self,
        embeddings=None,
        persist_dir="memory/document_store"
    ):

        self._embeddings = embeddings

        self.persist_dir = Path(persist_dir)

        self.users_dir = self.persist_dir / "users"
        self.shared_dir = self.persist_dir / "shared"

        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        self.user_stores = {}

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.shared_store = self._load_store(self.shared_dir)

    # --------------------------------------------------
    # USER ID NORMALIZATION
    # --------------------------------------------------

    def _normalize_user_id(self, user_id):

        if not user_id:
            raise ValueError("User ID is required.")

        normalized = str(user_id).strip().lower()

        if not normalized:
            raise ValueError("User ID cannot be empty.")

        return normalized

    # --------------------------------------------------
    # USER DIRECTORY
    # --------------------------------------------------

    def _get_user_directory(self, user_id):

        normalized_user_id = self._normalize_user_id(user_id)

        user_hash = hashlib.sha256(
            normalized_user_id.encode("utf-8")
        ).hexdigest()

        return self.users_dir / user_hash

    # --------------------------------------------------
    # LOAD EXISTING FAISS STORE
    # --------------------------------------------------

    def _load_store(self, directory):

        directory = Path(directory)

        index_file = directory / "index.faiss"

        if index_file.exists():

            try:

                logger.info(
                    "Loading FAISS index: %s",
                    directory
                )

                return FAISS.load_local(
                    str(directory),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

            except Exception:

                logger.exception(
                    "Failed to load FAISS index: %s",
                    directory
                )

                raise

        return None

    # --------------------------------------------------
    # EMBEDDINGS PROPERTY
    # --------------------------------------------------

    @property
    def embeddings(self):

        if not hasattr(self, "_embeddings"):

            from langchain_community.embeddings import (
                HuggingFaceEmbeddings
            )

            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        return self._embeddings

    # --------------------------------------------------
    # GET PRIVATE STORE
    # --------------------------------------------------

    def _get_store(self, user_id):

        normalized_user_id = self._normalize_user_id(user_id)

        if normalized_user_id not in self.user_stores:

            directory = self._get_user_directory(
                normalized_user_id
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

            self.user_stores[normalized_user_id] = (
                self._load_store(directory)
            )

        return self.user_stores[normalized_user_id]

    # --------------------------------------------------
    # SAVE STORE
    # --------------------------------------------------

    def _save_store(self, store, directory):

        if store is None:
            return

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        store.save_local(str(directory))

    # --------------------------------------------------
    # FILE VALIDATION
    # --------------------------------------------------

    def _validate_file(self, file_path):

        path = Path(file_path).expanduser().resolve(
            strict=True
        )

        if not path.is_file():

            raise ValueError(
                "The provided path is not a file."
            )

        if path.suffix.lower() not in {".pdf", ".txt"}:

            raise ValueError(
                "Only PDF and TXT files are supported."
            )

        return path

    # --------------------------------------------------
    # FILE HASH
    # --------------------------------------------------

    def _get_file_hash(self, file_path):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            for block in iter(
                lambda: file.read(65536),
                b""
            ):

                sha256.update(block)

        return sha256.hexdigest()

    # --------------------------------------------------
    # FIND EXISTING DOCUMENT CHUNKS
    # --------------------------------------------------

    def _get_document_ids(
        self,
        store,
        source_name,
        file_hash=None
    ):

        matching_ids = []

        if store is None:
            return matching_ids

        for doc_id in list(
            store.index_to_docstore_id.values()
        ):

            doc = store.docstore.search(doc_id)

            if not doc or not hasattr(doc, "metadata"):
                continue

            metadata = doc.metadata

            if metadata.get("source") != source_name:
                continue

            if file_hash is not None:

                if metadata.get("file_hash") != file_hash:
                    continue

            matching_ids.append(doc_id)

        return matching_ids

    # --------------------------------------------------
    # COMPARE DOCUMENT METADATA
    # --------------------------------------------------

    def _metadata_matches(
        self,
        store,
        document_ids,
        new_metadata
    ):

        if not document_ids:
            return False

        fields_to_compare = [
            "document_name",
            "document_version",
            "effective_date",
            "authority",
            "owner_department",
            "user_id",
            "is_shared"
        ]

        for doc_id in document_ids:

            doc = store.docstore.search(doc_id)

            if not doc or not hasattr(doc, "metadata"):
                return False

            existing_metadata = doc.metadata

            for field in fields_to_compare:

                if (
                    existing_metadata.get(field)
                    != new_metadata.get(field)
                ):

                    return False

        return True

    # --------------------------------------------------
    # UPDATE EXISTING DOCUMENT METADATA
    # --------------------------------------------------

    def _update_document_metadata(
        self,
        store,
        document_ids,
        metadata
    ):

        updated_count = 0

        for doc_id in document_ids:

            doc = store.docstore.search(doc_id)

            if not doc or not hasattr(doc, "metadata"):
                continue

            # Preserve chunk-specific metadata such as
            # PDF page numbers.

            doc.metadata.update(metadata)

            updated_count += 1

        return updated_count

    # --------------------------------------------------
    # DELETE DOCUMENT CHUNKS
    # --------------------------------------------------

    def _delete_documents(
        self,
        store,
        document_ids
    ):

        if not document_ids:
            return

        store.delete(ids=document_ids)

    # --------------------------------------------------
    # LOAD AND INGEST DOCUMENT
    # --------------------------------------------------

    def load(
        self,
        file_path,
        user_id=None,
        is_shared=False,
        source_name=None,
        document_name=None,
        document_version=None,
        effective_date=None,
        authority=None,
        owner_department=None
    ):

        # ----------------------------------------------
        # VALIDATE USER
        # ----------------------------------------------

        if not is_shared:

            user_id = self._normalize_user_id(
                user_id
            )

        else:

            user_id = None

        # ----------------------------------------------
        # VALIDATE FILE
        # ----------------------------------------------

        path = self._validate_file(file_path)

        source_name = (
            Path(source_name).name
            if source_name
            else path.name
        )

        if not source_name:

            raise ValueError(
                "Invalid document source name."
            )

        document_name = (
            document_name or source_name
        )

        # ----------------------------------------------
        # VALIDATE AUTHORITY
        # ----------------------------------------------

        if authority is not None:

            authority = str(authority).strip()

            if authority not in self.ALLOWED_AUTHORITIES:

                raise ValueError(
                    f"Invalid authority: {authority}"
                )

        # ----------------------------------------------
        # VALIDATE EFFECTIVE DATE
        # ----------------------------------------------

        if effective_date is not None:

            try:

                effective_date = date.fromisoformat(
                    str(effective_date)
                ).isoformat()

            except ValueError:

                raise ValueError(
                    "effective_date must use YYYY-MM-DD format."
                )

        # ----------------------------------------------
        # COMPUTE FILE HASH
        # ----------------------------------------------

        file_hash = self._get_file_hash(path)

        upload_date = datetime.now(
            timezone.utc
        ).isoformat()

        # ----------------------------------------------
        # BUILD METADATA
        # ----------------------------------------------

        metadata = {
            "source": source_name,
            "document_name": document_name,
            "document_version": document_version,
            "effective_date": effective_date,
            "authority": authority,
            "owner_department": owner_department,
            "user_id": user_id,
            "is_shared": is_shared,
            "file_hash": file_hash,
            "upload_date": upload_date
        }

        # ----------------------------------------------
        # SELECT FAISS STORE
        # ----------------------------------------------

        if is_shared:

            store = self.shared_store
            directory = self.shared_dir

        else:

            store = self._get_store(user_id)

            directory = self._get_user_directory(
                user_id
            )

        # ----------------------------------------------
        # CHECK FOR SAME FILENAME + SAME CONTENT
        # ----------------------------------------------

        existing_same_hash = self._get_document_ids(
            store,
            source_name,
            file_hash
        )

        if existing_same_hash:

            # Same content and same metadata:
            # Do not create duplicate chunks.

            if self._metadata_matches(
                store,
                existing_same_hash,
                metadata
            ):

                logger.info(
                    "Duplicate upload skipped: %s",
                    source_name
                )

                return {
                    "status": "skipped",
                    "reason": "Document already exists with identical content and metadata.",
                    "source": source_name,
                    "file_hash": file_hash,
                    "chunks": len(existing_same_hash)
                }

            # Same content but different metadata:
            # Update metadata without generating embeddings.

            updated_count = self._update_document_metadata(
                store,
                existing_same_hash,
                metadata
            )

            self._save_store(
                store,
                directory
            )

            logger.info(
                "Document metadata updated: %s",
                source_name
            )

            return {
                "status": "metadata_updated",
                "source": source_name,
                "file_hash": file_hash,
                "chunks": updated_count
            }

        # ----------------------------------------------
        # CHECK FOR SAME FILENAME + DIFFERENT CONTENT
        # ----------------------------------------------

        existing_same_source = self._get_document_ids(
            store,
            source_name
        )

        # Only delete modern documents with file_hash.
        # Preserve legacy documents without file_hash.

        modern_document_ids = []

        if store is not None:

            for doc_id in existing_same_source:

                doc = store.docstore.search(doc_id)

                if not doc or not hasattr(doc, "metadata"):
                    continue

                if doc.metadata.get("file_hash"):

                    modern_document_ids.append(doc_id)

        if modern_document_ids:

            logger.info(
                "Replacing existing document: %s",
                source_name
            )

            self._delete_documents(
                store,
                modern_document_ids
            )

        # ----------------------------------------------
        # LOAD DOCUMENT CONTENT
        # ----------------------------------------------

        if path.suffix.lower() == ".pdf":

            loader = PyPDFLoader(str(path))

        else:

            loader = TextLoader(
                str(path),
                encoding="utf-8"
            )

        documents = loader.load()

        if not documents:

            raise ValueError(
                "The uploaded document contains no readable content."
            )

        # ----------------------------------------------
        # ATTACH METADATA
        # ----------------------------------------------

        for doc in documents:

            doc.metadata.update(metadata)

        # ----------------------------------------------
        # SPLIT DOCUMENT INTO CHUNKS
        # ----------------------------------------------

        chunks = self.splitter.split_documents(
            documents
        )

        if not chunks:

            raise ValueError(
                "Document splitting produced no chunks."
            )

        # ----------------------------------------------
        # ADD DOCUMENT TO FAISS
        # ----------------------------------------------

        if store is None:

            store = FAISS.from_documents(
                chunks,
                self.embeddings
            )

        else:

            store.add_documents(chunks)

        # ----------------------------------------------
        # SAVE INDEX
        # ----------------------------------------------

        self._save_store(
            store,
            directory
        )

        # ----------------------------------------------
        # UPDATE IN-MEMORY STORE
        # ----------------------------------------------

        if is_shared:

            self.shared_store = store

        else:

            self.user_stores[user_id] = store

        logger.info(
            "Document ingested successfully: %s (%s chunks)",
            source_name,
            len(chunks)
        )

        return {
            "status": "ingested",
            "source": source_name,
            "file_hash": file_hash,
            "chunks": len(chunks)
        }

    # --------------------------------------------------
    # DOCUMENT SEARCH
    # --------------------------------------------------

    def search(
        self,
        query,
        user_id,
        k=5,
        source=None
    ):

        user_id = self._normalize_user_id(
            user_id
        )

        if not query or not str(query).strip():

            raise ValueError(
                "Search query cannot be empty."
            )

        if not isinstance(k, int) or not 1 <= k <= 20:

            raise ValueError(
                "k must be an integer between 1 and 20."
            )

        results = []

        stores = [
            self.shared_store,
            self._get_store(user_id)
        ]

        # ----------------------------------------------
        # SEARCH SHARED AND PRIVATE INDEXES
        # ----------------------------------------------

        for store in stores:

            if store is None:
                continue

            try:

                matches = store.similarity_search_with_score(
                    query,
                    k=k
                )

                for doc, score in matches:

                    if source:

                        if (
                            doc.metadata.get("source")
                            != source
                        ):

                            continue

                    results.append(
                        (doc, score)
                    )

            except Exception:

                logger.exception(
                    "Document search failed."
                )

        # ----------------------------------------------
        # SORT BY FAISS DISTANCE
        # ----------------------------------------------

        results.sort(
            key=lambda item: item[1]
        )

        final_results = [
            doc
            for doc, score in results[:k]
        ]

        logger.info(
            "Document search completed: user=%s results=%s",
            user_id,
            len(final_results)
        )

        return final_results

    # --------------------------------------------------
    # LIST DOCUMENT SOURCES
    # --------------------------------------------------

    def list_sources(self, user_id):

        user_id = self._normalize_user_id(
            user_id
        )

        results = {}

        stores = [
            ("shared", self.shared_store),
            ("private", self._get_store(user_id))
        ]

        for store_type, store in stores:

            if store is None:
                continue

            for doc_id in list(
                store.index_to_docstore_id.values()
            ):

                doc = store.docstore.search(doc_id)

                if not doc or not hasattr(doc, "metadata"):
                    continue

                source = doc.metadata.get(
                    "source",
                    "unknown"
                )

                key = (
                    store_type,
                    source
                )

                if key not in results:

                    results[key] = {
                        "source": source,
                        "type": store_type,
                        "chunks": 0
                    }

                results[key]["chunks"] += 1

        return sorted(
            results.values(),
            key=lambda item: (
                item["type"],
                item["source"]
            )
        )

    # --------------------------------------------------
    # LIST DOCUMENTS
    # --------------------------------------------------

    def list_documents(self, user_id):

        user_id = self._normalize_user_id(
            user_id
        )

        results = {}

        stores = [
            ("shared", self.shared_store),
            ("private", self._get_store(user_id))
        ]

        for store_type, store in stores:

            if store is None:
                continue

            for doc_id in list(
                store.index_to_docstore_id.values()
            ):

                doc = store.docstore.search(doc_id)

                if not doc or not hasattr(doc, "metadata"):
                    continue

                metadata = doc.metadata

                source = metadata.get(
                    "source",
                    "unknown"
                )

                file_hash = metadata.get(
                    "file_hash"
                )

                upload_date = metadata.get(
                    "upload_date"
                )

                # Legacy documents may not have upload_date.
                # Use their individual doc ID to avoid incorrectly
                # grouping unrelated legacy chunks.

                if upload_date:

                    group_key = (
                        store_type,
                        source,
                        file_hash,
                        upload_date
                    )

                else:

                    group_key = (
                        store_type,
                        source,
                        file_hash,
                        doc_id
                    )

                if group_key not in results:

                    results[group_key] = {
                        "source": source,
                        "type": store_type,
                        "file_hash": file_hash,
                        "upload_date": upload_date,
                        "document_name": metadata.get(
                            "document_name"
                        ),
                        "document_version": metadata.get(
                            "document_version"
                        ),
                        "effective_date": metadata.get(
                            "effective_date"
                        ),
                        "authority": metadata.get(
                            "authority"
                        ),
                        "owner_department": metadata.get(
                            "owner_department"
                        ),
                        "chunks": 0
                    }

                results[group_key]["chunks"] += 1

        return sorted(
            results.values(),
            key=lambda item: (
                item["type"],
                item["source"],
                item["upload_date"] or ""
            )
        )