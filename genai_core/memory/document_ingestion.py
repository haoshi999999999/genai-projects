import hashlib
import logging
import uuid

from datetime import date, datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


ALLOWED_AUTHORITIES = {
    "Official Policy",
    "Employee Handbook",
    "Reference Material",
    "Informational"
}


class DocumentIngestion:

    # ==================================================
    # 1. INITIALIZATION
    # ==================================================

    def __init__(
        self,
        embeddings=None,
        persist_dir="memory/document_store"
    ):

        self._embeddings = embeddings

        self.persist_dir = Path(persist_dir)

        self.users_dir = self.persist_dir / "users"

        self.shared_dir = self.persist_dir / "shared"

        self.users_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.shared_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.user_stores = {}

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.shared_store = self._load_store(
            self.shared_dir
        )

        logger.info(
            "DocumentIngestion initialized."
        )

    # ==================================================
    # 2. EMBEDDINGS
    # ==================================================

    @property
    def embeddings(self):

        if self._embeddings is None:

            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        return self._embeddings

    # ==================================================
    # 3. USER IDENTIFICATION
    # ==================================================

    def _normalize_user_id(
        self,
        user_id
    ):

        if user_id is None:

            raise ValueError(
                "A user ID is required for private documents."
            )

        normalized = str(user_id).strip().lower()

        if not normalized:

            raise ValueError(
                "User ID cannot be empty."
            )

        return normalized

    def _get_user_directory(
        self,
        user_id
    ):

        normalized = self._normalize_user_id(
            user_id
        )

        user_hash = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        return self.users_dir / user_hash

    # ==================================================
    # 4. FAISS STORE MANAGEMENT
    # ==================================================

    def _load_store(
        self,
        directory
    ):

        directory = Path(directory)

        index_file = directory / "index.faiss"

        pkl_file = directory / "index.pkl"

        if not index_file.exists() or not pkl_file.exists():

            return None

        try:

            store = FAISS.load_local(
                str(directory),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info(
                "Loaded FAISS index: %s",
                directory
            )

            return store

        except Exception:

            logger.exception(
                "Failed to load FAISS index: %s",
                directory
            )

            raise

    def _get_store(
        self,
        user_id=None,
        is_shared=False
    ):

        if is_shared:

            return self.shared_store

        normalized = self._normalize_user_id(
            user_id
        )

        if normalized not in self.user_stores:

            directory = self._get_user_directory(
                normalized
            )

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

            self.user_stores[normalized] = self._load_store(
                directory
            )

        return self.user_stores[normalized]

    def _save_store(
        self,
        store,
        user_id=None,
        is_shared=False
    ):

        if store is None:

            return

        if is_shared:

            directory = self.shared_dir

        else:

            directory = self._get_user_directory(
                user_id
            )

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        store.save_local(
            str(directory)
        )

        logger.info(
            "Saved FAISS index: %s",
            directory
        )

    # ==================================================
    # 5. FILE VALIDATION
    # ==================================================

    def _validate_file(
        self,
        file_path
    ):

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        if not path.is_file():

            raise ValueError(
                "The supplied path is not a file."
            )

        if path.suffix.lower() not in {".pdf", ".txt"}:

            raise ValueError(
                "Only PDF and TXT files are supported."
            )

        if path.stat().st_size == 0:

            raise ValueError(
                "The uploaded file is empty."
            )

        return path

    # ==================================================
    # 6. FILE HASHING
    # ==================================================

    def _get_file_hash(
        self,
        file_path
    ):

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    # ==================================================
    # 7. DOCUMENT IDENTITY
    # ==================================================

    def _create_document_id(
        self,
        document_name,
        user_id=None,
        is_shared=False
    ):

        normalized_name = " ".join(
            str(document_name).strip().lower().split()
        )

        if not normalized_name:

            raise ValueError(
                "Document name cannot be empty."
            )

        if is_shared:

            scope = "shared"

        else:

            scope = self._normalize_user_id(
                user_id
            )

        identity = f"{scope}:{normalized_name}"

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    # ==================================================
    # 8. DOCUMENT LOOKUP
    # ==================================================

    def _get_all_documents(
        self,
        store
    ):
        """
        Return all (document_id, Document) pairs stored in FAISS.

        FAISS's internal docstore is a dictionary:

            {
                document_id: Document,
                ...
            }

        Returning key/value pairs allows callers to access
        both the FAISS document ID and Document metadata.
        """

        if store is None:

            return []

        return list(
            store.docstore._dict.items()
        )

    def _get_document_ids(
        self,
        store,
        source_name,
        file_hash=None
    ):

        matches = []

        for doc_id, document in self._get_all_documents(store):

            metadata = document.metadata

            if metadata.get("source") != source_name:

                continue

            if file_hash is not None:

                if metadata.get("file_hash") != file_hash:

                    continue

            matches.append(doc_id)

        return matches

    def _get_version_ids(
        self,
        store,
        document_id
    ):

        version_ids = set()

        for _, document in self._get_all_documents(store):

            metadata = document.metadata

            if metadata.get("document_id") == document_id:

                version_id = metadata.get("version_id")

                if version_id:

                    version_ids.add(version_id)

        return version_ids

    # ==================================================
    # 9. METADATA COMPARISON
    # ==================================================

    def _metadata_matches(
        self,
        store,
        document_ids,
        expected_metadata
    ):

        if not document_ids:

            return False

        for doc_id, document in self._get_all_documents(store):

            if doc_id not in document_ids:

                continue

            metadata = document.metadata

            for key, expected_value in expected_metadata.items():

                if metadata.get(key) != expected_value:

                    return False

        return True

    # ==================================================
    # 10. METADATA UPDATE
    # ==================================================

    def _update_document_metadata(
        self,
        store,
        document_ids,
        new_metadata
    ):

        updated = 0

        for doc_id, document in self._get_all_documents(store):

            if doc_id not in document_ids:

                continue

            document.metadata.update(
                new_metadata
            )

            updated += 1

        logger.info(
            "Updated metadata for %s chunks.",
            updated
        )

        return updated

    # ==================================================
    # 11. DOCUMENT DELETION
    # ==================================================

    def _delete_documents(
        self,
        store,
        document_ids
    ):

        """
        Explicit deletion helper.

        Version ingestion does NOT call this method.
        Historical versions are preserved.
        """

        if store is None or not document_ids:

            return 0

        existing_ids = []

        for doc_id, _ in self._get_all_documents(store):

            if doc_id in document_ids:

                existing_ids.append(doc_id)

        if not existing_ids:

            return 0

        store.delete(
            existing_ids
        )

        logger.info(
            "Deleted %s document chunks.",
            len(existing_ids)
        )

        return len(existing_ids)

    # ==================================================
    # 12. DATE HELPERS
    # ==================================================

    def _parse_date(
        self,
        value
    ):

        if not value:

            return None

        # datetime must be checked before date because
        # datetime is a subclass of date in Python.
        if isinstance(value, datetime):

            return value.date()

        if isinstance(value, date):

            return value

        return date.fromisoformat(
            str(value)
        )

    def _get_effective_date(
        self,
        metadata
    ):

        effective_date = metadata.get(
            "effective_date"
        )

        if effective_date:

            return self._parse_date(
                effective_date
            )

        upload_date = metadata.get(
            "upload_date"
        )

        if upload_date:

            parsed_upload_date = datetime.fromisoformat(
                upload_date
            )

            # Convert timezone-aware timestamps to local time
            # before extracting the calendar date.
            if parsed_upload_date.tzinfo is not None:

                parsed_upload_date = parsed_upload_date.astimezone()

            return parsed_upload_date.date()

        return date.min

    # ==================================================
    # 13. VERSION LIFECYCLE
    # ==================================================

    def _refresh_version_statuses(
        self,
        store,
        document_id
    ):

        """
        Recalculate version status for one logical document.

        The latest effective version becomes current.

        Future-effective versions remain scheduled.
        Older effective versions become superseded.
        """

        if store is None:

            return

        versions = {}

        # _get_all_documents returns (doc_id, Document) tuples.
        # We only need the Document object in this method.
        for _, document in self._get_all_documents(store):

            metadata = document.metadata

            if metadata.get("document_id") != document_id:

                continue

            version_id = metadata.get("version_id")

            if not version_id:

                continue

            if version_id not in versions:

                versions[version_id] = {

                    "effective_date": self._get_effective_date(
                        metadata
                    ),

                    "upload_date": metadata.get(
                        "upload_date",
                        ""
                    ),

                    "chunks": []

                }

            versions[version_id]["chunks"].append(
                document
            )

        if not versions:

            return

        today = date.today()

        eligible_versions = []

        future_versions = []

        for version_id, information in versions.items():

            if information["effective_date"] <= today:

                eligible_versions.append(
                    (
                        version_id,
                        information
                    )
                )

            else:

                future_versions.append(
                    (
                        version_id,
                        information
                    )
                )

        current_version_id = None

        if eligible_versions:

            eligible_versions.sort(
                key=lambda item: (
                    item[1]["effective_date"],
                    item[1]["upload_date"]
                )
            )

            current_version_id = eligible_versions[-1][0]

        for version_id, information in versions.items():

            if version_id == current_version_id:

                status = "current"

            elif information["effective_date"] > today:

                status = "scheduled"

            else:

                status = "superseded"

            for document in information["chunks"]:

                document.metadata["is_current"] = (
                    version_id == current_version_id
                )

                document.metadata["version_status"] = status

    # ==================================================
    # 14. ACTIVE VERSION SELECTION
    # ==================================================

    def _get_active_version_ids(
        self,
        store,
        as_of=None
    ):

        """
        Return the version IDs that are effective as of a date.

        This is calculated dynamically so a scheduled document
        can become active when its effective date arrives,
        even without another upload.
        """

        if store is None:

            return set()

        reference_date = self._parse_date(
            as_of
        ) if as_of else date.today()

        grouped_versions = {}

        legacy_ids = set()

        for doc_id, document in self._get_all_documents(store):

            metadata = document.metadata

            document_id = metadata.get("document_id")

            version_id = metadata.get("version_id")

            if not document_id or not version_id:

                # Backward compatibility for old documents
                # that were ingested before version metadata.
                if not metadata.get("version_status"):

                    legacy_ids.add(doc_id)

                continue

            key = (
                document_id,
                version_id
            )

            if key not in grouped_versions:

                grouped_versions[key] = {

                    "effective_date": self._get_effective_date(
                        metadata
                    ),

                    "upload_date": metadata.get(
                        "upload_date",
                        ""
                    )

                }

        grouped_by_document = {}

        for (document_id, version_id), information in grouped_versions.items():

            grouped_by_document.setdefault(
                document_id,
                []
            ).append(
                (
                    version_id,
                    information
                )
            )

        active_ids = set()

        for document_id, versions in grouped_by_document.items():

            eligible = [

                item

                for item in versions

                if item[1]["effective_date"] <= reference_date

            ]

            if not eligible:

                continue

            eligible.sort(
                key=lambda item: (
                    item[1]["effective_date"],
                    item[1]["upload_date"]
                )
            )

            active_ids.add(
                eligible[-1][0]
            )

        # Legacy documents remain searchable.
        active_ids.update(
            legacy_ids
        )

        return active_ids

    # ==================================================
    # 15. DOCUMENT INGESTION
    # ==================================================

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

        """
        Ingest a PDF or TXT document.

        Existing behavior is retained.

        New behavior:
        - Creates stable logical document IDs.
        - Creates unique version IDs.
        - Preserves historical versions.
        - Tracks current and scheduled versions.
        - Detects duplicate content.
        - Supports metadata-only updates.
        """

        # --------------------------------------------------
        # 1. Normalize user and scope
        # --------------------------------------------------

        if not isinstance(is_shared, bool):

            raise ValueError(
                "is_shared must be a boolean."
            )

        if is_shared:

            normalized_user_id = None

        else:

            normalized_user_id = self._normalize_user_id(
                user_id
            )

        # --------------------------------------------------
        # 2. Validate file
        # --------------------------------------------------

        path = self._validate_file(
            file_path
        )

        safe_source_name = (
            Path(source_name).name
            if source_name
            else path.name
        )

        if not safe_source_name:

            raise ValueError(
                "A valid source filename is required."
            )

        # --------------------------------------------------
        # 3. Validate metadata
        # --------------------------------------------------

        document_name = (
            str(document_name).strip()
            if document_name
            else safe_source_name
        )

        if not document_name:

            raise ValueError(
                "Document name cannot be empty."
            )

        document_version = (
            str(document_version).strip()
            if document_version
            else "1.0"
        )

        if not document_version:

            raise ValueError(
                "Document version cannot be empty."
            )

        if len(document_name) > 200:

            raise ValueError(
                "Document name cannot exceed 200 characters."
            )

        if len(document_version) > 50:

            raise ValueError(
                "Document version cannot exceed 50 characters."
            )

        if authority is not None:

            authority = str(authority).strip()

            if authority not in ALLOWED_AUTHORITIES:

                raise ValueError(
                    f"Invalid authority: {authority}"
                )

        if owner_department is not None:

            owner_department = str(
                owner_department
            ).strip()

            if len(owner_department) > 100:

                raise ValueError(
                    "Owner department cannot exceed 100 characters."
                )

        parsed_effective_date = self._parse_date(
            effective_date
        )

        effective_date_string = (
            parsed_effective_date.isoformat()
            if parsed_effective_date
            else None
        )

        # --------------------------------------------------
        # 4. Calculate document identity and file hash
        # --------------------------------------------------

        document_id = self._create_document_id(
            document_name,
            user_id=normalized_user_id,
            is_shared=is_shared
        )

        file_hash = self._get_file_hash(
            path
        )

        upload_date = datetime.now(
            timezone.utc
        ).isoformat()

        # --------------------------------------------------
        # 5. Select FAISS store
        # --------------------------------------------------

        store = self._get_store(
            user_id=normalized_user_id,
            is_shared=is_shared
        )

        # --------------------------------------------------
        # 6. Prepare metadata
        # --------------------------------------------------

        metadata = {

            "source": safe_source_name,

            "document_id": document_id,

            "document_name": document_name,

            "version_id": None,

            "document_version": document_version,

            "version_status": "current",

            "is_current": True,

            "effective_date": effective_date_string,

            "authority": authority,

            "owner_department": owner_department,

            "user_id": normalized_user_id,

            "is_shared": is_shared,

            "file_hash": file_hash,

            "upload_date": upload_date

        }

        # --------------------------------------------------
        # 7. Check for identical content
        # --------------------------------------------------

        existing_documents = []

        if store is not None:

            for doc_id, document in self._get_all_documents(store):

                existing_metadata = document.metadata

                if (
                    existing_metadata.get("document_id") == document_id
                    and existing_metadata.get("file_hash") == file_hash
                ):

                    existing_documents.append(
                        doc_id
                    )

        # --------------------------------------------------
        # 8. Metadata-only update
        # --------------------------------------------------

        if existing_documents:

            existing_version_id = None

            for doc_id, document in self._get_all_documents(store):

                if doc_id in existing_documents:

                    existing_version_id = document.metadata.get(
                        "version_id"
                    )

                    break

            if not existing_version_id:

                existing_version_id = str(
                    uuid.uuid4()
                )

            metadata["version_id"] = existing_version_id

            # Preserve original upload date.
            first_document = next(

                document

                for doc_id, document in self._get_all_documents(store)

                if doc_id in existing_documents

            )

            metadata["upload_date"] = (
                first_document.metadata.get(
                    "upload_date",
                    upload_date
                )
            )

            metadata_changed = not self._metadata_matches(
                store,
                existing_documents,
                metadata
            )

            if not metadata_changed:

                return {

                    "status": "skipped",

                    "operation": "duplicate",

                    "document_id": document_id,

                    "version_id": existing_version_id,

                    "document_name": document_name,

                    "document_version": document_version,

                    "source": safe_source_name,

                    "file_hash": file_hash,

                    "chunks": len(existing_documents)

                }

            self._update_document_metadata(
                store,
                existing_documents,
                metadata
            )

            self._refresh_version_statuses(
                store,
                document_id
            )

            self._save_store(
                store,
                user_id=normalized_user_id,
                is_shared=is_shared
            )

            return {

                "status": "metadata_updated",

                "operation": "metadata_update",

                "document_id": document_id,

                "version_id": existing_version_id,

                "document_name": document_name,

                "document_version": document_version,

                "source": safe_source_name,

                "file_hash": file_hash,

                "chunks": len(existing_documents)

            }

        # --------------------------------------------------
        # 9. Load document content
        # --------------------------------------------------

        if path.suffix.lower() == ".pdf":

            loader = PyPDFLoader(
                str(path)
            )

        else:

            loader = TextLoader(
                str(path),
                encoding="utf-8"
            )

        loaded_documents = loader.load()

        if not loaded_documents:

            raise ValueError(
                "No readable content was found in the document."
            )

        # --------------------------------------------------
        # 10. Split document into chunks
        # --------------------------------------------------

        chunks = self.splitter.split_documents(
            loaded_documents
        )

        if not chunks:

            raise ValueError(
                "Document splitting produced no chunks."
            )

        # --------------------------------------------------
        # 11. Assign version metadata
        # --------------------------------------------------

        version_id = str(
            uuid.uuid4()
        )

        metadata["version_id"] = version_id

        for chunk in chunks:

            chunk.metadata.update(
                metadata
            )

        # --------------------------------------------------
        # 12. Add new version to FAISS
        # --------------------------------------------------

        if store is None:

            store = FAISS.from_documents(
                chunks,
                self.embeddings
            )

        else:

            store.add_documents(
                chunks
            )

        # --------------------------------------------------
        # 13. Refresh version lifecycle
        # --------------------------------------------------

        self._refresh_version_statuses(
            store,
            document_id
        )

        # --------------------------------------------------
        # 14. Persist FAISS index
        # --------------------------------------------------

        self._save_store(
            store,
            user_id=normalized_user_id,
            is_shared=is_shared
        )

        # Update in-memory references.
        if is_shared:

            self.shared_store = store

        else:

            self.user_stores[normalized_user_id] = store

        # --------------------------------------------------
        # 15. Return structured ingestion result
        # --------------------------------------------------

        logger.info(
            "Document ingested | "
            "Document: %s | Version: %s | Chunks: %s",
            document_name,
            document_version,
            len(chunks)
        )

        return {

            "status": "ingested",

            "operation": "new_version",

            "document_id": document_id,

            "version_id": version_id,

            "document_name": document_name,

            "document_version": document_version,

            "source": safe_source_name,

            "file_hash": file_hash,

            "effective_date": effective_date_string,

            "version_status": next(

                document.metadata.get("version_status")

                for _, document in self._get_all_documents(store)

                if document.metadata.get("version_id") == version_id

            ),

            "chunks": len(chunks)

        }


    # ==================================================
    # 16. DOCUMENT SEARCH
    # ==================================================
    def search(
        self,
        query,
        user_id,
        k=5,
        source=None,
        include_historical=False,
        as_of=None
    ):

        """
        Search private and shared document indexes.

        Default:
            Search only the effective version of each document.

        include_historical=True:
            Search all versions effective on or before as_of.

        Adaptive retrieval:
            Increase the FAISS candidate limit when filtering
            removes too many candidates.

        Existing calls using query, user_id, k, and source
        continue to work.
        """

        # --------------------------------------------------
        # 1. Validate input
        # --------------------------------------------------

        normalized_user_id = self._normalize_user_id(
            user_id
        )

        if not isinstance(query, str) or not query.strip():

            raise ValueError(
                "Search query cannot be empty."
            )

        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20:

            raise ValueError(
                "k must be between 1 and 20."
            )

        if as_of is not None:

            as_of = self._parse_date(
                as_of
            )

        # --------------------------------------------------
        # 2. Get shared and private stores
        # --------------------------------------------------

        stores = [

            self.shared_store,

            self._get_store(
                user_id=normalized_user_id,
                is_shared=False
            )

        ]

        results = []

        # Initial FAISS candidate limit.
        initial_candidate_limit = min(
            max(k * 5, 20),
            100
        )

        # --------------------------------------------------
        # 3. Search each FAISS store
        # --------------------------------------------------

        for store in stores:

            if store is None:

                continue

            try:

                total_documents = len(
                    self._get_all_documents(store)
                )

                if total_documents == 0:

                    continue

                candidate_limit = min(
                    initial_candidate_limit,
                    total_documents
                )

                # --------------------------------------------------
                # 4. Calculate version filtering information ONCE
                # --------------------------------------------------

                if include_historical:

                    reference_date = as_of or date.today()

                    active_version_ids = None

                else:

                    active_version_ids = self._get_active_version_ids(
                        store,
                        as_of=as_of
                    )

                    reference_date = None

                # --------------------------------------------------
                # 5. Adaptive candidate retrieval
                # --------------------------------------------------

                filtered_results = []

                while True:

                    candidates = store.similarity_search_with_score(
                        query,
                        k=candidate_limit
                    )

                    filtered_results = []

                    # --------------------------------------------------
                    # 6. Apply version and source filters
                    # --------------------------------------------------

                    if include_historical:

                        for document, score in candidates:

                            metadata = document.metadata

                            effective = self._get_effective_date(
                                metadata
                            )

                            if effective > reference_date:

                                continue

                            if source and metadata.get("source") != source:

                                continue

                            filtered_results.append(
                                (document, score)
                            )

                    else:

                        for document, score in candidates:

                            metadata = document.metadata

                            version_id = metadata.get(
                                "version_id"
                            )

                            # Versioned documents must belong
                            # to the active version.
                            if (
                                version_id
                                and version_id not in active_version_ids
                            ):

                                continue

                            # Legacy documents remain searchable.
                            if (
                                source
                                and metadata.get("source") != source
                            ):

                                continue

                            filtered_results.append(
                                (document, score)
                            )

                    # --------------------------------------------------
                    # 7. Decide whether more candidates are needed
                    # --------------------------------------------------

                    if len(filtered_results) >= k:

                        break

                    if candidate_limit >= total_documents:

                        break

                    # Increase candidate pool and search again.
                    candidate_limit = min(
                        candidate_limit * 2,
                        total_documents
                    )

                # --------------------------------------------------
                # 8. Add valid results
                # --------------------------------------------------

                results.extend(
                    filtered_results
                )

            except Exception:

                logger.exception(
                    "FAISS search failed."
                )

                raise

        # --------------------------------------------------
        # 9. Sort by similarity score
        # --------------------------------------------------

        # Lower FAISS distance indicates a closer match.
        results.sort(
            key=lambda item: item[1]
        )

        # --------------------------------------------------
        # 10. Return top-k documents
        # --------------------------------------------------

        return [

            document

            for document, score in results[:k]

        ]

    # ==================================================
    # 17. LIST DOCUMENT SOURCES
    # ==================================================

    def list_sources(
        self,
        user_id
    ):

        normalized_user_id = self._normalize_user_id(
            user_id
        )

        stores = [

            (
                "shared",
                self.shared_store
            ),

            (
                "private",
                self._get_store(
                    user_id=normalized_user_id,
                    is_shared=False
                )
            )

        ]

        sources = {}

        for store_type, store in stores:

            if store is None:

                continue

            for _, document in self._get_all_documents(store):

                metadata = document.metadata

                source = metadata.get(
                    "source",
                    "unknown"
                )

                key = (
                    store_type,
                    source
                )

                if key not in sources:

                    sources[key] = {

                        "source": source,

                        "store_type": store_type,

                        "chunks": 0

                    }

                sources[key]["chunks"] += 1

        return list(
            sources.values()
        )

    # ==================================================
    # 18. LIST DOCUMENTS AND VERSION HISTORY
    # ==================================================

    def list_documents(
        self,
        user_id
    ):

        """
        Return logical documents grouped by document ID,
        with individual versions and chunk counts.

        Existing document metadata is retained in the output.
        """

        normalized_user_id = self._normalize_user_id(
            user_id
        )

        stores = [

            (
                "shared",
                self.shared_store
            ),

            (
                "private",
                self._get_store(
                    user_id=normalized_user_id,
                    is_shared=False
                )
            )

        ]

        grouped_documents = {}

        for store_type, store in stores:

            if store is None:

                continue

            for doc_id, document in self._get_all_documents(store):

                metadata = document.metadata

                document_id = metadata.get(
                    "document_id"
                )

                # Backward compatibility for documents
                # created before document_id existed.
                if not document_id:

                    legacy_identity = (

                        store_type,

                        metadata.get("source"),

                        metadata.get("file_hash")

                    )

                    document_id = hashlib.sha256(
                        str(legacy_identity).encode("utf-8")
                    ).hexdigest()

                version_id = metadata.get(
                    "version_id"
                )

                # Backward compatibility for older chunks.
                if not version_id:

                    legacy_identity = (
                        store_type,
                        metadata.get("source"),
                        metadata.get("file_hash"),
                        metadata.get("creationdate"),
                        metadata.get("total_pages")
                    )

                    version_id = "legacy:" + hashlib.sha256(
                        str(legacy_identity).encode("utf-8")
                    ).hexdigest()

                key = (
                    store_type,
                    document_id
                )

                if key not in grouped_documents:

                    grouped_documents[key] = {

                        "document_id": document_id,

                        "document_name": metadata.get(
                            "document_name",
                            metadata.get("source", "Unknown")
                        ),

                        "store_type": store_type,

                        "is_shared": store_type == "shared",

                        "versions": {}

                    }

                versions = grouped_documents[key]["versions"]

                if version_id not in versions:

                    versions[version_id] = {

                        "version_id": version_id,

                        "source": metadata.get(
                            "source"
                        ),

                        "document_name": metadata.get(
                            "document_name"
                        ),

                        "document_version": metadata.get(
                            "document_version"
                        ) or "Legacy",

                        "version_status": metadata.get(
                            "version_status",
                            "current"
                        ),

                        "is_current": metadata.get(
                            "is_current",
                            True
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

                        "file_hash": metadata.get(
                            "file_hash"
                        ),

                        "upload_date": metadata.get(
                            "upload_date"
                        ),

                        "chunks": 0

                    }

                versions[version_id]["chunks"] += 1

        # --------------------------------------------------
        # Recalculate statuses dynamically.
        # --------------------------------------------------

        for (store_type, document_id), document_info in grouped_documents.items():

            store = (

                self.shared_store

                if store_type == "shared"

                else self._get_store(
                    user_id=normalized_user_id,
                    is_shared=False
                )

            )

            active_ids = self._get_active_version_ids(
                store
            )

            for version_id, version_info in document_info["versions"].items():

                # Legacy records without an effective date retain
                # their existing status.
                if version_id.startswith("legacy:"):

                    effective = self._parse_date(
                        version_info["effective_date"]
                    ) if version_info["effective_date"] else None

                    if effective and effective > date.today():

                        version_info["version_status"] = "scheduled"

                        version_info["is_current"] = False

                    continue

                if version_id in active_ids:

                    version_info["version_status"] = "current"

                    version_info["is_current"] = True

                else:

                    effective = self._parse_date(
                        version_info["effective_date"]
                    ) if version_info["effective_date"] else date.min

                    if effective > date.today():

                        version_info["version_status"] = "scheduled"

                    else:

                        version_info["version_status"] = "superseded"

                    version_info["is_current"] = False

            document_info["versions"] = sorted(

                document_info["versions"].values(),

                key=lambda item: (

                    item.get("effective_date") or "",

                    item.get("upload_date") or ""

                ),

                reverse=True

            )

        return sorted(

            grouped_documents.values(),

            key=lambda item: item["document_name"].lower()

        )