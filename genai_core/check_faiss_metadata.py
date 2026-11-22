import hashlib
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

USER_EMAIL = "bob@test.com"

TARGET_SOURCE = "metadata_test_policy.txt"

BASE_DIRECTORY = (
    Path(__file__).resolve().parent
    / "memory"
    / "document_store"
)


# --------------------------------------------------
# GENERATE USER-SPECIFIC FAISS DIRECTORY
# --------------------------------------------------

user_hash = hashlib.sha256(
    USER_EMAIL.strip().lower().encode("utf-8")
).hexdigest()

user_directory = (
    BASE_DIRECTORY
    / "users"
    / user_hash
)

print("\nUser FAISS directory:")
print(user_directory)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# LOAD EXISTING FAISS INDEX
# --------------------------------------------------

index_file = user_directory / "index.faiss"

if not index_file.exists():

    raise FileNotFoundError(
        f"FAISS index not found: {index_file}"
    )

print("\nLoading existing FAISS index...")

store = FAISS.load_local(
    str(user_directory),
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully.")


# --------------------------------------------------
# INSPECT DOCUMENT METADATA
# --------------------------------------------------

print("\nSearching stored chunks...")

found_documents = 0

for doc_id in store.index_to_docstore_id.values():

    document = store.docstore.search(doc_id)

    if isinstance(document, str):
        continue

    metadata = document.metadata

    if metadata.get("source") != TARGET_SOURCE:
        continue

    found_documents += 1

    print("\n" + "=" * 60)
    print(f"DOCUMENT CHUNK #{found_documents}")
    print("=" * 60)

    print("\nContent:")
    print(document.page_content[:300])

    print("\nMetadata:")

    expected_fields = [
        "source",
        "file_hash",
        "user_id",
        "is_shared",
        "document_name",
        "document_version",
        "effective_date",
        "upload_date",
        "authority",
        "owner_department"
    ]

    for field in expected_fields:

        print(
            f"{field}: {metadata.get(field, '<MISSING>')}"
        )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)

if found_documents:

    print(
        f"SUCCESS: Found {found_documents} "
        f"chunk(s) for {TARGET_SOURCE}."
    )

else:

    print(
        f"NOT FOUND: No chunks found for {TARGET_SOURCE}."
    )

print("=" * 60)