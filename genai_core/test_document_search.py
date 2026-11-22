from langchain_huggingface import HuggingFaceEmbeddings
from memory.document_ingestion import DocumentIngestion


def main():

    # Enter the email address belonging to the document owner.
    user_id = input("Enter document owner's email: ").strip()

    if not user_id:
        print("User ID cannot be empty.")
        return

    # Initialize the embedding model.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load the existing FAISS document index.
    docs = DocumentIngestion(embeddings)

    # List indexed document sources.
    print("\nIndexed document sources:")

    for source in docs.list_sources(user_id):
        print(f"Document: {source['source']}")
        print(f"Chunks: {source['chunks']}")
        print()

    query = "How many vacation days do employees receive?"

    # ==================================================
    # TEST A: Search across accessible documents
    # ==================================================

    print("\n========== TEST A: ALL DOCUMENTS ==========")

    results = docs.search(
        query,
        user_id=user_id,
        k=3
    )

    print(f"\nRetrieved {len(results)} chunks\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Source:", result.metadata.get("source"))
        print("Content:", result.page_content[:200])
        print()

    # ==================================================
    # TEST B: Search only the selected document
    # ==================================================

    print("\n========== TEST B: SELECTED DOCUMENT ==========")

    results = docs.search(
        query,
        user_id=user_id,
        k=3,
        source="sample_employee_handbook.pdf"
    )

    print(f"\nRetrieved {len(results)} chunks\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Source:", result.metadata.get("source"))
        print("Content:", result.page_content[:200])
        print()


if __name__ == "__main__":
    main()