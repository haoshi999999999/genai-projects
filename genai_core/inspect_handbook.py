from memory.document_ingestion import DocumentIngestion

docs = DocumentIngestion(persist_dir="memory/document_store")
user_id = "bob@test.com"

store = docs._get_store(user_id=user_id, is_shared=False)
all_documents = docs._get_all_documents(store)

print("=" * 70)
print("Alice's private FAISS store")
print("Total records:", len(all_documents))
print("=" * 70)

for index, (faiss_id, document) in enumerate(all_documents, start=1):
    print(f"\n{'=' * 70}")
    print(f"RECORD {index}")
    print("=" * 70)

    print("FAISS ID:", faiss_id)
    print("Content preview:", document.page_content[:250])

    print("\nMetadata:")
    for key, value in sorted(document.metadata.items()):
        print(f"  {key}: {value}")

print("\nInspection completed. No FAISS data was modified.")