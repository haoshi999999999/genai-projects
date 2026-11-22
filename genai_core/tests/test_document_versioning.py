import hashlib

from datetime import date, timedelta

import pytest

from langchain_core.embeddings import Embeddings
from memory.document_ingestion import DocumentIngestion


# ==========================================================
# 1. Fake Embeddings
# ==========================================================

class FakeEmbeddings(Embeddings):
    """
    Lightweight embedding implementation for automated tests.

    Avoids downloading or calling a real embedding model.
    """

    def _embed(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        # Convert the first 8 bytes into a deterministic vector.
        return [
            byte / 255.0
            for byte in digest[:8]
        ]

    def embed_documents(self, texts):
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(self, text):
        return self._embed(text)


# ==========================================================
# 2. Pytest Fixtures
# ==========================================================

@pytest.fixture
def document_store(tmp_path):
    """
    Creates an isolated document store for each test.

    tmp_path is automatically created and cleaned up by pytest.
    """

    embeddings = FakeEmbeddings()

    store_path = tmp_path / "document_store"

    docs = DocumentIngestion(
        embeddings=embeddings,
        persist_dir=str(store_path)
    )

    return docs


@pytest.fixture
def sample_document(tmp_path):
    """
    Creates a temporary vacation policy document.
    """

    file_path = tmp_path / "vacation_policy.txt"

    file_path.write_text(
        """
        Company Vacation Policy

        Employees receive 20 vacation days per year.

        Employees may carry over 5 unused days.
        """,
        encoding="utf-8"
    )

    return file_path


# ==========================================================
# 3. Test: First Document Upload
# ==========================================================

def test_first_document_upload(
    document_store,
    sample_document
):
    result = document_store.load(
        file_path=str(sample_document),
        user_id="alice"
    )

    assert result["status"] == "ingested"
    assert result["operation"] == "new_version"
    assert result["version_status"] == "current"

    #result = document_store.load(
    #    file_path=str(sample_document),
    #    user_id="bob@test.com",
    #    document_name="Vacation Policy",
    #    document_version="1.0",
    #    effective_date="2026-01-01",
    #    authority="Official Policy",
    #    owner_department="HR"
    #)

    #assert result["status"] == "ingested"
    #assert result["operation"] == "new_version"

    #print("\nFirst upload:", result)

    #print("\n========== STEP 1 ==========")
    #print("Document store:", type(document_store))
    #print("Sample document:", sample_document)
    #print("File exists:", sample_document.exists())

    #print("\n========== STEP 2 ==========")
    #result = document_store.load(
    #    file_path=str(sample_document),
    #    user_id="alice"
    #)

    #print("\n========== STEP 3 ==========")
    #print("Load result:", result)

    #print("\n========== STEP 4 ==========")
    #assert result["status"] == "ingested"

    #print("Status assertion passed!")

    #assert result["operation"] == "new_version"

    #print("Operation assertion passed!")


# ==========================================================
# 4. Test: Duplicate Document Upload
# ==========================================================

def test_duplicate_document_upload(
    document_store,
    sample_document
):

    first_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    second_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    assert first_result["status"] == "ingested"

    assert second_result["status"] == "skipped"
    assert second_result["operation"] == "duplicate"

    print("\nDuplicate upload:", second_result)


# ==========================================================
# 5. Test: New Version Upload
# ==========================================================

def test_new_document_version(
    document_store,
    sample_document
):

    # ------------------------------------------------------
    # Step 1: Upload version 1.0.
    # ------------------------------------------------------

    first_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 2: Change the document content.
    # ------------------------------------------------------

    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 25 vacation days per year.

        Employees may carry over 7 unused days.
        """,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Step 3: Upload version 2.0.
    # ------------------------------------------------------

    second_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2026-02-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 4: Verify both upload operations.
    # ------------------------------------------------------

    assert first_result["status"] == "ingested"

    assert second_result["status"] == "ingested"
    assert second_result["operation"] == "new_version"

    print("\nVersion 1:", first_result)
    print("\nVersion 2:", second_result)

    # ------------------------------------------------------
    # Step 5: Retrieve the logical document.
    # ------------------------------------------------------

    documents = document_store.list_documents(
        user_id="bob@test.com"
    )

    vacation_documents = [
        item for item in documents
        if item.get("document_name") == "Vacation Policy"
    ]

    # There should be ONE logical document.
    assert len(vacation_documents) == 1

    vacation_document = vacation_documents[0]

    # ------------------------------------------------------
    # Step 6: Verify version history.
    # ------------------------------------------------------

    versions = vacation_document["versions"]

    # One logical document should contain two versions.
    assert len(versions) == 2

    version_labels = {
        version["document_version"]
        for version in versions
    }

    assert version_labels == {"1.0", "2.0"}

    # ------------------------------------------------------
    # Step 7: Verify version lifecycle statuses.
    # ------------------------------------------------------

    version_statuses = {
        version["document_version"]: version["version_status"]
        for version in versions
    }

    assert version_statuses["2.0"] == "current"
    assert version_statuses["1.0"] == "superseded"

    print(
        "\nDocument version history:",
        vacation_document
    )


# ==========================================================
# 6. Test: Private Document Isolation
# ==========================================================

def test_private_document_isolation(
    document_store,
    sample_document
):

    # ------------------------------------------------------
    # Step 1: Bob uploads a private document.
    # ------------------------------------------------------

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 2: Alice retrieves her documents.
    # ------------------------------------------------------

    alice_documents = document_store.list_documents(
        user_id="alice@test.com"
    )

    alice_vacation_documents = [
        item for item in alice_documents
        if item.get("document_name") == "Vacation Policy"
    ]

    # Alice should not see Bob's private document.
    assert len(alice_vacation_documents) == 0


# ==========================================================
# 7. Test: Scheduled Document Version
# ==========================================================

def test_future_effective_version_is_scheduled(
    document_store,
    sample_document
):

    # ------------------------------------------------------
    # Step 1: Calculate effective dates.
    # ------------------------------------------------------

    current_effective_date = (
        date.today() - timedelta(days=30)
    )

    future_effective_date = (
        date.today() + timedelta(days=30)
    )

    # ------------------------------------------------------
    # Step 2: Upload version 1.0.
    # This version is already effective.
    # ------------------------------------------------------

    first_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date=current_effective_date.isoformat(),
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 3: Change the document content.
    # ------------------------------------------------------

    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 30 vacation days per year.

        Employees may carry over 10 unused days.
        """,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Step 4: Upload version 2.0.
    # This version becomes effective in 30 days.
    # ------------------------------------------------------

    second_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date=future_effective_date.isoformat(),
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 5: Verify upload operations.
    # ------------------------------------------------------

    assert first_result["status"] == "ingested"

    assert second_result["status"] == "ingested"
    assert second_result["operation"] == "new_version"

    # ------------------------------------------------------
    # Step 6: Retrieve document version history.
    # ------------------------------------------------------

    documents = document_store.list_documents(
        user_id="bob@test.com"
    )

    vacation_documents = [
        item for item in documents
        if item.get("document_name") == "Vacation Policy"
    ]

    assert len(vacation_documents) == 1

    vacation_document = vacation_documents[0]

    versions = vacation_document["versions"]

    assert len(versions) == 2

    # ------------------------------------------------------
    # Step 7: Verify version lifecycle statuses.
    # ------------------------------------------------------

    version_statuses = {
        version["document_version"]: version["version_status"]
        for version in versions
    }

    # Version 1.0 is currently effective.
    assert version_statuses["1.0"] == "current"

    # Version 2.0 is not effective yet.
    assert version_statuses["2.0"] == "scheduled"

    print("\nScheduled version test:")
    print("Current version: 1.0")
    print("Scheduled version: 2.0")
    print("Future effective date:", future_effective_date)
    print("Version statuses:", version_statuses)



# ==========================================================
# 8. Test: Active Version Retrieval
# ==========================================================

def test_search_returns_only_active_version(
    document_store,
    sample_document
):

    # ------------------------------------------------------
    # Step 1: Upload version 1.0.
    # This version is already effective.
    # ------------------------------------------------------

    first_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 2: Change the document content.
    # ------------------------------------------------------

    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 30 vacation days per year.

        Employees may carry over 10 unused days.
        """,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Step 3: Upload version 2.0.
    # This version becomes effective in the future.
    # ------------------------------------------------------

    future_effective_date = (
        date.today() + timedelta(days=30)
    )

    second_result = document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date=future_effective_date.isoformat(),
        authority="Official Policy",
        owner_department="HR"
    )

    assert first_result["status"] == "ingested"
    assert second_result["status"] == "ingested"

    # ------------------------------------------------------
    # Step 4: Search using the default behavior.
    # ------------------------------------------------------

    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5
    )

    # Search must return results.
    assert len(results) > 0

    # ------------------------------------------------------
    # Step 5: Verify the retrieved versions.
    # ------------------------------------------------------

    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    # Only version 1.0 should be returned.
    assert retrieved_versions == {"1.0"}

    # ------------------------------------------------------
    # Step 6: Verify the actual document content.
    # ------------------------------------------------------

    retrieved_content = "\n".join(
        document.page_content
        for document in results
    )

    assert "20 vacation days" in retrieved_content

    assert "30 vacation days" not in retrieved_content

    print("\nActive version retrieval:")
    print("Retrieved versions:", retrieved_versions)
    print("Retrieved content:", retrieved_content)


# ==========================================================
# 9. Test: Retrieval As Of Future Date
# ==========================================================

def test_search_returns_version_effective_on_requested_date(
    document_store,
    sample_document
):

    # ------------------------------------------------------
    # Step 1: Upload version 1.0.
    # ------------------------------------------------------

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 2: Upload version 2.0.
    # ------------------------------------------------------

    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 30 vacation days per year.

        Employees may carry over 10 unused days.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2026-12-12",
        authority="Official Policy",
        owner_department="HR"
    )

    # ------------------------------------------------------
    # Step 3: Search as of December 15, 2026.
    # ------------------------------------------------------

    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5,
        as_of="2026-12-15"
    )

    assert len(results) > 0

    # ------------------------------------------------------
    # Step 4: Verify version 2.0 is retrieved.
    # ------------------------------------------------------

    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    assert retrieved_versions == {"2.0"}

    retrieved_content = "\n".join(
        document.page_content
        for document in results
    )

    assert "30 vacation days" in retrieved_content

    assert "20 vacation days" not in retrieved_content

    print("\nHistorical date retrieval:")
    print("Reference date: 2026-12-15")
    print("Retrieved versions:", retrieved_versions)
    print("Retrieved content:", retrieved_content)


def test_search_includes_historical_versions(
    document_store,
    sample_document
):
    # Upload version 1.0
    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # Change the document content for version 2.0
    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 30 vacation days per year.

        Employees may carry over 10 unused days.
        """,
        encoding="utf-8"
    )

    # Upload version 2.0
    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2026-12-12",
        authority="Official Policy",
        owner_department="HR"
    )

    # Search with historical retrieval enabled
    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5,
        as_of="2026-12-15",
        include_historical=True
    )

    assert len(results) > 0

    # Both versions should be retrieved
    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    assert retrieved_versions == {"1.0", "2.0"}

    # Verify that both versions' content is present
    retrieved_content = "\n".join(
        document.page_content
        for document in results
    )

    assert "20 vacation days" in retrieved_content
    assert "30 vacation days" in retrieved_content


def test_search_does_not_starve_active_version(
    document_store,
    sample_document,
    monkeypatch
):
    # --------------------------------------------------
    # 1. Upload an old version with enough chunks
    # --------------------------------------------------

    old_content = (
        "Company Vacation Policy. "
        "Employees receive 20 vacation days per year. "
        "Employees may carry over 5 unused days. "
    ) * 700

    sample_document.write_text(
        old_content,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # --------------------------------------------------
    # 2. Upload a new active version
    # --------------------------------------------------

    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 30 vacation days per year.

        Employees may carry over 10 unused days.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2026-02-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # --------------------------------------------------
    # 3. Get the private FAISS store
    # --------------------------------------------------

    store = document_store._get_store(
        user_id="bob@test.com",
        is_shared=False
    )

    all_documents = [
        document
        for _, document in document_store._get_all_documents(store)
    ]

    old_documents = [
        document
        for document in all_documents
        if document.metadata.get("document_version") == "1.0"
    ]

    active_documents = [
        document
        for document in all_documents
        if document.metadata.get("document_version") == "2.0"
    ]

    # Make sure the test has enough old-version chunks
    # to exceed the initial FAISS candidate limit.
    assert len(old_documents) > 25

    assert len(active_documents) > 0

    # --------------------------------------------------
    # 4. Simulate FAISS returning old documents first
    # --------------------------------------------------

    ordered_candidates = [
        (document, float(index))
        for index, document in enumerate(old_documents)
    ]

    ordered_candidates.extend(
        [
            (document, float(100 + index))
            for index, document in enumerate(active_documents)
        ]
    )

    def fake_similarity_search(
        query,
        k=4,
        **kwargs
    ):
        return ordered_candidates[:k]

    monkeypatch.setattr(
        store,
        "similarity_search_with_score",
        fake_similarity_search
    )

    # --------------------------------------------------
    # 5. Search for the active version
    # --------------------------------------------------

    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5
    )

    # --------------------------------------------------
    # 6. Verify the active version is returned
    # --------------------------------------------------

    assert len(results) > 0

    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    assert retrieved_versions == {"2.0"}




def test_search_does_not_starve_source_filter(
    document_store,
    sample_document,
    monkeypatch
):
    # 1. Upload an older version with many chunks
    old_content = (
        "Company Vacation Policy. "
        "Employees receive 20 vacation days per year. "
        "Employees may carry over 5 unused days. "
    ) * 700

    sample_document.write_text(old_content, encoding="utf-8")

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # 2. Upload the active version
    sample_document.write_text(
        """
        Company Vacation Policy

        Employees receive 30 vacation days per year.

        Employees may carry over 10 unused days.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2026-02-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # 3. Get the private FAISS store
    store = document_store._get_store(
        user_id="bob@test.com",
        is_shared=False
    )

    all_documents = [
        document
        for _, document in document_store._get_all_documents(store)
    ]

    old_documents = [
        document for document in all_documents
        if document.metadata.get("document_version") == "1.0"
    ]

    active_documents = [
        document for document in all_documents
        if document.metadata.get("document_version") == "2.0"
    ]

    assert len(old_documents) > 25
    assert len(active_documents) > 0

    # 4. Simulate different document sources
    for document in old_documents:
        document.metadata["source"] = "other_policy.pdf"

    for document in active_documents:
        document.metadata["source"] = "target_policy.pdf"

    # 5. Simulate FAISS returning nonmatching sources first
    ordered_candidates = [
        (document, float(index))
        for index, document in enumerate(old_documents)
    ]

    ordered_candidates.extend(
        [
            (document, float(100 + index))
            for index, document in enumerate(active_documents)
        ]
    )

    def fake_similarity_search(query, k=4, **kwargs):
        return ordered_candidates[:k]

    monkeypatch.setattr(
        store,
        "similarity_search_with_score",
        fake_similarity_search
    )

    # 6. Search using a source filter
    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5,
        source="target_policy.pdf"
    )

    # 7. Verify
    assert len(results) > 0

    retrieved_sources = {
        document.metadata.get("source")
        for document in results
    }

    assert retrieved_sources == {"target_policy.pdf"}

    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    assert retrieved_versions == {"2.0"}



def test_search_returns_shared_and_private_documents(
    document_store,
    sample_document
):
    # 1. Upload a private document
    sample_document.write_text(
        """
        Private Engineering Policy

        Engineers must complete security training annually.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Private Engineering Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="Engineering"
    )

    # 2. Upload a shared document
    sample_document.write_text(
        """
        Shared Company Policy

        All employees must complete annual compliance training.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Shared Company Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR",
        is_shared=True
    )

    # 3. Search both indexes
    results = document_store.search(
        query="annual employee training policy",
        user_id="bob@test.com",
        k=20
    )

    # 4. Verify both documents are searchable
    retrieved_names = {
        document.metadata.get("document_name")
        for document in results
    }

    assert "Private Engineering Policy" in retrieved_names
    assert "Shared Company Policy" in retrieved_names


def test_historical_search_does_not_starve_active_documents(
    document_store,
    sample_document,
    monkeypatch
):
    # 1. Upload a large document effective in the future
    future_content = (
        "Future Vacation Policy. "
        "Employees will receive 40 vacation days per year. "
    ) * 700

    sample_document.write_text(future_content, encoding="utf-8")

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2027-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # 2. Upload an older version
    sample_document.write_text(
        """
        Vacation Policy

        Employees receive 20 vacation days per year.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # 3. Get private FAISS store
    store = document_store._get_store(
        user_id="bob@test.com",
        is_shared=False
    )

    all_documents = [
        document
        for _, document in document_store._get_all_documents(store)
    ]

    future_documents = [
        document for document in all_documents
        if document.metadata.get("document_version") == "2.0"
    ]

    historical_documents = [
        document for document in all_documents
        if document.metadata.get("document_version") == "1.0"
    ]

    assert len(future_documents) > 25
    assert len(historical_documents) > 0

    # 4. Simulate future documents appearing first
    ordered_candidates = [
        (document, float(index))
        for index, document in enumerate(future_documents)
    ]

    ordered_candidates.extend(
        [
            (document, float(100 + index))
            for index, document in enumerate(historical_documents)
        ]
    )

    def fake_similarity_search(query, k=4, **kwargs):
        return ordered_candidates[:k]

    monkeypatch.setattr(
        store,
        "similarity_search_with_score",
        fake_similarity_search
    )

    # 5. Search historical documents
    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5,
        include_historical=True
    )

    # 6. Verify future version is excluded
    assert len(results) > 0

    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    assert retrieved_versions == {"1.0"}


def test_as_of_search_does_not_starve_correct_version(
    document_store,
    sample_document,
    monkeypatch
):
    # 1. Upload version 2.0 with many chunks
    new_content = (
        "Vacation Policy Version 2.0. "
        "Employees receive 30 vacation days per year. "
    ) * 700

    sample_document.write_text(new_content, encoding="utf-8")

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="2.0",
        effective_date="2026-02-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # 2. Upload version 1.0
    sample_document.write_text(
        """
        Vacation Policy Version 1.0.

        Employees receive 20 vacation days per year.
        """,
        encoding="utf-8"
    )

    document_store.load(
        file_path=str(sample_document),
        user_id="bob@test.com",
        document_name="Vacation Policy",
        document_version="1.0",
        effective_date="2026-01-01",
        authority="Official Policy",
        owner_department="HR"
    )

    # 3. Get private FAISS store
    store = document_store._get_store(
        user_id="bob@test.com",
        is_shared=False
    )

    all_documents = [
        document
        for _, document in document_store._get_all_documents(store)
    ]

    new_documents = [
        document for document in all_documents
        if document.metadata.get("document_version") == "2.0"
    ]

    old_documents = [
        document for document in all_documents
        if document.metadata.get("document_version") == "1.0"
    ]

    assert len(new_documents) > 25
    assert len(old_documents) > 0

    # 4. Simulate newer version appearing first
    ordered_candidates = [
        (document, float(index))
        for index, document in enumerate(new_documents)
    ]

    ordered_candidates.extend(
        [
            (document, float(100 + index))
            for index, document in enumerate(old_documents)
        ]
    )

    def fake_similarity_search(query, k=4, **kwargs):
        return ordered_candidates[:k]

    monkeypatch.setattr(
        store,
        "similarity_search_with_score",
        fake_similarity_search
    )

    # 5. Search as of January 15, 2026
    results = document_store.search(
        query="vacation days employees",
        user_id="bob@test.com",
        k=5,
        as_of="2026-01-15"
    )

    # 6. Verify version 1.0 is returned
    assert len(results) > 0

    retrieved_versions = {
        document.metadata.get("document_version")
        for document in results
    }

    assert retrieved_versions == {"1.0"}
