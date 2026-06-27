import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import (
    EMBED_MODEL,
    chunk_id,
    chunk_text,
    content_hash,
    metadata_value,
    mysql_current_ids,
    prepare_content_document,
    prepare_mysql_row,
    prepare_source,
    quote_mysql_identifier,
    source_is_current,
)


def test_chunk_text_uses_overlap():
    assert chunk_text("one two three four five", 3, 1) == [
        "one two three",
        "three four five",
        "five",
    ]


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (3, -1), (3, 3), (3, 4)])
def test_chunk_text_rejects_invalid_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size, overlap)


def test_chunk_ids_are_stable_and_source_specific():
    assert chunk_id("a.pdf", 1, 0) == chunk_id("a.pdf", 1, 0)
    assert chunk_id("a.pdf", 1, 0) != chunk_id("b.pdf", 1, 0)


def test_prepare_source_reads_csv_rows(tmp_path):
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "name,plan,notes\nAda,Pro,Needs monthly report\nGrace,,\n",
        encoding="utf-8",
    )

    ids, documents, metadatas, empty_rows, row_count = prepare_source(csv_path)

    assert len(ids) == 2
    assert documents == [
        "name: Ada; plan: Pro; notes: Needs monthly report",
        "name: Grace",
    ]
    assert metadatas[0]["source_file"] == "customers.csv"
    assert metadatas[0]["source_type"] == "csv"
    assert metadatas[0]["row"] == 2
    assert metadatas[0]["embedding_model"] == EMBED_MODEL
    assert empty_rows == 0
    assert row_count == 3


def test_quote_mysql_identifier_escapes_backticks():
    assert quote_mysql_identifier("ads_search_ready") == "`ads_search_ready`"
    assert quote_mysql_identifier("we`ird") == "`we``ird`"
    with pytest.raises(ValueError):
        quote_mysql_identifier("")


def test_metadata_value_normalizes_supported_chroma_types():
    assert metadata_value(None) is None
    assert metadata_value("") is None
    assert metadata_value(Decimal("12.50")) == 12.5
    assert metadata_value(date(2026, 6, 26)) == "2026-06-26"
    assert metadata_value(b"hello") == "hello"


def test_prepare_mysql_row_uses_embedding_content_as_document():
    row = {
        "id": 42,
        "title": "Road bike",
        "city": "Chennai",
        "price": Decimal("12500.00"),
        "embedding_content": "title: Road bike; category: Cycles",
    }

    doc_id, document, metadata = prepare_mysql_row(row, "embedding_content", "id")

    assert doc_id
    assert document == "title: Road bike; category: Cycles"
    assert metadata["source_type"] == "mysql"
    assert metadata["source_table"] == "ads_search_ready"
    assert metadata["primary_key_column"] == "id"
    assert metadata["primary_key_value"] == 42
    assert metadata["source_content_hash"] == content_hash(document)
    assert metadata["city"] == "Chennai"
    assert metadata["price"] == 12500.0
    assert "embedding_content" not in metadata


def test_prepare_content_document_extracts_json_semantic_text():
    document, metadata = prepare_content_document(
        '{"semantic_text": "bike for sale", "city": "Chennai", "price": 1000}'
    )

    assert document == "bike for sale"
    assert metadata["content_format"] == "json"
    assert metadata["content_city"] == "Chennai"
    assert metadata["content_price"] == 1000


def test_prepare_content_document_normalizes_labeled_text():
    document, metadata = prepare_content_document(
        "Title: Bachelor Mansion for Daily Rent Description: Mansion for Daily Rent. "
        "Listing meta title: Bachelor-Mansion-for-Daily-Rent-Sitra-Coimbatore "
        "Main category: Accommodation & Spaces Subcategory: Mansion "
        "Listing rental duration: Per Day State: Tamil Nadu City: Coimbatore "
        "Locality: Sitra Selected attributes: Speciality Selected attribute values: AC"
    )

    assert "\nDescription: Mansion for Daily Rent." in document
    assert "\nState: Tamil Nadu" in document
    assert "\nCity: Coimbatore" in document
    assert metadata["content_format"] == "labeled_text"
    assert metadata["content_title"] == "Bachelor Mansion for Daily Rent"
    assert metadata["content_main_category"] == "Accommodation & Spaces"
    assert metadata["content_subcategory"] == "Mansion"
    assert metadata["content_listing_rental_duration"] == "Per Day"
    assert metadata["content_state"] == "Tamil Nadu"
    assert metadata["content_city"] == "Coimbatore"
    assert metadata["content_locality"] == "Sitra"


def test_prepare_content_document_flattens_json_without_semantic_text():
    document, metadata = prepare_content_document(
        '{"ad": {"title": "Road bike", "condition": "used"}, "tags": ["cycle"]}'
    )

    assert "ad.title: Road bike" in document
    assert "ad.condition: used" in document
    assert "tags.1: cycle" in document
    assert metadata["content_ad_title"] == "Road bike"


def test_prepare_mysql_row_parses_json_embedding_content():
    row = {
        "id": 7,
        "embedding_content": (
            '{"semantic_text": "title: Road bike; category: Cycles", '
            '"city": "Chennai"}'
        ),
    }

    _, document, metadata = prepare_mysql_row(row, "embedding_content", "id")

    assert document == "title: Road bike; category: Cycles"
    assert metadata["content_format"] == "json"
    assert metadata["content_city"] == "Chennai"


def test_prepare_mysql_row_skips_empty_embedding_content():
    row = {"id": 1, "embedding_content": " "}
    assert prepare_mysql_row(row, "embedding_content", "id") is None


class FakeCollection:
    def __init__(self, ids, documents, model=EMBED_MODEL):
        self.data = {
            "ids": ids,
            "documents": documents,
            "metadatas": [{"embedding_model": model} for _ in ids],
        }

    def get(self, **_kwargs):
        return self.data


class FakeMysqlCollection:
    def __init__(self, ids, hashes, documents=None, model=EMBED_MODEL):
        self.data = {
            "ids": ids,
            "documents": documents or [f"document {doc_id}" for doc_id in ids],
            "metadatas": [
                {"embedding_model": model, "source_content_hash": hash_value}
                for hash_value in hashes
            ],
        }

    def get(self, **_kwargs):
        requested_ids = _kwargs["ids"]
        selected_ids = []
        selected_documents = []
        selected_metadatas = []
        for doc_id, document, metadata in zip(
            self.data["ids"], self.data["documents"], self.data["metadatas"]
        ):
            if doc_id in requested_ids:
                selected_ids.append(doc_id)
                selected_documents.append(document)
                selected_metadatas.append(metadata)
        return {
            "ids": selected_ids,
            "documents": selected_documents,
            "metadatas": selected_metadatas,
        }


def test_mysql_current_ids_matches_model_and_content_hash():
    ids = ["id-1", "id-2"]
    metadatas = [
        {"source_content_hash": "hash-1"},
        {"source_content_hash": "hash-2"},
    ]
    collection = FakeMysqlCollection(["id-1", "id-2"], ["hash-1", "old-hash"])

    assert mysql_current_ids(
        collection,
        ids,
        ["document id-1", "changed document"],
        metadatas,
    ) == {"id-1"}


def test_mysql_current_ids_accepts_legacy_matching_document_without_hash():
    collection = FakeMysqlCollection(
        ["id-1"],
        [None],
        documents=["stored document"],
    )

    assert mysql_current_ids(
        collection,
        ["id-1"],
        ["stored document"],
        [{"source_content_hash": "new-hash"}],
    ) == {"id-1"}


def test_mysql_current_ids_rejects_different_embedding_model():
    collection = FakeMysqlCollection(["id-1"], ["hash-1"], model="other-model")

    assert mysql_current_ids(
        collection,
        ["id-1"],
        ["document id-1"],
        [{"source_content_hash": "hash-1"}],
    ) == set()


def test_source_is_current_matches_ids_text_and_model():
    collection = FakeCollection(["id-1"], ["stored text"])
    assert source_is_current(collection, "source.pdf", ["id-1"], ["stored text"])
    assert not source_is_current(collection, "source.pdf", ["id-1"], ["new text"])


def test_source_is_current_rejects_different_embedding_model():
    collection = FakeCollection(["id-1"], ["stored text"], model="another-model")
    assert not source_is_current(collection, "source.pdf", ["id-1"], ["stored text"])
