from __future__ import annotations

from collections.abc import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import CHROMA_DB_DIR


DEFAULT_COLLECTION_NAME = "pdf_rag_documents"


def get_chroma_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    rebuild: bool = False,
) -> Collection:
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(collection_name)


def build_or_load_index(
    documents: Sequence[Document] | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    rebuild: bool = False,
) -> VectorStoreIndex:
    collection = get_chroma_collection(
        collection_name=collection_name,
        rebuild=rebuild,
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if documents:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            list(documents),
            storage_context=storage_context,
            show_progress=True,
        )

    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    raise ValueError(
        "No documents were provided and no persisted ChromaDB index exists."
    )
