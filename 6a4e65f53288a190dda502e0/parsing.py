from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import Document
from llama_parse import LlamaParse

from config import DATA_DIR, LLAMAPARSE_API_KEY


def list_pdf_files(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(path for path in data_dir.glob("*.pdf") if path.is_file())


def parse_pdfs(data_dir: Path = DATA_DIR) -> list[Document]:
    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        return []

    parser = LlamaParse(
        api_key=LLAMAPARSE_API_KEY,
        result_type="markdown",
        verbose=True,
    )

    documents: list[Document] = []
    for pdf_file in pdf_files:
        parsed_docs = parser.load_data(str(pdf_file))
        for document in parsed_docs:
            document.metadata.setdefault("source_file", pdf_file.name)
            document.metadata.setdefault("source_path", str(pdf_file.resolve()))
        documents.extend(parsed_docs)

    return documents
