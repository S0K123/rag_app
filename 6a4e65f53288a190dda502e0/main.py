from __future__ import annotations

from agents import DEFAULT_REPORT_REQUEST, generate_markdown_report
from config import DATA_DIR, REPORT_PATH
from database import build_or_load_index
from parsing import list_pdf_files, parse_pdfs


def save_report(markdown_content: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown_content, encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list_pdf_files(DATA_DIR)
    if not pdf_files:
        empty_report = (
            "# Report\n\n"
            "No PDF files were found in the `data/` directory.\n\n"
            "Add one or more PDF files and rerun `python main.py`."
        )
        save_report(empty_report)
        print(f"No PDFs found. Placeholder report saved to: {REPORT_PATH}")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Parsing with LlamaParse...")
    documents = parse_pdfs(DATA_DIR)
    print(f"Parsed {len(documents)} document chunk(s). Building vector index...")

    index = build_or_load_index(documents=documents, rebuild=True)

    print("Generating Markdown report with Gemini...")
    report = generate_markdown_report(
        index=index,
        report_request=DEFAULT_REPORT_REQUEST,
    )

    save_report(report)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
