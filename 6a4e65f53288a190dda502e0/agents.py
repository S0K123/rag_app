from __future__ import annotations

from llama_index.core import PromptTemplate, VectorStoreIndex


DEFAULT_REPORT_REQUEST = """
Create a comprehensive Markdown report from the retrieved PDF content.

Requirements:
- Start with a clear title.
- Add an executive summary.
- Extract the most important findings, decisions, dates, metrics, entities, and risks.
- Include a section for notable supporting evidence from the source material.
- Include a section for missing or ambiguous information.
- End with actionable recommendations or next steps.
- Use only the retrieved context and do not invent facts.
""".strip()


REPORT_PROMPT = PromptTemplate(
    """
You are a senior AI reporting analyst.
Your task is to read the retrieved PDF context and produce a polished Markdown report.

Instructions:
- Rely only on the provided context.
- Do not hallucinate or fill gaps with outside knowledge.
- Write in clear Markdown with meaningful headings and concise bullet points.
- Surface source-specific facts when available.
- If information is missing, state that explicitly.
- Make the report useful for business or technical stakeholders.

Context information:
---------------------
{context_str}
---------------------

User request:
{query_str}

Markdown report:
""".strip()
)


def create_report_query_engine(index: VectorStoreIndex):
    return index.as_query_engine(
        similarity_top_k=6,
        text_qa_template=REPORT_PROMPT,
    )


def generate_markdown_report(
    index: VectorStoreIndex,
    report_request: str = DEFAULT_REPORT_REQUEST,
) -> str:
    query_engine = create_report_query_engine(index)
    response = query_engine.query(report_request)
    return str(response)
