
import asyncio
from agents import Runner
from schema.schemas import RetrievalOutput,DraftAnswer, SummaryOutput, AttributionOutput, IntentOutput  

async def run_summary_pipeline(query: str, summarization_agent):
    result = await Runner.run(summarization_agent, query)
    return result.final_output


async def run_lookup_pipeline(query: str, retrival_agent, answering_agent, sorce_Attributtion_agent):
    retrieval_result = await Runner.run(retrival_agent, query)
    chunks = retrieval_result.final_output.chunks

    if not chunks:
        return {"answer": "No relevant information found.", "sources": []}

    chunks_text = "\n\n".join([f"[{c.document_id}:{c.chunk_id}] {c.text}" for c in chunks])
    answering_input = f"User Query: {query}\n\nRetrieved Chunks:\n{chunks_text}"
    answer_result = await Runner.run(answering_agent, answering_input)

    attribution_input = f"User Query: {query}\n\nDraft Answer: {answer_result.final_output.answer}\n\nChunks:\n{chunks_text}"
    attribution_result = await Runner.run(sorce_Attributtion_agent, attribution_input)

    return attribution_result.final_output


async def run_compare_pipeline(query: str, retrival_agent, answering_agent, sorce_Attributtion_agent):
    retrieval_result = await Runner.run(retrival_agent, query)
    chunks = retrieval_result.final_output.chunks

    if not chunks:
        return {"answer": "Nothing to compare – no data found.", "sources": []}

    chunks_text = "\n\n".join([f"[{c.document_id}:{c.chunk_id}] {c.text}" for c in chunks])
    comparison_input = f"Compare as per user query: {query}\n\nChunks:\n{chunks_text}"

    answer_result = await Runner.run(answering_agent, comparison_input)

    attribution_input = f"User Query: {query}\n\nDraft Answer: {answer_result.final_output.answer}\n\nChunks:\n{chunks_text}"
    attribution_result = await Runner.run(sorce_Attributtion_agent, attribution_input)

    return attribution_result.final_output


async def run_qa_pipeline(query, retrival_agent, answering_agent, sorce_Attributtion_agent):
    retrieval_result = await Runner.run(retrival_agent, query)
    chunks = retrieval_result.final_output.chunks

    if not chunks:
        return {"answer": "I could not find relevant information in the documents.", "sources": []}

    chunks_text = "\n\n".join([f"[{c.document_id}:{c.chunk_id}] {c.text}" for c in chunks])
    answering_input = f"User Query: {query}\n\nRetrieved Chunks:\n{chunks_text}"
    answer_result = await Runner.run(answering_agent, answering_input)

    attribution_input = f"User Query: {query}\n\nDraft Answer: {answer_result.final_output.answer}\n\nChunks:\n{chunks_text}"
    attribution_result = await Runner.run(sorce_Attributtion_agent, attribution_input)

    return attribution_result.final_output


