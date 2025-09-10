import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Adjust the path
 
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from dotenv import load_dotenv
from agents import Agent,Runner,trace,function_tool
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict
import sendgrid
from docx import Document
import os
from sendgrid.helpers.mail import Mail,Email,To,Content
import google.generativeai as genai
import asyncio
from workers.vector_db import search_in_document, search_across_documents
from helper_function_tool import *
from typing import List,Dict,Optional
from sqlalchemy.orm import Session
from db.db_utils import get_chat_history, save_message

@function_tool
def document_search_tool(document_id: str, query: str, top_k: int = 5,mode:str  = "both"):
    """
    Your job is to answer questions about a document by using the document_search_tool.

    Search for the most relevant chunks in a given document using semantic similarity.
    Returns top_k results with their text and metadata.

    Modes:
    - "code": return raw chunks only
    - "llm": return LLM-processed concise answer
    - "both": return raw chunks and LLM summary


    Steps:
    1. Call document_search_tool with the user's query and document_id.
    2. Review the retrieved chunks (text + metadata).
    3. Formulate a **concise, clear, factual answer** only based on retrieved content.
    4. If the answer cannot be found in the document, clearly say: "I could not find this in the document."

    Always return your answer in plain text, not JSON.
    """
    results = search_in_document(document_id=document_id, query=query or "summary", top_k=top_k)

    chunks = [r.payload["text"] for r in results]

    if mode == "code":
        # naive summary: first few sentences joined
        return " ".join(chunks)[:800]  

    elif mode == "llm":
        # give raw chunks back, let Master Agent summarize
        return "\n\n".join(chunks)

    elif mode == "hybrid":
        draft_summary = " ".join(chunks)[:500]
        return f"DRAFT SUMMARY:\n{draft_summary}\n\nRAW CHUNKS:\n" + "\n\n".join(chunks)
    

@function_tool
def summarization_tool(text: str, mode: str = "hybrid", max_length: int = 200):
    """
    Your job is to condense long text into a digestible summary.
    
    The tool supports three modes:
    1. "code" → Rule-based summarization (extractive, simple, no LLM cost).
    2. "llm" → Use LLM to generate an abstractive, fluent summary.
    3. "hybrid" → Generate both, compare, and merge into the most accurate + readable form.

    Steps:
    1. If mode="code":
        - Break text into sentences.
        - Select top N sentences covering main points (based on length + keyword density).
    2. If mode="llm":
        - Call LLM with summarization prompt.
    3. If mode="hybrid":
        - First run code summarization.
        - Then run LLM summarization.
        - Combine both: preserve factual accuracy from code, fluency from LLM.
    
    Always return the final summary in plain text, not JSON.
    """
    if mode == "code":
        return code_based_summary(text, max_length=max_length)
    elif mode == "llm":
        return llm_summary(text, max_length=max_length)
    elif mode == "hybrid":
        code_sum = code_based_summary(text, max_length=max_length)
        llm_sum = llm_summary(text, max_length=max_length)
        return merge_summaries(code_sum, llm_sum)
    else:
        return "Invalid mode. Choose from: code, llm, hybrid."


@function_tool
def explanation_tool(text: str, audience: str = "general", mode: str = "hybrid"):
    """
    Your job is to explain content clearly for a given audience (executive, HR, developer, sales, general).
    
    The tool supports three modes:
    1. "code" → Rule-based simplification (replace jargon, shorten sentences).
    2. "llm" → Use LLM to rewrite explanation in audience-specific tone.
    3. "hybrid" → Combine both for factual + audience-friendly result.

    Steps:
    1. If mode="code":
        - Run simple transformations:
            * Split long sentences.
            * Replace technical words with simpler synonyms (basic dictionary).
        - Return cleaned explanation.
    2. If mode="llm":
        - Call LLM with role prompt (e.g., "Explain this to an HR manager").
    3. If mode="hybrid":
        - Run code simplification.
        - Pass simplified text to LLM for tone adaptation.
        - Return refined explanation.
    
    Always return the explanation in plain text, not JSON.
    """
    if mode == "code":
        return code_based_explanation(text, audience=audience)
    elif mode == "llm":
        return llm_explanation(text, audience=audience)
    elif mode == "hybrid":
        simplified = code_based_explanation(text, audience=audience)
        llm_exp = llm_explanation(simplified, audience=audience)
        return merge_explanations(simplified, llm_exp)
    else:
        return "Invalid mode. Choose from: code, llm, hybrid."

@function_tool
def conversation_memory_tool(user_id: str, query: str = None, mode: str = "hybrid"):
    """
    Your job is to maintain and recall conversation history across sessions.

    The tool supports three modes:
    1. "code" → Simple database lookup (store/retrieve past Q&A).
    2. "llm" → Use LLM to generate context-aware memory (merge similar queries, adapt recall).
    3. "hybrid" → Combine deterministic history + LLM contextual recall.

    Steps:
    1. If mode="code":
        - Retrieve user history from memory DB (based on user_id).
        - Return most recent conversations.
    2. If mode="llm":
        - Retrieve history.
        - Pass to LLM for clustering/summarizing old exchanges into context.
    3. If mode="hybrid":
        - Retrieve raw history (code).
        - Summarize & contextualize (llm).
        - Return merged memory snapshot.
    
    Always return memory output in plain text, not JSON.
    """
    if mode == "code":
        return db_lookup_memory(user_id)
    elif mode == "llm":
        history = db_lookup_memory(user_id)
        return llm_memory_summary(history)
    elif mode == "hybrid":
        history = db_lookup_memory(user_id)
        summary = llm_memory_summary(history)
        return merge_memory(history, summary)
    else:
        return "Invalid mode. Choose from: code, llm, hybrid."




@function_tool
def query_expansion_tool(query: str, mode: str = "hybrid") -> List[str]:
    """
    Your job is to expand a user query into multiple variations for better search coverage.

    The tool supports three modes:
    1. "code" → Simple keyword synonym expansion (dictionary/thesaurus based).
    2. "llm" → Use LLM to generate semantic variations (rephrased, contextual).
    3. "hybrid" → Combine deterministic synonyms + LLM paraphrasing.

    Steps:
    1. If mode="code":
        - Use predefined synonym dictionary or keyword expansion logic.
        - Return a list of expanded queries (basic coverage).
    2. If mode="llm":
        - Call LLM with the original query.
        - Generate semantic variations (different wording, phrasing).
        - Return expanded queries list.
    3. If mode="hybrid":
        - Generate both code-based synonyms and LLM-based expansions.
        - Merge results, remove duplicates, return final expanded query list.

    Always return plain text list of queries, not JSON.
    """
    if mode == "code":
        return code_based_expansion(query)
    elif mode == "llm":
        return llm_query_expansion(query)
    elif mode == "hybrid":
        synonyms = code_based_expansion(query)
        llm_variations = llm_query_expansion(query)
        return merge_expansions(synonyms, llm_variations)
    else:
        return ["Invalid mode. Choose from: code, llm, hybrid."]
    




@function_tool
def cross_document_reasoning_tool(query: str, top_k: int = 5, mode: str = "hybrid"):
    """
    Your job is to answer questions that require reasoning across multiple documents.

    The tool supports three modes:
    1. "code" → Retrieve top_k chunks from all documents (semantic search only).
    2. "llm" → Use LLM to analyze retrieved chunks and form a synthesized answer.
    3. "hybrid" → Combine deterministic retrieval + LLM synthesis for best accuracy.

    Steps:
    1. Perform search across all documents using semantic similarity.
    2. If mode="code":
        - Return the raw retrieved chunks grouped by document.
    3. If mode="llm":
        - Pass retrieved chunks to LLM.
        - Ask LLM to merge insights and form a coherent answer.
    4. If mode="hybrid":
        - First return code-based chunks.
        - Then use LLM to structure + summarize them into a final concise answer.
    
    Always return your answer in plain text, not JSON.
    """
    chunks = search_across_documents(query=query, top_k=top_k)

    if mode == "code":
        return format_chunks(chunks)

    elif mode == "llm":
        return llm_cross_doc_reasoning(query, chunks)

    elif mode == "hybrid":
        raw_chunks = format_chunks(chunks)
        llm_answer = llm_cross_doc_reasoning(query, chunks)
        return f"Raw Chunks:\n{raw_chunks}\n\nLLM Synthesized Answer:\n{llm_answer}"

    else:
        return "Invalid mode. Choose from: code, llm, hybrid."




# @function_tool
# def contextual_qa_tool(query: str, doc_id: str = None, top_k: int = 5, mode: str = "hybrid", session_id: Optional[int] = None, db: Optional[Session] = None):
#     """
#     Contextual QA Tool with reasoning and multi-turn support.

#     Parameters:
#     - query: User question.
#     - doc_id: (Optional) Restrict to a specific document.
#     - top_k: Number of relevant chunks to retrieve.
#     - mode: "code" → return chunks, "llm" → LLM answer, "hybrid" → both.
#     - chat_history: Optional list of previous Q&A [{'role':'user','content':'...'}, {'role':'assistant','content':'...'}]

#     Steps:
#     1. Retrieve top_k relevant chunks from the document(s).
#     2. Format chunks for readability.
#     3. Use LLM to reason over the chunks, optionally incorporating previous chat.
#     4. Return plain text answer, with context references.
#     """
#     if db:
#         save_message(session_id=session_id, role="user", content=query, db=db)

#     chat_history = get_chat_history(session_id=session_id,db=db) if db else []

#     chunks = search_document_context(query=query, doc_id=doc_id, top_k=top_k)

#     if mode == "code":
#         return format_context_chunks(chunks)

#     elif mode == "llm":
#         return llm_contextual_answer(query, chunks, chat_history)

#     elif mode == "hybrid":
#         raw_chunks = format_context_chunks(chunks)
#         llm_answer = llm_contextual_answer(query, chunks, chat_history)
#         return f"Raw Chunks:\n{raw_chunks}\n\nLLM Synthesized Answer:\n{llm_answer}"

#     else:
#         return "Invalid mode. Choose from: code, llm, hybrid."
    









@function_tool
def source_attribution_tool(query: str, doc_id: str = None, top_k: int = 5, mode: str = "hybrid", chat_history: list = None):
    """
    Source Attribution Tool (Traceability)

    This tool provides reasoned answers with explicit citations to document chunks.
    Supports multi-turn Q&A, ensuring traceable, trustworthy answers.

    Parameters:
    - query: User question
    - doc_id: Optional document ID to restrict search
    - top_k: Number of relevant chunks to retrieve
    - mode: "code", "llm", or "hybrid"
    - chat_history: Optional previous Q&A for multi-turn context

    Returns:
    - Plain text answer with citations (or raw chunks if mode="code")
    """
    # Step 1: Retrieve relevant chunks
    chunks = search_document_context(query=query, doc_id=doc_id, top_k=top_k)

    # Step 2: Mode-based responses
    if mode == "code":
        return format_chunks_with_ids(chunks)

    elif mode == "llm":
        return llm_answer_with_sources(query, chunks, chat_history)

    elif mode == "hybrid":
        raw_chunks = format_chunks_with_ids(chunks)
        llm_answer = llm_answer_with_sources(query, chunks, chat_history)
        return f"Raw Chunks:\n{raw_chunks}\n\nLLM Synthesized Answer with Sources:\n{llm_answer}"

    else:
        return "Invalid mode. Choose from: code, llm, hybrid."
    




@function_tool
def domain_adaptive_tool(text: str, domain: str = "general", mode: str = "hybrid"):
    """
    Domain-Adaptive Tool
    
    Adapts explanations based on domain (general, legal, medical, technical, simple).
    Instructions are stored in domains.yaml for flexibility.

    Modes:
    1. "code" → Lightweight cleanup (split long sentences, remove redundancy).
    2. "llm" → Use LLM to rewrite text in domain-specific tone.
    3. "hybrid" → Cleanup first, then refine with LLM.

    Returns plain text, not JSON.
    """
    if mode == "code":
        return code_based_cleanup(text)
    elif mode == "llm":
        return llm_domain_adaptation(text, domain=domain)
    elif mode == "hybrid":
        simplified = code_based_cleanup(text)
        llm_exp = llm_domain_adaptation(simplified, domain=domain)
        return merge_domain_versions(simplified, llm_exp)
    else:
        return "❌ Invalid mode. Choose from: code, llm, hybrid."
