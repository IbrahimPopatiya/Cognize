from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Adjust the path

import re
from backend.crud.document_crud import get_dict_from_json_document
from backend.workers.vector_db import search_in_document


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def code_based_summary(text: str, max_length: int = 200) -> str:
        """Simple rule-based summarization: take first few sentences until max_length."""
        sentences = text.split(".")
        summary = []
        count = 0
        for s in sentences:
            if count + len(s) > max_length:
                break
            summary.append(s.strip())
            count += len(s)
        return " • " + "\n • ".join(summary)

def llm_summary(text: str, max_length: int = 200) -> str:
    """Summarize using GPT for fluency."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"Summarize the following text in {max_length} words or less:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def merge_summaries(code_sum: str, llm_sum: str) -> str:
    """Merge code-based and LLM summaries into a hybrid result."""
    return f"🔹 Rule-based Summary:\n{code_sum}\n\n🔹 LLM-enhanced Summary:\n{llm_sum}"




def code_based_explanation(text: str, audience: str) -> str:
    """
    Rule-based explanation (cheap, no LLM).
    Just simplifies the text depending on the audience.
    """
    audience = audience.lower()
    if audience == "executive":
        return f"[Executive Summary]\n{text[:300]}..."
    elif audience == "hr":
        return f"[HR-friendly Explanation]\nFocus on people impact:\n{text}"
    elif audience == "developer":
        return f"[Developer Notes]\nTechnical focus:\n{text}"
    elif audience == "sales":
        return f"[Sales Pitch]\nCustomer value:\n{text}"
    else:
        return f"[General Explanation]\n{text}"


def llm_explanation(text: str, audience: str) -> str:
    """
    Use LLM to rewrite/explain text for the target audience.
    """
    prompt = f"""
    You are an assistant that adapts explanations for different audiences.
    Audience: {audience}

    Rewrite the following text in a way that is easy to understand for this audience.
    Be concise, clear, and keep only relevant details.

    Text:
    {text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def merge_explanations(code_exp: str, llm_exp: str) -> str:
    """
    Merge rule-based and LLM-based explanations into one hybrid answer.
    """
    return f"""[Hybrid Explanation]

Rule-based Summary:
{code_exp}

LLM-Generated Summary:
{llm_exp}
"""

def db_lookup_memory(user_id: str, limit: int = 20) -> list[dict]:
    """
    Fetch past conversation messages for a user from the database.
    Returns a list of messages in the format:
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how can I help?"}
    ]
    """
    # Example mock (replace with your actual DB query)
    history = [
        {"role": "user", "content": "What is our leave policy?"},
        {"role": "assistant", "content": "You get 20 days of annual leave."},
        {"role": "user", "content": "What about sick leave?"},
        {"role": "assistant", "content": "You get 10 days of sick leave."},
    ]
    return history[-limit:]  # get latest N messages


def llm_memory_summary(history: list[dict], max_length: int = 150) -> str:
    """
    Use LLM to summarize a user's conversation history.
    """
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # choose cost-efficient model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": f"Summarize this conversation in {max_length} words:\n{history_text}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def merge_memory(history: list[dict], summary: str) -> dict:
    """
    Combine raw history and LLM summary into a structured format.
    """
    return {
        "raw_history": history,
        "summary": summary
    }




file_path = "../synonyms.json"
SYNONYM_DICT = get_dict_from_json_document(file_path)

def code_based_expansion(query: str) -> List[str]:
    """
    Expand query using predefined synonyms (rule-based).
    """
    expanded = [query]
    for word, synonyms in SYNONYM_DICT.items():
        if re.search(rf"\b{word}\b", query, re.IGNORECASE):
            expanded.extend([query.replace(word, s) for s in synonyms])
    return list(set(expanded))  # remove duplicates


def llm_query_expansion(query: str) -> List[str]:
    """
    Use LLM to generate semantic query variations.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You expand search queries into useful semantic variations."},
            {"role": "user", "content": f"Expand this query into 5 different useful search variations: {query}"}
        ],
        max_tokens=100,
        temperature=0.7
    )
    expansions = response.choices[0].message.content.split("\n")
    return [e.strip("-• ") for e in expansions if e.strip()]


def merge_expansions(code_list: List[str], llm_list: List[str]) -> List[str]:
    """
    Merge code-based and LLM-based expansions, removing duplicates.
    """
    return list(set(code_list + llm_list))






def format_chunks(chunks):
    """
    Format retrieved chunks into readable text grouped by document.
    """
    output = []
    for c in chunks:
        doc_id = c.payload.get("document_id", "unknown_doc")
        text = c.payload.get("text", "")
        output.append(f"[{doc_id}] {text}")
    return "\n".join(output)


def llm_cross_doc_reasoning(query, chunks):
    """
    Use LLM to synthesize a cross-document answer from retrieved chunks.
    """
    context = "\n\n".join(
        [f"[{c.payload.get('document_id', 'unknown_doc')}] {c.payload.get('text', '')}" for c in chunks]
    )

    prompt = f"""
You are a reasoning assistant. 
The user asked: "{query}"

Here are relevant excerpts from multiple documents:
{context}

Please synthesize an answer that:
1. Combines insights across documents.
2. Removes redundancy.
3. Clearly points out if documents agree or conflict.
4. Returns a concise, plain text answer.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message["content"]



def format_context_chunks(chunks):
    """
    Format chunks retrieved from a single document for readability.
    Include reference IDs for traceability.
    """
    output = []
    for c in chunks:
        doc_id = c.payload.get("document_id", "unknown_doc")
        text = c.payload.get("text", "")
        output.append(f"[{doc_id}] {text}")
    return "\n".join(output)


def llm_contextual_answer(query, chunks, chat_history=None):
    """
    Use LLM to synthesize a context-aware answer from chunks.
    Supports multi-turn conversation by including previous chat.
    """
    context_text = "\n\n".join([f"[{c.payload.get('document_id','unknown_doc')}] {c.payload.get('text','')}" for c in chunks])
    
    messages = []
    if chat_history:
        messages.extend(chat_history)
    
    messages.append({
        "role": "user",
        "content": f"""
User asked: "{query}"

Context from document(s):
{context_text}

Please:
1. Answer using the provided context.
2. Reference chunks where relevant.
3. Note if context is insufficient.
4. Be concise, clear, and factual.
"""
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message["content"]




def search_document_context(query, doc_id=None, top_k=5):
    """
    Retrieve top_k relevant chunks from a document using your existing search function.
    """
    # Use existing infrastructure to fetch chunks
    results = search_in_document(document_id=doc_id, query=query or "summary", top_k=top_k)
    
    chunks = []
    for r in results:
        # Keep payload intact for traceability
        chunks.append(r)
    return chunks


### helper functions of sorce attribution tool ###

def format_chunks_with_ids(chunks):
    """
    Format chunks with document_id and chunk_id for traceability.
    """
    output = []
    for c in chunks:
        doc_id = c.payload.get("document_id", "unknown_doc")
        chunk_id = c.payload.get("chunk_id", "unknown_chunk")
        text = c.payload.get("text", "")
        output.append(f"[{doc_id}][{chunk_id}] {text}")
    return "\n".join(output)


def llm_answer_with_sources(query, chunks, chat_history=None):
    """
    Use LLM to synthesize an answer with traceable sources.
    Supports multi-turn conversation.
    """
    context_text = "\n\n".join([f"[{c.payload.get('document_id','unknown_doc')}][{c.payload.get('chunk_id','unknown_chunk')}] {c.payload.get('text','')}" for c in chunks])

    messages = []
    if chat_history:
        messages.extend(chat_history)

    # Explicit job + instructions for the LLM
    messages.append({
        "role": "user",
        "content": f"""
Your job: You are a Contextual QA Assistant with Source Attribution.
Provide accurate, concise answers using only the provided context.
Always include citations to document and chunk IDs where information comes from.
Indicate if context is insufficient or conflicting.

User asked: "{query}"

Context from document(s):
{context_text}

Instructions:
1. Reason over the context and generate a clear answer.
2. Reference each supporting fact with [document_id][chunk_id].
3. Highlight conflicts or missing information.
4. Keep answer factual, professional, and concise.
"""
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message["content"]





#### helper functions for domain adaptation ####

import yaml
with open("../domains.yaml", "r") as f:
    DOMAIN_PROMPTS = yaml.safe_load(f)

def code_based_cleanup(text: str) -> str:
    """
    Lightweight cleanup for readability:
    - Split long sentences
    - Remove duplicate spaces
    - Normalize formatting
    """
    import re

    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    cleaned = ". ".join(sentences)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def llm_domain_adaptation(text: str, domain: str) -> str:
    """
    Use LLM to rewrite text in a domain-specific tone.
    Domain instructions come from domains.yaml.
    """
    instruction = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a domain-adaptive rewriter.\n{instruction}"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message["content"]


def merge_domain_versions(cleaned: str, adapted: str) -> str:
    """
    Merge rule-based cleanup and LLM domain adaptation (hybrid mode).
    """
    return f"Cleaned version:\n{cleaned}\n\nDomain-adapted version:\n{adapted}"