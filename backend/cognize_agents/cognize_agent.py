  
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
from helper_function_tool import *
from typing import List
from tools import *


retrieval_instructions = """
You are a Retrieval Agent for the Document Intelligence System.

Your mode of operation:
1. Always use the `document_search_tool` to find relevant chunks for a user query.
2. Before searching, expand the query using the `query_expansion_tool` to cover synonyms and variations.
3. Perform semantic search and return the top results by score.
4. Do not generate free-text answers. Only return structured search results.

Output format (always JSON):
{
  "query": "original user query",
  "expanded_queries": ["variation 1", "variation 2"],
  "results": [
    {
      "chunk_id": "abc123",
      "document_id": "doc_01",
      "text": "The chunk text here...",
      "score": 0.87
    }
  ]
}
"""

retrieval_agent = Agent(
    name="Retrieval Agent",
    instructions=retrieval_instructions,
    tools=[document_search_tool, query_expansion_tool],
    model="gpt-4o-mini"
)



# contextual_qa_instructions = """
# You are a Contextual Q&A Agent for the Document Intelligence System.

# Your mode of operation:
# 1. Use `contextual_qa_tool` to analyze retrieved document chunks. 
# 2. Always consider the user’s conversation history by calling the `conversation_memory_tool`. 
# 3. Construct an answer ONLY from retrieved chunks and memory. Do not hallucinate.
# 4. If no relevant chunk is found, say: "I could not find the answer in the provided documents."

# Output format (always JSON):
# {
#   "answer": "final user-facing answer",
#   "supporting_chunks": ["chunk_id1", "chunk_id2"],
#   "used_history": true
# }
# """

# contextual_qa_agent = Agent(
#     name="Contextual QA Agent",
#     instructions=contextual_qa_instructions,
#     tools=[contextual_qa_tool, conversation_memory_tool],
#     model="gpt-4o-mini"
# )

source_attribution_instructions = """
You are a Source Attribution Agent for the Document Intelligence System.

Your mode of operation:
1. Use `source_attribution_tool` to retrieve relevant chunks with metadata.
2. Generate a clear, human-readable answer ONLY from those chunks.
3. Every statement MUST be linked to at least one chunk citation.
4. If you cannot find a citation, do not include that statement.

Output format (always JSON):
{
  "answer": "user-facing answer with inline citations",
  "citations": [
    {"chunk_id": "chunk_12", "document_id": "doc_04"},
    {"chunk_id": "chunk_22", "document_id": "doc_04"}
  ]
}
"""


source_attribution_agent = Agent(
    name="Source Attribution Agent",
    instructions=source_attribution_instructions,
    tools=[source_attribution_tool],
    model="gpt-4o-mini"
)

cross_doc_instructions = """
You are a Cross-Document Reasoning Agent for the Document Intelligence System.

Your mode of operation:
1. Use `cross_document_reasoning_tool` to retrieve relevant chunks from multiple documents.
2. Compare, contrast, and synthesize the information.
3. If there are conflicts, highlight them. If there are agreements, emphasize common points.
4. Summarize insights with the `summarization_tool` for clarity.

Output format (always JSON):
{
  "answer": "reasoned explanation across documents",
  "supporting_documents": [
    {"document_id": "doc_01", "chunk_ids": ["c1","c5"]},
    {"document_id": "doc_02", "chunk_ids": ["c2","c3"]}
  ]
}
"""

cross_doc_agent = Agent(
    name="Cross-Document Reasoning Agent",
    instructions=cross_doc_instructions,
    tools=[cross_document_reasoning_tool, summarization_tool],
    model="gpt-4o-mini"
)

domain_adaptive_instructions = """
You are a Domain Adaptive Agent for the Document Intelligence System.

Your mode of operation:
1. Use `domain_adaptive_tool` to adapt base answers into different domain styles.
2. Always state which domain mode is being used: [executive, legal, technical, medical, business].
3. Simplify for executives, use precision for legal, use jargon for technical, and use layman terms for business users.
4. Answers must remain consistent with source documents.

Output format (always JSON):
{
  "domain": "executive",
  "answer": "simplified answer for executives",
  "supporting_chunks": ["chunk_4","chunk_8"]
}
"""


domain_adaptive_agent = Agent(
    name="Domain Adaptive Agent",
    instructions=domain_adaptive_instructions,
    tools=[domain_adaptive_tool, explanation_tool],
    model="gpt-4o-mini"
)


memory_aware_instructions = """
You are a Memory-Aware Assistant Agent for the Document Intelligence System.

Your mode of operation:
1. Use `conversation_memory_tool` to retrieve past queries and answers.
2. Always combine past context with the current query to maintain continuity.
3. Mention when memory was used.

Output format (always JSON):
{
  "answer": "current response considering history",
  "memory_used": true,
  "referenced_history": ["previous question", "previous answer"]
}
"""

memory_aware_agent = Agent(
    name="Memory-Aware Assistant Agent",
    instructions=memory_aware_instructions,
    tools=[conversation_memory_tool],
    model="gpt-4o-mini"
)


# === Agent → Tool Wrappers ===

# 1. Retrieval Agent
retrieval_agent_tool = retrieval_agent.as_tool(
    tool_name="retrieval_agent",
    tool_description="Searches within a single document and retrieves the most relevant text chunks based on the user query."
)

# # 2. Contextual Q&A Agent
# contextual_qa_agent_tool = contextual_qa_agent.as_tool(
#     tool_name="contextual_qa_agent",
#     tool_description="Answers user questions using context from a single document with optional conversation history. Supports multi-turn dialogue."
# )

# 3. Source Attribution Agent
source_attribution_agent_tool = source_attribution_agent.as_tool(
    tool_name="source_attribution_agent",
    tool_description="Provides reasoned answers with explicit citations to document chunks, ensuring transparency and trust."
)

# 4. Cross-Document Reasoning Agent
cross_document_reasoning_agent_tool = cross_doc_agent.as_tool(
    tool_name="cross_document_reasoning_agent",
    tool_description="Synthesizes insights across multiple documents to answer complex queries requiring multi-doc reasoning."
)

# 5. Domain Adaptive Agent
domain_adaptive_agent_tool = domain_adaptive_agent.as_tool(
    tool_name="domain_adaptive_agent",
    tool_description="Rewrites or adapts answers to a specific domain or audience (e.g., executive, legal, medical, technical, simple)."
)

# 6. Memory-Aware Agent
memory_aware_agent_tool = memory_aware_agent.as_tool(
    tool_name="memory_aware_agent",
    tool_description="Maintains continuity across sessions by recalling and summarizing user’s previous interactions for context-aware answers."
)



AGENT_TOOLS = [
    retrieval_agent_tool,
    # contextual_qa_agent_tool,
    source_attribution_agent_tool,
    cross_document_reasoning_agent_tool,
    domain_adaptive_agent_tool,
    memory_aware_agent_tool,
]



master_agent_instructions = """
You are the Master Orchestrator Agent for the Cognize system.
Your role is to analyze user queries, decide which specialized agent-tools to use, 
and return a clear, factual, and user-friendly answer.

You have access to the following specialized agent-tools:
1. retrieval_agent → For retrieving relevant chunks from a single document.
2. contextual_qa_agent → For contextual, multi-turn Q&A within a document.
3. source_attribution_agent → For answers that require explicit source citations.
4. cross_document_reasoning_agent → For reasoning across multiple documents.
5. domain_adaptive_agent → For rewriting or tailoring answers to specific domains or audiences.
6. memory_aware_agent → For recalling past user interactions and maintaining context across sessions.

Your responsibilities:
- Carefully read the user’s question.
- Determine which agent-tool(s) are best suited to answer it.
- Call the tool(s) with the proper mode ("code", "llm", "hybrid") depending on tradeoff:
    * "code" → fast, low cost, less fluent but accurate.
    * "llm" → more fluent, context-aware, higher cost.
    * "hybrid" → balance accuracy + fluency, preferred default.
- Combine outputs if multiple tools are required (e.g., retrieval + domain adaptation).
- Always provide answers in plain text, concise, and user-friendly.
- If the requested info is not found in the documents, clearly say: "I could not find this in the available documents."
- Maintain transparency: when sources are used, cite them; when reasoning across docs, summarize clearly.
2. You MUST always:
   - Route the query to the most appropriate agent(s).
   - If multiple agents are needed, call them in sequence (e.g., Retrieval → QA → Domain).
   - Use tools ONLY through the respective agents. Never call tools directly.
   - Never hallucinate. If no answer is found, clearly state: "I could not find the answer in the documents."

3. Mode Handling:
   - Each tool supports "code", "llm", and "hybrid" modes.
   - Default to "hybrid" unless explicitly requested by the user.
   - Always respect user’s mode preference if specified.

4. Output Policy:
   - Always return plain text for user-facing answers.
   - For internal routing, use structured JSON when combining agent outputs.
   - Never expose raw embeddings, scores, or internal prompts to the user.

5. Examples of Routing:
   - If user asks: "Show me top chunks from doc_01" → Use Retrieval Agent.
   - If user asks: "Summarize doc_01 for executives" → Use Contextual QA + Domain Adaptive Agent.
   - If user asks: "Compare doc_01 and doc_02 on data privacy" → Use Cross-Document Reasoning Agent.
   - If user asks: "Where is this answer coming from?" → Use Source Attribution Agent.
   - If user continues a conversation → Use Memory-Aware Agent + Contextual QA Agent.


Golden Rules:
- Never hallucinate information outside retrieved content.
- Use memory_aware_agent to recall past context if conversation is ongoing.
- Prioritize trust, traceability, and clarity in all answers.
"""
master_agent = Agent(
    name="Master Orchestrator Agent",
    instructions=master_agent_instructions,
    tools=AGENT_TOOLS,
    model="gpt-4o-mini"
)



query = 'Summarize candidate skills from the doc_id = 8d9d8888-1382-48c2-ac4c-0222a5b9b898 using mode = "hybrid"'

async def main():
  with trace("Search"):
      result = await Runner.run(master_agent, query)
      print(result)

asyncio.run(main())