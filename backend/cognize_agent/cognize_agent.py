import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from dotenv import load_dotenv
from agents import Agent,Runner,trace,function_tool,WebSearchTool,model_settings
from openai.types.responses import ResponseTextDeltaEvent
from typing import Dict
import sendgrid
from docx import Document
import os
from sendgrid.helpers.mail import Mail,Email,To,Content
import google.generativeai as genai
import asyncio
from workers.vector_db import search_in_document, search_across_documents
from schema.schemas import RetrievedChunk,RetrievalOutput,DraftAnswer,VerifiedSource,AttributionOutput,SummaryOutput,IntentOutput,AnswerResponse

def _search_one_document_tool(document_id: str, query: str, top_k: int = 30):
    result = search_in_document(document_id, query, top_k)
    return result

def _search_many_documents_tool(query: str, top_k: int = 5):
    result = search_across_documents(query, top_k)
    return result

search_one_document_tool = function_tool(_search_one_document_tool)
search_many_documents_tool = function_tool(_search_many_documents_tool)

# @function_tool
# def search_many_documents_tool(query: str, top_k: int = 5):
    # return search_across_documents(query, top_k)
# search_one_document_tool = function_tool(
#     func=search_in_document,
#     name="search_one_document",
#     description="Search for relevant text chunks within a specific document using its document_id."
# )

# search_many_documents_tool = function_tool(
#     func=search_across_documents,
#     name="search_many_documents",
#     description="Search for relevant text chunks across all documents when no specific document_id is provided."
# )

retrival_instructions="""
    "You are a retrieval agent for an internal knowledge system. "
    "Your task is to fetch the most relevant text chunks for a given query. "
    "If a document_id is provided, search only inside that document. "
    "If no document_id is provided, search across all documents. "
    "Always return the top 5 chunks in JSON format. "
    "Each chunk must include: document_id, chunk_id, text, and similarity_score. "
    "Do not generate answers, just return retrieved chunks with metadata."
"""

retrival_agent = Agent(
    name="WriterAgent",
    instructions=retrival_instructions,
    model="gpt-4o-mini",
    tools=[search_one_document_tool, search_many_documents_tool],
    output_type=RetrievalOutput
)


answering_instructions="""
You are an answering agent for an internal knowledge system.
You will be provided with the original query, and chunks retrive by the retrival_agent.
Your task is to take a user query and a set of retrieved document chunks, then generate a clear and concise answer.

Requirements:
- Use only the provided chunks as evidence.
- Always cite sources by including document_id and, if available, chunk_id or page number.
- If the chunks do not contain enough information, say “I don’t know.”
- Do not hallucinate or invent information beyond the given chunks.
- Format the output as a short, natural-language answer followed by a list of sources.

"""

answering_agent = Agent(
    name="AnsweringAgent",
    instructions=answering_instructions,
    model="gpt-4o-mini",
    output_type=DraftAnswer
)


sorce_Attributtion_instructions="""
You are a source attribution agent for an internal knowledge system.

You will receive:
- A user query.
- An answer produced by the Answering Agent.
- A set of retrieved chunks used to generate the answer.

Your task:
1. Verify that every statement or claim in the answer has at least one supporting citation.
2. Citations must include: document_id and a short excerpt from the chunk text. Include chunk_id or page if available.
3. If citations are missing, incomplete, or incorrect, repair them by linking the relevant chunks to the claims.
4. Do not alter the meaning of the answer. Only add or fix citations.
5. If no supporting chunk exists for part of the answer, clearly mark it as “unsupported.”

Output format:
- `verified_answer`: the original answer text (unchanged).
- `sources`: a list of verified sources, each with document_id, chunk_id, and excerpt.
- `unsupported`: list of statements from the answer that could not be matched to any chunk (if any).
"""
sorce_Attributtion_agent = Agent(
    name="SourceAttributionAgent",  
    instructions=sorce_Attributtion_instructions,
    model="gpt-4o-mini",
    output_type=AttributionOutput
)

summarization_instructions="""
You are a Summarization Agent for a knowledge system.

Inputs you may receive:
1. A direct document (raw text).
2. Retrieved chunks from a document database.

Your task:
1. Only activate if the user explicitly asks for a summary (e.g., “summarize,” “give in points,” “short summary,” “executive summary”).
2. If a direct document is given → summarize that document.
3. If the document is already uploaded → fetch all its chunks from retrieval and summarize them together.
4. Summarize the content clearly and concisely:
   - Use bullet points if the user says “in points.”
   - Use a short paragraph if the user says “short summary.”
   - Use a polished, professional style if the user says “executive summary.”
5. You may reorganize or add logical points to improve clarity and make the summary perfect and easy to understand.
6. Preserve key information while removing redundancy and irrelevant detail.
7. For multiple chunks/documents, merge them into a single coherent summary.

Output format:
- `summary_type`: bullet_list | short_summary | executive_summary
- `summary`: the clean, concise summary text
"""

summarization_agent = Agent(
    name="SummarizationAgent",  
    instructions=summarization_instructions,
    model="gpt-4o-mini",
    output_type=SummaryOutput
)

intent_classifier_instructions = """
You are an intent classifier for an internal knowledge assistant. 
Input: a single user query (free text).
Output: a JSON object with keys: intent, entities (list), confidence (0-1 or null).

Rules:
- Classify intent into exactly one of: qa, summarize, lookup, compare.
  - qa = general question about content
  - summarize = user explicitly requests summary (words like 'summarize', 'short summary', 'in points', 'executive summary')
  - lookup = user requests exact id/number or asks to 'find policy id', 'policy number', 'document id is <id>'
  - compare = user asks for comparison of two or more items (words like 'compare', 'difference', 'vs', 'versus')
- If a document id appears (patterns like 'document id is <uuid>' or 'doc id <id>'), include it in entities.
- If user mentions two items to compare, include both items as entities.
- Confidence: set between 0.0 and 1.0 if you are certain, else null.
- Output only a JSON matching the schema, nothing else.
"""


intent_classifier_agent = Agent(
    name="IntentClassifierAgent",
    instructions=intent_classifier_instructions,
    model="gpt-4o-mini",
    output_type=IntentOutput
)

# tool1 = retrival_agent.as_tool(tool_name="retrival_agent", tool_description="Fetch relevant document chunks based on a user query and optional document_id.")
# tool2 = answering_agent.as_tool(tool_name="answering_agent", tool_description="Generate a concise answer using provided document chunks as evidence, citing sources.")
# tool3 = sorce_Attributtion_agent.as_tool(tool_name="sorce_Attributtion_agent", tool_description="Verify and correct citations in an answer based on provided document chunks.")
# tool4 = summarization_agent.as_tool(tool_name="summarization_agent", tool_description="Generate a clear and concise summary of a document or retrieved chunks when explicitly requested by the user.")    


# tools = [tool1, tool2, tool3,tool4]
# master_instructions = master_instructions = """
# You are a master agent orchestrating multiple specialized agents to handle user queries.

# Rules for orchestration:
# 1. If the user requests a summary (e.g., 'summarize,' 'give in points,' 'short summary,' 'executive summary'):
#    → Call the Summarization Agent directly and return its output.

# 2. If the user asks a question or seeks information:
#    a. Call the Retrieval Agent first. 
#       - Input: the query and (if provided) document_id.
#       - Output: RetrievalOutput {chunks}.
#    b. Take the 'chunks' from RetrievalOutput and include them **verbatim** in the input to the Answering Agent, along with the original user query.
#       - Input format for Answering Agent:
#         {
#           "query": <user_query>,
#           "chunks": <retrieved_chunks>
#         }
#    c. Take the answer from Answering Agent and pass it, along with the retrieved chunks, into the Source Attribution Agent.
#       - Input format for Source Attribution Agent:
#         {
#           "query": <user_query>,
#           "draft_answer": <answer_from_answering_agent>,
#           "chunks": <retrieved_chunks>
#         }

# 3. Always ensure the chain is: Retrieval → Answering → Source Attribution, unless the user explicitly asked for a summary.

# 4. If no chunks are retrieved:
#    - Do NOT call the Answering Agent.
#    - Respond gracefully: "I could not find relevant information in the documents."
# """

# master_agent = Agent(   
#     name="MasterAgent",  
#     instructions=master_instructions,
#     model="gpt-4o-mini",
#     tools=tools,
# )    

# # document_id = '68ef83f3-4afa-43f4-8f22-8d2213772e1d'
# user_query = "what is Google Cloud Customer Success Playbooks. document id is '68ef83f3-4afa-43f4-8f22-8d2213772e1d'. give me the chunks only."
# # user_query = "Provide a concise summary of the document with ID '11af4319-99e5-437a-9de5-f2691a1fc523', book title is 'short stories fro children"
# async def main():
#         result = await Runner.run(master_agent,user_query)
#         print(result.final_output)

# if __name__ == "__main__":
#     asyncio.run(main())

