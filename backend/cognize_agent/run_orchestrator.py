from cognize_agent.pipeline import run_summary_pipeline, run_lookup_pipeline, run_compare_pipeline, run_qa_pipeline
from agents import Runner
from cognize_agent.cognize_agent import retrival_agent, answering_agent, sorce_Attributtion_agent, summarization_agent, intent_classifier_agent

PIPELINE_REGISTRY = {
    "summarize": run_summary_pipeline,
    "lookup": run_lookup_pipeline,
    "compare": run_compare_pipeline,
    "qa": run_qa_pipeline   # default fallback
}


async def run_orchestrator(query: str, agents):
    """
    agents = {
        "retrival": retrival_agent,
        "answering": answering_agent,
        "attribution": sorce_Attributtion_agent,
        "summarization": summarization_agent,
        "intent": intent_classifier_agent
    }
    """

    # 1. classify intent
    intent_result = await Runner.run(agents["intent"], query)
    intent_obj = intent_result.final_output
    intent = intent_obj.intent or "qa"   # default fallback

    print(f"[INTENT] {intent}")

    # 2. pick pipeline
    pipeline = PIPELINE_REGISTRY.get(intent, run_qa_pipeline)
    
    # 3. run pipeline with proper agents
    return await pipeline(query, agents["retrival"], agents["answering"], agents["attribution"])



async def main():
    agents_dict = {
        "retrival": retrival_agent,
        "answering": answering_agent,
        "attribution": sorce_Attributtion_agent,
        "summarization": summarization_agent,
        "intent": intent_classifier_agent
    }
    query = "summarize document 68ef83f3-4afa-43f4-8f22-8d2213772e1d"
    result = await run_orchestrator(query, agents_dict)
    print(result)
