import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import AgentExecutor, create_react_agent # Removed ReAct
from langchain.prompts import PromptTemplate # Changed import
from langchain.chains import LLMChain # Added LLMChain
from langchain.memory import ConversationBufferMemory # Import memory
from langchain_core.messages import SystemMessage # For potential system message in prompt

# Removed Tool import as it's not directly used by this chain
# from .prometheus_tool import PrometheusQueryTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro") # Default to gemini-1.5-pro

# --- Query Generation Chain Configuration ---

def create_promql_generation_chain(memory: ConversationBufferMemory) -> LLMChain:
    """Creates and returns a chain that generates PromQL from natural language, considering history."""
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0, # Keep temperature low for deterministic PromQL
        convert_system_message_to_human=True
    )

    # Update prompt template to include history and CPU guidance
    # Note: LangChain automatically formats `chat_history` from the memory object.
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=(
        """You are a Prometheus query generation assistant. Given a conversation history and a follow-up user question, translate the user question into a precise PromQL query suitable for the `/api/v1/query` endpoint.
        Focus ONLY on generating the query string based on the user's latest request in the context of the history.
        Do not add any explanation, comments, or surrounding text. Just the query.

        **Query Generation Rules & Defaults:**
        - **Container Count:** When asked "how many containers" or similar, use `count(container_last_seen)`. Do NOT use `rate` or `count_over_time` unless the user explicitly asks about containers seen *only* within a specific past window.
        - **CPU Usage:**
            - ALWAYS calculate CPU usage as a percentage of core usage.
            - Use the formula: `sum(rate(container_cpu_usage_seconds_total{{...}}[interval])) * 100`. Ensure you select appropriate labels inside `{{...}}` based on the user query (e.g., `name="container_name"`).
            - If the user asks for *total* or *overall* CPU usage without specifying a group, apply the `sum()` aggregator as shown above.
            - If the user specifies a time `interval` (e.g., "last 10 minutes"), use it (e.g., `[10m]`).
            - If no interval is specified, **default to `[5m]`**.
            - AVOID using `kube_pod_container_resource_limits_cpu_cores`.
        - **Memory Usage:**
            - When asked for memory usage, aim for a standard unit like MiB or GiB if possible (e.g., `container_memory_usage_bytes / 1024 / 1024`).
            - If asked for a percentage, calculate relative to available resources if the relevant metrics (like limits or node capacity) are known and queryable.
        - **Labels & Aggregation:**
            - **DO NOT generate queries using `label_values(...)`.**
            - If asked for a list of label values (e.g., "list container names"), generate a query grouping by that label (e.g., `count(container_last_seen) by (name)`).
            - **Single Value Requests:** If the user asks for a single numerical value (e.g., "what's the *total* CPU usage?", "give me the network traffic *value*", "show the memory usage *percentage*"), try to construct a query that results in a single scalar or a vector with only one element.
            - Use aggregation functions (`sum`, `avg`, `max`, `count`) with `by (...)` or `without (...)` clauses appropriately to achieve this when the user implies summarization. For example, for total CPU across all containers, use `sum(rate(container_cpu_usage_seconds_total[5m])) * 100`. For average memory, use `avg(container_memory_usage_bytes)`.
            - Prefer queries that directly compute the final value (like percentages) within PromQL.

        **Conversation History:**
        {chat_history}

        **User Question:**
        {input}

        **PromQL Query:**"""
        )
    )

    # Create the LLMChain, passing the memory object
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory, # Pass memory to the chain
        verbose=True # Add verbose logging for debugging
    )

    logger.info("PromQL generation chain with memory created successfully.")
    return chain

# --- Removed Agent Executor and related code --- 

# --- Main Execution (for testing if needed) ---
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found. Cannot run chain.")
    else:
        try:
            # Need to instantiate memory for testing
            test_memory = ConversationBufferMemory(memory_key="chat_history")
            promql_chain = create_promql_generation_chain(memory=test_memory)

            # Example interaction
            query1 = "What is the CPU usage for container 'uptime-kuma'?"
            print(f"\n--- Generating PromQL for: {query1} ---")
            # Use predict for simpler input/output when memory is handled by the chain
            generated_query1 = promql_chain.predict(input=query1)
            print(f"\n--- Generated Query 1 ---\n{generated_query1}")
            # Manually update memory for testing sequence (though chain handles it internally)
            # test_memory.save_context({"input": query1}, {"output": generated_query1})
            # print("\nMemory after Q1:", test_memory.load_memory_variables({}))

            # query2 = "Make that over 10 minutes instead."
            # print(f"\n--- Generating PromQL for: {query2} ---")
            # generated_query2 = promql_chain.predict(input=query2)
            # print(f"\n--- Generated Query 2 ---\n{generated_query2}")

        except Exception as e:
            logger.exception(f"Error running PromQL chain: {e}") 