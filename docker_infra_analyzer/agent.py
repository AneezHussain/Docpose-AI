import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate # Optional: If customizing prompt

# Load environment variables from .env file in the parent directory
# Assumes .env is in the 'app' directory and this script is in 'app/docker_infra_analyzer'
# dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env') # Old path
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Look for .env in the current directory
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro") # Default if not set

# Dynamically import tools to avoid circular dependencies if tools need env vars too
try:
    from .tools import docker_tools
except ImportError as e:
    logger.error(f"Failed to import docker_tools: {e}")
    # Handle the error appropriately, maybe raise an exception or use dummy tools
    docker_tools = [] # Avoid crashing if tools can't be imported

def create_docker_agent():
    """Creates and returns a Langchain agent configured with Docker tools and Gemini LLM."""

    if not GOOGLE_API_KEY:
        msg = "GOOGLE_API_KEY environment variable not found. Please set it in your .env file."
        logger.error(msg)
        raise ValueError(msg)

    if not GEMINI_MODEL:
        msg = "GEMINI_MODEL environment variable not found. Please set it in your .env file."
        logger.warning(msg) # Warn but proceed with default if needed
        # Or raise ValueError(msg) if model must be specified

    logger.info(f"Initializing Gemini LLM with model: {GEMINI_MODEL}")
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1, # Lower temperature for more deterministic infrastructure tasks
        )
    except Exception as e:
        msg = f"Failed to initialize ChatGoogleGenerativeAI: {e}"
        logger.error(msg, exc_info=True)
        raise ValueError(msg)

    if not docker_tools:
         logger.warning("No Docker tools were loaded. The agent may not be able to interact with Docker.")
         # Depending on requirements, you might want to raise an error here

    # Optional: Customize the ReAct prompt if needed
    # system_prompt = """You are a helpful Docker infrastructure assistant..."""
    # prompt = ChatPromptTemplate.from_messages(
    #     [("system", system_prompt),
    #      ("human", "{input}"),
    #      ("placeholder", "{agent_scratchpad}")]
    # )
    # agent_executor = create_react_agent(llm, docker_tools, prompt=prompt)

    logger.info(f"Creating ReAct agent with {len(docker_tools)} tools.")
    try:
        agent_executor = create_react_agent(llm, docker_tools)
        logger.info("Agent created successfully.")
        return agent_executor
    except Exception as e:
        msg = f"Failed to create ReAct agent: {e}"
        logger.error(msg, exc_info=True)
        raise RuntimeError(msg) # Use RuntimeError for agent creation failures

# Example usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # More verbose logging for testing
    logger.info("Running agent.py directly for testing...")
    try:
        agent = create_docker_agent()
        test_query = "List the running docker containers"
        logger.info(f"Invoking agent with query: '{test_query}'")
        result = agent.invoke({"input": test_query})
        logger.info(f"Agent Result:\n{result}")

        test_query_inspect = "Inspect container c187b000d6e7" # Replace with a valid container ID/name on your system
        logger.info(f"Invoking agent with query: '{test_query_inspect}'")
        result_inspect = agent.invoke({"input": test_query_inspect})
        logger.info(f"Agent Inspect Result:\n{result_inspect}")

    except (ValueError, RuntimeError, ImportError) as e:
        logger.error(f"Agent test failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during testing: {e}", exc_info=True) 