import streamlit as st
import os
import logging
import re
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Import message types

# Configure logging (Moved Up)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import from the 'app' directory
import sys
import os # Make sure os is imported

# Add the workspace root to sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
    logger.info(f"Added {workspace_root} to sys.path") # Add log

try:
    # Import the specific chain creation function and the tool
    from app.prometheus_agent import create_promql_generation_chain
    from app.prometheus_tool import PrometheusQueryTool
except ImportError as e:
    st.error(f"Failed to import dependencies: {e}. Ensure app structure is correct.")
    st.stop()

# Load environment variables
load_dotenv()

# Check for necessary API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")

# --- Memory Initialization --- (Moved outside main for clarity)
# Use Streamlit session state to persist memory across reruns
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    logger.info("Initialized ConversationBufferMemory in session state.")

# Add helper function for conversion and formatting
def format_tool_output(prompt: str, query: str, result: str) -> str:
    """Analyzes the query and result to provide human-readable output with conversions."""
    value = None
    unit = ""
    # Attempt to parse the result as a float if it looks like a number
    try:
        # Check if result is not one of the placeholder/error strings from the tool
        if result and not result.startswith("("):
            value = float(result)
    except (ValueError, TypeError):
        # Result wasn't a simple number, use standard formatting
        logger.debug(f"Result '{result}' is not a simple number, using standard format.")
        pass # Fall through to standard formatting

    if value is not None:
        # --- Unit Detection & Conversion (Heuristics based on query) ---
        # Bytes conversion
        if "_bytes" in query and "rate(" not in query: # Avoid converting rate of bytes
            if value > (1024 * 1024 * 1024):
                value /= (1024 * 1024 * 1024)
                unit = "GiB"
            elif value > (1024 * 1024):
                value /= (1024 * 1024)
                unit = "MiB"
            elif value > 1024:
                value /= 1024
                unit = "KiB"
            else:
                unit = "bytes"
            return f"The result is approximately **{value:.2f} {unit}**."

        # CPU Percentage (check if query looks like our percentage calculation)
        if "rate(container_cpu_usage_seconds_total" in query and "* 100" in query:
            unit = "% CPU"
            # Extract context (e.g., container name) if possible - simplistic approach
            context = ""
            # Use the moved import re
            # Regex to find name="..." with different quote styles
            # Use triple quotes for the raw string to avoid conflicts with internal quotes
            match = re.search(r"""name="([^"]*)"|'name'='([^']*)'|"name"="([^"]*)""", query)
            if match:
                # Get the first non-None capture group
                container_name = next((g for g in match.groups() if g is not None), None)
                if container_name:
                     context = f" for container '{container_name}'"
            return f"The calculated CPU usage{context} is **{value:.2f}{unit}**."

        # Generic value
        return f"The calculated value is **{value:.2f}**."

    else:
        # Standard formatting if extraction failed or wasn't applicable
        return f"**Query:**\n```promql\n{query}\n```\n**Result:**\n```\n{result}\n```"

def main():
    st.set_page_config(page_title="Prometheus Monitoring Query", layout="wide")
    st.title("📊 Prometheus Monitoring Query Generator (with Memory)")
    st.caption(f"Using Prometheus at: {PROMETHEUS_URL or 'Not Configured'}")

    if not GOOGLE_API_KEY:
        st.error("🚨 Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
        st.stop()

    if not PROMETHEUS_URL:
        st.warning("⚠️ Prometheus URL not found. Please set the PROMETHEUS_URL environment variable in your .env file. Using default 'http://localhost:9090' for tool.")

    # Access memory from session state
    memory = st.session_state.chat_memory

    # Initialize the LLM chain with memory (cached)
    @st.cache_resource
    def get_chain(_memory): # Pass memory to caching function
        try:
            # Pass the persistent memory object to the chain creation function
            return create_promql_generation_chain(memory=_memory)
        except Exception as e:
            st.error(f"Failed to initialize the PromQL generation chain: {e}")
            logger.exception("Chain Initialization Error")
            return None

    # Instantiate the tool (no caching needed as it's lightweight)
    @st.cache_resource
    def get_tool():
        return PrometheusQueryTool()

    # Get chain instance, passing the session state memory
    promql_chain = get_chain(memory)
    prometheus_tool = get_tool()

    if promql_chain is None or prometheus_tool is None:
        st.stop()

    # Display chat messages from history (now using memory object)
    # Check if memory has messages before iterating
    if memory.chat_memory and memory.chat_memory.messages:
        for message in memory.chat_memory.messages:
            with st.chat_message(message.type): # Use message.type ('human' or 'ai')
                # Handle potential dict content for AI messages
                if isinstance(message.content, str):
                     # Display AI response string directly if it's just a string (e.g., error message)
                     st.markdown(message.content)
                # Check if it's an AI message containing our dict structure
                elif message.type == 'ai' and isinstance(message.content, str):
                     # Attempt to parse the string content if needed, but ideally store dict directly
                     # For simplicity, let's assume AI content is stored as the dict string for now
                     # A better approach would be custom memory class or different storage format
                     # We'll display the raw string for now, as parsing it back is complex here.
                     st.markdown(message.content) # Display raw string content of AI message
                else:
                    st.markdown(str(message.content)) # Fallback for other content types

    # React to user input
    if prompt := st.chat_input("Ask about your Prometheus metrics..."):
        # Display user message is handled implicitly by memory
        # We don't need to manually add user message to st.session_state.messages anymore

        # Generate PromQL query using the chain (1 LLM call)
        # Chain now automatically uses memory
        generated_query = ""
        tool_output = ""
        final_output = ""
        try:
            with st.spinner("Generating PromQL query..."):
                # Use predict, memory is handled internally by the chain
                generated_query = promql_chain.predict(input=prompt).strip()
                if not generated_query:
                    raise ValueError("LLM failed to generate a query.")
                logger.info(f"Generated PromQL: {generated_query}")

        except Exception as e:
            logger.exception(f"Error generating PromQL: {e}")
            error_message = f"An error occurred generating the query: {e}"
            with st.chat_message("assistant"):
                st.error(error_message)
            # Add error to memory manually if needed, although chain might handle errors
            # memory.save_context({"input": prompt}, {"output": error_message}) # Example
            st.stop()

        # Decide whether to extract only the value
        extract_value = False
        prompt_lower = prompt.lower()
        # Keywords suggesting a single value is desired
        value_keywords = ["value", "percentage", "how much", "how many", "count", "total", "average"]
        if any(keyword in prompt_lower for keyword in value_keywords):
             extract_value = True
        # NEW: Always extract value for simple byte metrics to allow conversion
        elif "_bytes" in generated_query and "rate(" not in generated_query and " by (" not in generated_query:
            logger.info("Query contains '_bytes' metric, setting extract_value=True for conversion.")
            extract_value = True
        # Check if query uses aggregation without grouping (likely single value)
        elif any(agg in generated_query for agg in ["sum(", "avg(", "count(", "max(", "min("]) and "by (" not in generated_query:
             extract_value = True

        # Execute the query using the tool
        try:
            logger.info(f"Attempting to execute query: {generated_query}") # Log query
            logger.info(f"Extract value only flag set to: {extract_value}") # Log flag

            with st.spinner(f"Executing query: `{generated_query}`..." + (" (extracting value)" if extract_value else "")):
                # Pass the arguments as a dictionary matching the args_schema
                tool_input_dict = {
                    "query": generated_query,
                    "extract_value_only": extract_value
                }
                tool_output = prometheus_tool.run(tool_input_dict)
                logger.info(f"Raw tool output: {tool_output}") # Log raw tool output

        except Exception as e:
            logger.exception(f"Error executing Prometheus query: {e}")
            error_message = f"An error occurred executing the query `{generated_query}`: {e}"
            # Format output for display and memory
            assistant_response_content = f"**Query:**\n```promql\n{generated_query}\n```\n**Error:**\n```\n{error_message}\n```"
            with st.chat_message("assistant"):
                st.error(error_message)
            memory.chat_memory.add_ai_message(assistant_response_content)
            st.stop()

        # Format successful output using the helper function
        final_output = format_tool_output(prompt, generated_query, tool_output)

        with st.chat_message("assistant"):
            st.markdown(final_output)

        # Manually save the final processed output to memory
        memory.chat_memory.add_ai_message(final_output)

        # Rerun to clear the input box and reflect memory changes
        # st.rerun() # Optional: uncomment if you want input cleared after submission

if __name__ == "__main__":
    main() 