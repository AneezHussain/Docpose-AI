import streamlit as st
import logging
import sys
import os

# Add the parent directory to sys.path to allow imports from agent.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from docker_infra_analyzer.agent import create_docker_agent, GOOGLE_API_KEY
    from docker_infra_analyzer.tools import get_docker_client # To check Docker connection
except ImportError as e:
    st.error(f"Failed to import necessary modules. Ensure the structure is correct and dependencies are installed: {e}")
    st.stop() # Stop execution if imports fail

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Docker Infra Analyzer", layout="wide")

st.title("🐳 Docker Infrastructure Analyzer Agent")
st.caption("Ask questions about your local Docker environment.")

# --- Initialization and Checks ---
@st.cache_resource # Cache the agent executor for efficiency
def get_agent_executor():
    """Creates and returns the agent executor, handling potential errors."""
    try:
        if not GOOGLE_API_KEY:
             st.error("🔴 GOOGLE_API_KEY is not set. Please configure it in app/.env")
             return None
        executor = create_docker_agent()
        return executor
    except ValueError as ve:
        st.error(f"🔴 Agent Initialization Error: {ve}")
        return None
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during agent initialization: {e}")
        logger.error("Agent initialization failed", exc_info=True)
        return None

@st.cache_data # Check Docker connection status once
def check_docker_connection():
    """Checks if the Docker client can connect to the daemon."""
    client = get_docker_client()
    return client is not None

# Check Docker connection status
docker_connected = check_docker_connection()
if not docker_connected:
    st.warning("🟡 Warning: Could not connect to the Docker daemon. Please ensure Docker Desktop or Docker Engine is running. Agent tools may fail.")
else:
    st.success("🟢 Successfully connected to Docker daemon.")


# --- Agent Interaction ---
agent_executor = get_agent_executor()

if agent_executor:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about your Docker setup (e.g., 'List running containers', 'Inspect container my_container')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Thinking..."):
                    # Invoke the agent
                    result = agent_executor.invoke({"input": prompt})
                    assistant_response = result.get("output", "Sorry, I couldn't process that request.")

                # Simulate stream of response with milliseconds delay
                # (Actual streaming depends on Langchain/LLM setup, ReAct shows steps)
                # For now, just display the final result.
                # In a real streaming scenario, you'd update message_placeholder incrementally.
                full_response = assistant_response
                message_placeholder.markdown(full_response)

            except Exception as e:
                logger.error(f"Error invoking agent: {e}", exc_info=True)
                full_response = f"🔴 An error occurred: {e}"
                message_placeholder.error(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.error("🔴 Agent could not be initialized. Please check the logs and ensure your API key and Docker setup are correct.")

# Add a sidebar note
st.sidebar.info(
    "**Note:** This agent interacts with your *local* Docker environment. "
    "Ensure Docker Desktop or Docker Engine is running."
) 