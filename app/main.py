import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from openai import OpenAI
import traceback
from sqlalchemy import text
import yaml

# Import database modules
from database import get_db, create_tables
from repository import get_latest_conversation, add_message, clear_conversation, get_all_conversations, get_conversation, create_conversation, delete_conversation
# Import Docker Compose generator
from docker_agent import get_docker_supervisor

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page config
st.set_page_config(
    page_title="DocPose AI",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stTextInput, .stTextArea {
        background-color: white;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .logo-text {
        font-weight: bold;
        font-size: 2.5rem;
        margin: 0;
        padding: 0;
        color: #1E3A8A;
    }
    .error-message {
        color: #ff0000;
        background-color: #ffeeee;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .chat-button {
        text-align: left !important;
        margin-bottom: 0.5rem;
        width: 100%;
    }
    .code-block {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
    }
    .docker-compose-container {
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        background-color: #f9f9f9;
    }
    .docker-compose-header {
        font-weight: bold;
        margin-bottom: 10px;
        color: #1E3A8A;
    }
    .docker-compose-code {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        overflow-x: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><h1 class="logo-text">DocPose AI</h1></div>', unsafe_allow_html=True)

# Ensure database tables exist
try:
    create_tables()
except Exception as e:
    st.error(f"Error creating database tables: {str(e)}")
    st.stop()

def generate_title_from_messages(messages):
    """Generate a title from the first user message"""
    for message in messages:
        if message["role"] == "user":
            # Take first 30 chars of first user message
            return message["content"][:30] + "..." if len(message["content"]) > 30 else message["content"]
    return "New Conversation"

def load_conversation(convo_id, db):
    """Load a conversation by ID"""
    convo = get_conversation(db, convo_id)
    if convo:
        st.session_state.conversation_id = convo.id
        st.session_state.current_conversation_title = convo.title
        st.session_state.messages = [{"role": msg.role, "content": msg.content} for msg in convo.messages]
        st.rerun()

def is_docker_compose_request(prompt):
    """Check if the user prompt is requesting Docker Compose generation"""
    docker_keywords = [
        "docker compose", "docker-compose", "generate docker", "create docker", 
        "docker file", "dockerfile", "containerize", "containerization",
        "docker services", "docker containers"
    ]
    
    # Check for Docker keywords
    if any(keyword.lower() in prompt.lower() for keyword in docker_keywords):
        return True
    
    return False

def generate_docker_compose(prompt):
    """Generate Docker Compose file based on user requirements"""
    try:
        # Get the Docker Compose supervisor
        docker_supervisor = get_docker_supervisor(debug=True)
        
        # Generate Docker Compose file
        result = docker_supervisor.generate_docker_compose(prompt)
        
        # Format the result for display
        if "docker_compose" in result:
            docker_compose = result["docker_compose"]
            if isinstance(docker_compose, dict):
                docker_compose = yaml.dump(docker_compose, sort_keys=False)
            
            # Format the response
            response = f"""
I've generated a Docker Compose configuration based on your requirements:

```yaml
{docker_compose}
```

This Docker Compose file includes the services you requested. You can save this to a file named `docker-compose.yml` and run it with `docker-compose up -d`.
"""
            if "explanation" in result and result["explanation"]:
                response += f"\n\n**Explanation:**\n{result['explanation']}"
            
            # Add information about which method was used (for transparency)
            if "method" in result:
                method_names = {
                    "direct_generation": "Direct Generation",
                    "chain_generation": "LangChain Sequence",
                    "supervisor_direct": "Supervisor Direct",
                    "supervisor_fallback": "Supervisor Fallback",
                    "supervisor_format": "Supervisor Format",
                    "supervisor_emergency": "Emergency Generation",
                    "error_fallback": "Error Recovery"
                }
                method_name = method_names.get(result["method"], result["method"])
                response += f"\n\n<small>_Generation method: {method_name}_</small>"
            
            return response
        elif "error" in result:
            error_method = result.get("method", "unknown")
            return f"""
I encountered an issue while generating the Docker Compose configuration, but I've created a basic template that you can modify:

```yaml
{result.get('docker_compose', 'version: "3.8"\nservices:\n  # Add your services here')}
```

Error details: {result['error']}

<small>_Generation method: {error_method}_</small>

Would you like to provide more details about what you need in your Docker Compose configuration?
"""
        else:
            return "I apologize, but I couldn't generate a Docker Compose file. Please check your requirements and try again."
    
    except Exception as e:
        error_message = f"Error generating Docker Compose file: {str(e)}"
        traceback.print_exc()
        
        # Provide a fallback response with a simple Docker Compose template
        return f"""
I encountered an error while generating your Docker Compose file: {error_message}

Here's a simple template to get you started:

```yaml
version: "3.8"
services:
  app:
    image: your-image
    ports:
      - "8080:8080"
    environment:
      - EXAMPLE_VAR=value
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Please modify this template according to your requirements.
"""

def main():
    # Initialize session state
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
    if "current_model" not in st.session_state:
        st.session_state.current_model = "gemini"  # Default model
    if "error" not in st.session_state:
        st.session_state.error = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_conversation_title" not in st.session_state:
        st.session_state.current_conversation_title = ""
    
    # Get database connection
    try:
        db = get_db()
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.stop()
    
    # Initialize conversation if needed
    if "conversation_id" not in st.session_state:
        try:
            # Get the latest conversation or create a new one
            conversation = get_latest_conversation(db)
            st.session_state.conversation_id = conversation.id
            st.session_state.current_conversation_title = conversation.title
            # Load messages into session state for display
            st.session_state.messages = [{"role": msg.role, "content": msg.content} for msg in conversation.messages]
            
            # Add welcome message if conversation is new/empty
            if not st.session_state.messages:
                welcome_message = "Hello! I'm DocPose AI. How can I help you today?"
                result = add_message(db, st.session_state.conversation_id, "assistant", welcome_message)
                if result:
                    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
                else:
                    st.session_state.error = "Failed to add welcome message. Please check database connection."
        except Exception as e:
            st.session_state.error = f"Error initializing conversation: {str(e)}"
            st.session_state.messages = []
            traceback.print_exc()
    
    # Display any error message
    if st.session_state.error:
        st.markdown(f'<div class="error-message">{st.session_state.error}</div>', unsafe_allow_html=True)
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("Chat")
        
        # New chat button
        if st.button("➕ New Chat"):
            try:
                # Create a new conversation
                new_conversation = create_conversation(db)
                st.session_state.conversation_id = new_conversation.id
                st.session_state.current_conversation_title = new_conversation.title
                st.session_state.messages = []
                
                # Add welcome message
                welcome_message = "Hello! I'm DocPose AI. How can I help you today?"
                add_message(db, new_conversation.id, "assistant", welcome_message)
                st.session_state.messages.append({"role": "assistant", "content": welcome_message})
                st.rerun()
            except Exception as e:
                st.session_state.error = f"Error creating new conversation: {str(e)}"
        
        # Load all conversations
        try:
            st.session_state.conversations = get_all_conversations(db)
        except Exception as e:
            st.session_state.error = f"Error loading conversations: {str(e)}"
            st.session_state.conversations = []
        
        # Conversation list
        if st.session_state.conversations:
            st.subheader("Your conversations")
            
            for idx, convo in enumerate(st.session_state.conversations):
                # Display the conversation with a title or a placeholder
                title = convo.title if convo.title != "New Conversation" else f"Chat {convo.created_at.strftime('%b %d, %Y')}"
                
                # Determine if this is the active conversation
                is_active = convo.id == st.session_state.conversation_id
                
                # Use columns to display the conversation and delete button side by side
                col1, col2 = st.columns([4, 1])
                
                # Display the conversation button
                with col1:
                    # Use a unique key for each button to avoid duplicate keys
                    button_key = f"chat_button_{idx}_{convo.id}"
                    
                    # If this is the active conversation, style it differently
                    if is_active:
                        st.button(f"📝 {title}", key=button_key, disabled=True, use_container_width=True)
                    else:
                        if st.button(f"💬 {title}", key=button_key, use_container_width=True):
                            load_conversation(convo.id, db)
                
                # Display delete button
                with col2:
                    delete_key = f"delete_button_{idx}_{convo.id}"
                    if st.button("🗑️", key=delete_key):
                        if delete_conversation(db, convo.id):
                            # If we deleted the active conversation, load another one
                            if convo.id == st.session_state.conversation_id:
                                st.session_state.conversation_id = None
                                st.session_state.messages = []
                                st.rerun()
                            else:
                                # Just refresh the conversation list
                                st.rerun()
                        else:
                            st.session_state.error = f"Error deleting conversation"
        
        st.divider()
        
        st.header("Settings")
        
        # Database connection status
        try:
            db.execute(text("SELECT 1"))
            st.success("Database connected")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            
        # Model selection
        st.subheader("Model")
        model_type = st.radio(
            "Select AI Provider",
            ["OpenAI", "Google Gemini"],
            index=1 if st.session_state.current_model == "gemini" else 0
        )
        
        st.session_state.current_model = "gemini" if model_type == "Google Gemini" else "openai"
        
        # Model-specific settings
        if st.session_state.current_model == "openai":
            # OpenAI model selection
            openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
            selected_openai_model = st.selectbox("Select OpenAI model", openai_models, index=0)
            st.session_state.openai_model = selected_openai_model
            
            # API key input
            openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
            if openai_api_key:
                st.session_state.openai_api_key = openai_api_key
        else:
            # Gemini model selection
            gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash"]
            selected_gemini_model = st.selectbox("Select Gemini model", gemini_models, index=0)
            st.session_state.gemini_model = selected_gemini_model
        
        # Clear chat button
        if st.button("Clear Conversation"):
            try:
                clear_conversation(db, st.session_state.conversation_id)
                st.session_state.messages = []
                # Add welcome message
                welcome_message = "Hello! I'm DocPose AI. How can I help you today?"
                result = add_message(db, st.session_state.conversation_id, "assistant", welcome_message)
                if result:
                    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
                st.rerun()
            except Exception as e:
                st.session_state.error = f"Error clearing conversation: {str(e)}"

    # Initialize client based on model
    if st.session_state.current_model == "openai":
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
        client = OpenAI(api_key=st.session_state.openai_api_key)
    else:
        if not st.session_state.gemini_api_key:
            st.error("Please enter your Google API key in the sidebar.")
            return
        model = genai.GenerativeModel(st.session_state.gemini_model)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message to database
        try:
            result = add_message(db, st.session_state.conversation_id, "user", prompt)
            if not result:
                st.session_state.error = "Failed to save user message to database."
        except Exception as e:
            st.session_state.error = f"Error saving user message: {str(e)}"
        
        # Generate response based on selected model
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Check if this is a Docker Compose request
                if is_docker_compose_request(prompt):
                    # Generate Docker Compose using the agent
                    full_response = generate_docker_compose(prompt)
                    message_placeholder.markdown(full_response)
                else:
                    if st.session_state.current_model == "openai":
                        # Use OpenAI API
                        stream = client.chat.completions.create(
                            model=st.session_state.openai_model,
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                        
                        # Display the streaming response
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                    else:
                        # Use Gemini API
                        # Format messages for Gemini
                        gemini_messages = []
                        for message in st.session_state.messages:
                            role = "user" if message["role"] == "user" else "model"
                            gemini_messages.append({"role": role, "parts": [message["content"]]})
                        
                        # Create a chat session
                        chat = model.start_chat(history=gemini_messages[:-1])
                        
                        # Get streaming response
                        response = chat.send_message(
                            prompt,
                            stream=True
                        )
                        
                        # Display the streaming response
                        for chunk in response:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Save assistant message to database
                try:
                    result = add_message(db, st.session_state.conversation_id, "assistant", full_response)
                    if not result:
                        st.session_state.error = "Failed to save assistant response to database."
                except Exception as e:
                    st.session_state.error = f"Error saving assistant response: {str(e)}"
                    
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.markdown(f"❌ {error_message}")
                st.session_state.error = error_message

if __name__ == "__main__":
    main() 