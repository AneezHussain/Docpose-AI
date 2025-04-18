import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import DockerComposeFile, DockerService, DockerNetwork, DockerVolume
from models import DockerServiceVolume, DockerServicePort, DockerServiceEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("docker_agent")


class DockerComposeGenerator:
    """Docker Compose Generator using LangChain and Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        """Initialize the Docker Compose Generator.
        
        Args:
            api_key: Google Gemini API key
            debug: Enable debug mode with verbose logging
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.debug = debug
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info("Initializing Docker Compose Generator")
        
        if not self.api_key:
            logger.error("Google API key is missing")
            raise ValueError("Google API key is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=self.api_key,
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=4096
        )
        
        # Define tools
        self.tools = self._create_tools()
        
        # Create the Docker Compose agent
        self.agent = self._create_direct_chain()
        logger.info("Docker Compose Generator initialized successfully")
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the Docker Compose Agent."""
        logger.debug("Creating Docker Compose tools")
        
        return [
            Tool(
                name="generate_docker_compose",
                func=self._generate_docker_compose,
                description="Generate a Docker Compose file based on user requirements",
            ),
            Tool(
                name="explain_docker_compose",
                func=self._explain_docker_compose,
                description="Explain the Docker Compose file and how it meets the user requirements",
            )
        ]
    
    def _create_direct_chain(self) -> RunnableSequence:
        """Create a direct Runnable chain for Docker Compose generation without using an agent."""
        logger.debug("Creating Docker Compose chain")
        
        template = """
        You are a Docker Compose expert. Your job is to generate Docker Compose files based on user requirements.
        
        Guidelines:
        1. Analyze the user requirements carefully
        2. Consider best practices for Docker Compose configurations
        3. Generate a valid Docker Compose YAML that meets all requirements
        4. Provide explanations for your choices
        5. Be security-conscious: avoid exposing unnecessary ports or using root users
        6. Always use version 3.8 or higher for the Docker Compose file
        7. Always validate your YAML structure before returning it
        8. Always use proper indentation and YAML syntax
        9. Use environment variables appropriately
        10. Consider volumes for data persistence
        11. Set appropriate restart policies for services
        
        User requirements:
        {requirements}
        
        Generate a complete Docker Compose file in YAML format.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["requirements"]
        )
        
        # Create a Runnable sequence using the pipe operator
        return prompt | self.llm | StrOutputParser()
    
    def _generate_docker_compose(self, requirements: str) -> str:
        """Generate a Docker Compose file based on user requirements."""
        logger.info(f"Generating Docker Compose from requirements (length: {len(requirements)})")
        logger.debug(f"Requirements: {requirements[:100]}...")
        
        prompt = f"""
        Generate a Docker Compose file based on the following requirements:
        
        {requirements}
        
        Follow these guidelines:
        1. Use Docker Compose version 3.8
        2. Include all necessary services, networks, and volumes
        3. Set appropriate environment variables
        4. Configure proper networking between services
        5. Use best practices for security and reliability
        6. Add comments in the YAML to explain key decisions
        
        Return the Docker Compose file as valid YAML.
        """
        
        # Generate Docker Compose configuration using the LLM
        logger.debug("Sending request to Gemini API")
        response = self.llm.invoke(prompt)
        logger.debug("Received response from Gemini API")
        
        # Extract YAML content from response
        yaml_content = response.content
        if "```yaml" in yaml_content:
            yaml_content = yaml_content.split("```yaml")[1].split("```")[0].strip()
        elif "```" in yaml_content:
            yaml_content = yaml_content.split("```")[1].split("```")[0].strip()
        
        # Parse YAML to validate and structure it
        try:
            logger.debug("Validating YAML content")
            docker_compose_dict = yaml.safe_load(yaml_content)
            result = yaml.dump(docker_compose_dict, sort_keys=False)
            logger.info(f"Successfully generated Docker Compose YAML ({len(result)} chars)")
            return result
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML generated: {str(e)}")
            return f"Invalid YAML: {str(e)}\n\nRaw content:\n{yaml_content}"
    
    def _explain_docker_compose(self, docker_compose: str) -> str:
        """Explain the Docker Compose file and how it meets the user requirements."""
        logger.info("Generating explanation for Docker Compose file")
        
        prompt = f"""
        Explain the following Docker Compose configuration:
        
        ```yaml
        {docker_compose}
        ```
        
        In your explanation:
        1. Describe each service and its purpose
        2. Explain the networking setup
        3. Detail the data persistence strategy
        4. Highlight security considerations
        5. Mention any performance optimizations
        6. Explain how to start and use this Docker Compose file
        """
        
        logger.debug("Sending explanation request to Gemini API")
        response = self.llm.invoke(prompt)
        logger.debug("Received explanation from Gemini API")
        
        explanation = response.content
        logger.info(f"Generated explanation ({len(explanation)} chars)")
        return explanation
    
    def generate(self, requirements: str) -> Dict[str, Any]:
        """Generate a Docker Compose file based on user requirements."""
        logger.info("Starting Docker Compose generation process")
        
        # Try to directly generate without agent if simpler
        try:
            logger.info("Attempting direct generation approach")
            yaml_output = self._generate_docker_compose(requirements)
            explanation = self._explain_docker_compose(yaml_output)
            
            logger.info("Direct generation successful")
            return {
                "docker_compose": yaml_output,
                "explanation": explanation,
                "method": "direct_generation"
            }
        except Exception as e:
            logger.warning(f"Direct generation failed: {str(e)}")
            logger.info("Falling back to chain-based generation")
            
            # Fall back to direct chain
            try:
                # Use invoke() instead of run() with the new RunnableSequence
                logger.debug("Invoking LangChain sequence")
                chain_result = self.agent.invoke({"requirements": requirements})
                
                # Extract YAML from the result
                logger.debug("Processing chain result")
                yaml_content = chain_result
                if "```yaml" in yaml_content:
                    yaml_content = yaml_content.split("```yaml")[1].split("```")[0].strip()
                elif "```" in yaml_content:
                    yaml_content = yaml_content.split("```")[1].split("```")[0].strip()
                
                logger.info("Chain-based generation successful")
                return {
                    "docker_compose": yaml_content,
                    "explanation": "Docker Compose configuration generated based on your requirements.",
                    "method": "chain_generation"
                }
            except Exception as nested_e:
                logger.error(f"Chain-based generation failed: {str(nested_e)}")
                return {
                    "error": f"Error generating Docker Compose: {str(nested_e)}",
                    "docker_compose": "version: '3.8'\nservices:\n  # Error generating Docker Compose",
                    "method": "error_fallback"
                }


# Supervisor Agent for Docker Compose generation
class DockerComposeSupervisor:
    """Supervisor agent that coordinates Docker Compose generation."""
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        """Initialize the Docker Compose Supervisor.
        
        Args:
            api_key: Google Gemini API key
            debug: Enable debug mode with verbose logging
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.debug = debug
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info("Initializing Docker Compose Supervisor")
        
        if not self.api_key:
            logger.error("Google API key is missing")
            raise ValueError("Google API key is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=self.api_key,
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2048
        )
        
        # Create the generator agent
        self.generator = DockerComposeGenerator(api_key=self.api_key, debug=self.debug)
        logger.info("Docker Compose Supervisor initialized successfully")
    
    def analyze_requirements(self, user_input: str) -> Dict[str, Any]:
        """Analyze user requirements and structure them for the generator agent."""
        logger.info("Analyzing user requirements")
        logger.debug(f"User input: {user_input[:100]}...")
        
        prompt = f"""
        Analyze the following user request for Docker Compose generation:
        
        ```
        {user_input}
        ```
        
        Extract and structure the requirements into the following format:
        
        1. Core services needed
        2. Dependencies between services
        3. Environment requirements
        4. Volume/data persistence needs
        5. Networking requirements
        6. Special configurations
        
        Be detailed and specific. Your analysis will be used to generate a Docker Compose file.
        """
        
        logger.debug("Sending analysis request to Gemini API")
        response = self.llm.invoke(prompt)
        logger.debug("Received analysis from Gemini API")
        
        # Format the structured requirements
        structured_requirements = response.content
        logger.info(f"Requirements analysis complete ({len(structured_requirements)} chars)")
        
        return {"structured_requirements": structured_requirements}
    
    def generate_docker_compose(self, user_input: str) -> Dict[str, Any]:
        """Generate Docker Compose file based on user input."""
        logger.info("Starting Docker Compose generation process with supervisor")
        
        try:
            # Step 1: Analyze the requirements
            logger.info("Step 1: Analyzing requirements")
            analysis = self.analyze_requirements(user_input)
            
            # Step 2: Generate the Docker Compose file using the generator agent
            logger.info("Step 2: Generating Docker Compose file")
            result = self.generator.generate(analysis["structured_requirements"])
            
            # Log which method was used
            if "method" in result:
                logger.info(f"Generation method used: {result['method']}")
            
            # Step 3: Format the result for the user
            logger.info("Step 3: Formatting results")
            if isinstance(result, dict):
                # If result is a dictionary with docker_compose key
                if "docker_compose" in result:
                    return {
                        "docker_compose": result["docker_compose"],
                        "requirements_analysis": analysis["structured_requirements"],
                        "explanation": result.get("explanation", ""),
                        "method": result.get("method", "unknown")
                    }
                # Handle error case
                elif "error" in result:
                    logger.warning(f"Generation failed with error: {result['error']}")
                    # Fall back to direct generation
                    logger.info("Falling back to direct generation")
                    direct_result = self._direct_generation(user_input)
                    return {
                        "docker_compose": direct_result,
                        "requirements_analysis": analysis["structured_requirements"],
                        "method": "supervisor_fallback"
                    }
                else:
                    # For other dictionary results
                    return {
                        "docker_compose": yaml.dump(result, sort_keys=False),
                        "requirements_analysis": analysis["structured_requirements"],
                        "method": "supervisor_format"
                    }
            else:
                # If result is a string (direct output)
                return {
                    "docker_compose": result,
                    "requirements_analysis": analysis["structured_requirements"],
                    "method": "supervisor_direct"
                }
        except Exception as e:
            # Handle any errors during generation
            error_message = f"Error during Docker Compose generation: {str(e)}"
            logger.error(error_message)
            # Fall back to direct generation without the agent
            logger.info("Falling back to emergency direct generation")
            direct_result = self._direct_generation(user_input)
            return {
                "docker_compose": direct_result,
                "error": error_message,
                "method": "supervisor_emergency"
            }
    
    def _direct_generation(self, user_input: str) -> str:
        """Fallback method to directly generate Docker Compose without using the agent."""
        logger.info("Performing direct generation as fallback")
        
        prompt = f"""
        Generate a Docker Compose file based on the following user request:
        
        ```
        {user_input}
        ```
        
        Follow these guidelines:
        1. Use Docker Compose version 3.8
        2. Include all necessary services, networks, and volumes
        3. Set appropriate environment variables
        4. Configure proper networking between services
        5. Use best practices for security and reliability
        
        Return ONLY the Docker Compose file as valid YAML inside a code block.
        """
        
        logger.debug("Sending direct generation request to Gemini API")
        response = self.llm.invoke(prompt)
        logger.debug("Received direct generation from Gemini API")
        
        # Extract YAML content from response
        yaml_content = response.content
        if "```yaml" in yaml_content:
            yaml_content = yaml_content.split("```yaml")[1].split("```")[0].strip()
        elif "```" in yaml_content:
            yaml_content = yaml_content.split("```")[1].split("```")[0].strip()
        
        logger.info(f"Direct generation complete ({len(yaml_content)} chars)")
        return yaml_content


# Create a singleton instance of the supervisor
docker_supervisor = None

def get_docker_supervisor(api_key: Optional[str] = None, debug: bool = True) -> DockerComposeSupervisor:
    """Get or create a singleton instance of the Docker Compose Supervisor."""
    global docker_supervisor
    if docker_supervisor is None:
        logger.info("Creating new Docker Compose Supervisor instance")
        docker_supervisor = DockerComposeSupervisor(api_key=api_key, debug=debug)
    return docker_supervisor 