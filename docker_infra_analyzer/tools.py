import docker
from docker.errors import DockerException, NotFound
from langchain_core.tools import Tool
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_docker_client():
    """Initializes and returns a Docker client.

    Returns:
        docker.DockerClient or None: The Docker client instance or None if connection fails.
    """
    try:
        client = docker.from_env()
        client.ping() # Check connection
        logger.info("Successfully connected to Docker daemon.")
        return client
    except DockerException as e:
        logger.error(f"Failed to connect to Docker daemon: {e}")
        return None


def _format_json_output(data):
    """Formats Python dictionary data as a JSON string for LLM consumption."""
    try:
        # Attempt to serialize. Handle potential non-serializable types gracefully.
        return json.dumps(data, indent=2, default=str)
    except TypeError as e:
        logger.error(f"Error serializing data to JSON: {e}")
        return f"Error: Could not serialize data to JSON. {e}"

def list_running_containers_tool():
    """Tool to list all currently running Docker containers.

    Returns:
        str: A JSON string listing running containers (ID, name, image, status) or an error message.
    """
    client = get_docker_client()
    if not client:
        return "Error: Could not connect to Docker daemon."
    try:
        containers = client.containers.list()
        if not containers:
            return "No running containers found."
        container_info = [
            {
                "id": c.short_id,
                "name": c.name,
                "image": c.image.tags[0] if c.image.tags else 'N/A',
                "status": c.status
            }
            for c in containers
        ]
        return _format_json_output(container_info)
    except DockerException as e:
        logger.error(f"Docker error listing containers: {e}")
        return f"Error listing containers: {e}"
    except Exception as e:
        logger.error(f"Unexpected error listing containers: {e}")
        return f"An unexpected error occurred: {e}"

def inspect_container_tool(container_id_or_name: str):
    """Tool to inspect a specific Docker container by its ID or name.

    Args:
        container_id_or_name (str): The ID (short or long) or name of the container to inspect.

    Returns:
        str: A JSON string containing detailed container information (especially ports) or an error message.
    """
    client = get_docker_client()
    if not client:
        return "Error: Could not connect to Docker daemon."
    if not container_id_or_name:
        return "Error: Please provide a container ID or name."
    try:
        container = client.containers.get(container_id_or_name)
        attrs = container.attrs
        # Extract key information, especially ports
        info = {
            "Id": attrs.get("Id", "N/A"),
            "Name": attrs.get("Name", "N/A"),
            "State": attrs.get("State", {}),
            "Created": attrs.get("Created", "N/A"),
            "Image": attrs.get("Config", {}).get("Image", "N/A"),
            "NetworkSettings": {
                "Ports": attrs.get("NetworkSettings", {}).get("Ports", {})
            },
            "HostConfig": {
                "PortBindings": attrs.get("HostConfig", {}).get("PortBindings", {})
            }
        }
        return _format_json_output(info)
    except NotFound:
        return f"Error: Container '{container_id_or_name}' not found."
    except DockerException as e:
        logger.error(f"Docker error inspecting container {container_id_or_name}: {e}")
        return f"Error inspecting container {container_id_or_name}: {e}"
    except Exception as e:
        logger.error(f"Unexpected error inspecting container {container_id_or_name}: {e}")
        return f"An unexpected error occurred: {e}"

def list_docker_images_tool():
    """Tool to list all locally available Docker images.

    Returns:
        str: A JSON string listing images (ID, tags, size) or an error message.
    """
    client = get_docker_client()
    if not client:
        return "Error: Could not connect to Docker daemon."
    try:
        images = client.images.list()
        if not images:
            return "No local images found."
        image_info = [
            {
                "id": img.short_id,
                "tags": img.tags,
                # Size comes in bytes, convert to MB for readability
                "size_mb": round(img.attrs.get('Size', 0) / (1024 * 1024), 2)
            }
            for img in images
        ]
        return _format_json_output(image_info)
    except DockerException as e:
        logger.error(f"Docker error listing images: {e}")
        return f"Error listing images: {e}"
    except Exception as e:
        logger.error(f"Unexpected error listing images: {e}")
        return f"An unexpected error occurred: {e}"

def get_docker_info_tool():
    """Tool to get general information about the Docker daemon (like version, OS type, number of containers/images).

    Returns:
        str: A JSON string with Docker system information or an error message.
    """
    client = get_docker_client()
    if not client:
        return "Error: Could not connect to Docker daemon."
    try:
        info = client.info()
        # Select relevant info for the agent
        relevant_info = {
            "ServerVersion": info.get("ServerVersion"),
            "OperatingSystem": info.get("OperatingSystem"),
            "OSType": info.get("OSType"),
            "Architecture": info.get("Architecture"),
            "Containers": info.get("Containers"),
            "ContainersRunning": info.get("ContainersRunning"),
            "ContainersPaused": info.get("ContainersPaused"),
            "ContainersStopped": info.get("ContainersStopped"),
            "Images": info.get("Images"),
            "MemoryLimit": info.get("MemoryLimit"),
            "SwapLimit": info.get("SwapLimit")
        }
        return _format_json_output(relevant_info)
    except DockerException as e:
        logger.error(f"Docker error getting info: {e}")
        return f"Error getting Docker info: {e}"
    except Exception as e:
        logger.error(f"Unexpected error getting Docker info: {e}")
        return f"An unexpected error occurred: {e}"

# --- Langchain Tool Definitions ---

docker_tools = [
    Tool(
        name="list_running_containers",
        func=list_running_containers_tool,
        description="Lists all currently running Docker containers with their ID, name, image, and status. Use this to find out what containers are active."
    ),
    Tool(
        name="inspect_container",
        func=inspect_container_tool,
        description=(
            "Provides detailed information about a specific container, identified by its name or ID. "
            "Crucially, this tool shows exposed ports and port bindings (HostConfig.PortBindings and NetworkSettings.Ports). "
            "Input should be the container ID or name. Example: 'my_container' or 'a1b2c3d4e5f6'"
        )
    ),
    Tool(
        name="list_docker_images",
        func=list_docker_images_tool,
        description="Lists all Docker images available locally, including their tags and sizes."
    ),
    Tool(
        name="get_docker_info",
        func=get_docker_info_tool,
        description="Returns general information about the Docker daemon environment, including version, OS, architecture, and counts of containers and images."
    )
] 