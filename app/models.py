from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class ChatMessage(BaseModel):
    """Represents a single chat message in the conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True


class Conversation(BaseModel):
    """Represents a conversation with multiple messages."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = "New Conversation"
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True


class DockerServiceVolume(BaseModel):
    """Represents a volume in a Docker service."""
    source: str
    target: str


class DockerServicePort(BaseModel):
    """Represents a port mapping in a Docker service."""
    published: str
    target: str


class DockerServiceEnvironment(BaseModel):
    """Represents an environment variable in a Docker service."""
    name: str
    value: str


class DockerService(BaseModel):
    """Represents a service in a Docker Compose file."""
    name: str
    image: Optional[str] = None
    build: Optional[Dict[str, Any]] = None
    volumes: Optional[List[DockerServiceVolume]] = None
    ports: Optional[List[DockerServicePort]] = None
    environment: Optional[List[DockerServiceEnvironment]] = None
    depends_on: Optional[List[str]] = None
    restart: Optional[str] = None
    command: Optional[str] = None
    networks: Optional[List[str]] = None


class DockerNetwork(BaseModel):
    """Represents a network in a Docker Compose file."""
    name: str
    external: Optional[bool] = None
    driver: Optional[str] = None


class DockerVolume(BaseModel):
    """Represents a volume in a Docker Compose file."""
    name: str
    external: Optional[bool] = None
    driver: Optional[str] = None


class DockerComposeFile(BaseModel):
    """Represents a Docker Compose file."""
    version: str = "3.8"
    services: List[DockerService]
    networks: Optional[List[DockerNetwork]] = None
    volumes: Optional[List[DockerVolume]] = None


class DockerComposeRequest(BaseModel):
    """Represents a request for Docker Compose generation."""
    description: str
    requirements: List[str]
    optional_info: Optional[Dict[str, Any]] = None 