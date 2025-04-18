from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import logging

from database import ConversationDB, MessageDB
from models import ChatMessage, Conversation


def create_conversation(db: Session) -> Conversation:
    """Create a new conversation in the database."""
    conversation_id = str(uuid.uuid4())
    db_conversation = ConversationDB(id=conversation_id)
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    
    return Conversation(
        id=db_conversation.id,
        title=db_conversation.title,
        messages=[],
        created_at=db_conversation.created_at,
        updated_at=db_conversation.updated_at
    )


def get_conversation(db: Session, conversation_id: str) -> Optional[Conversation]:
    """Get a conversation by ID."""
    db_conversation = db.query(ConversationDB).filter(ConversationDB.id == conversation_id).first()
    
    if not db_conversation:
        return None
    
    return Conversation(
        id=db_conversation.id,
        title=db_conversation.title,
        messages=[
            ChatMessage(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at
            ) for message in db_conversation.messages
        ],
        created_at=db_conversation.created_at,
        updated_at=db_conversation.updated_at
    )


def get_all_conversations(db: Session) -> List[Conversation]:
    """Get all conversations ordered by most recent first."""
    db_conversations = db.query(ConversationDB).order_by(ConversationDB.updated_at.desc()).all()
    
    return [
        Conversation(
            id=convo.id,
            title=convo.title,
            messages=[],  # Don't load messages for conversation list
            created_at=convo.created_at,
            updated_at=convo.updated_at
        ) for convo in db_conversations
    ]


def get_latest_conversation(db: Session) -> Optional[Conversation]:
    """Get the most recent conversation, or create one if none exists."""
    db_conversation = db.query(ConversationDB).order_by(ConversationDB.created_at.desc()).first()
    
    if not db_conversation:
        return create_conversation(db)
    
    return Conversation(
        id=db_conversation.id,
        title=db_conversation.title,
        messages=[
            ChatMessage(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at
            ) for message in db_conversation.messages
        ],
        created_at=db_conversation.created_at,
        updated_at=db_conversation.updated_at
    )


def add_message(db: Session, conversation_id: str, role: str, content: str) -> ChatMessage:
    """Add a message to a conversation."""
    try:
        # First verify the conversation exists
        db_conversation = db.query(ConversationDB).filter(ConversationDB.id == conversation_id).first()
        if not db_conversation:
            print(f"Error: Conversation with ID {conversation_id} not found")
            return None
            
        message_id = str(uuid.uuid4())
        db_message = MessageDB(
            id=message_id,
            role=role,
            content=content,
            conversation_id=conversation_id
        )
        
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        return ChatMessage(
            id=db_message.id,
            role=db_message.role,
            content=db_message.content,
            created_at=db_message.created_at
        )
    except Exception as e:
        print(f"Error adding message: {str(e)}")
        db.rollback()
        return None


def clear_conversation(db: Session, conversation_id: str) -> None:
    """Clear all messages from a conversation."""
    try:
        db.query(MessageDB).filter(MessageDB.conversation_id == conversation_id).delete()
        db.commit()
    except Exception as e:
        print(f"Error clearing conversation: {str(e)}")
        db.rollback()


def update_conversation_title(db: Session, conversation_id: str, title: str) -> bool:
    """Update the title of a conversation."""
    try:
        db_conversation = db.query(ConversationDB).filter(ConversationDB.id == conversation_id).first()
        if not db_conversation:
            print(f"Error: Conversation with ID {conversation_id} not found")
            return False
            
        db_conversation.title = title
        db.commit()
        return True
    except Exception as e:
        print(f"Error updating conversation title: {str(e)}")
        db.rollback()
        return False


def delete_conversation(db: Session, conversation_id: str) -> bool:
    """Delete a conversation and all its messages."""
    try:
        # First delete all messages in the conversation
        db.query(MessageDB).filter(MessageDB.conversation_id == conversation_id).delete()
        
        # Then delete the conversation itself
        result = db.query(ConversationDB).filter(ConversationDB.id == conversation_id).delete()
        
        db.commit()
        return result > 0
    except Exception as e:
        print(f"Error deleting conversation: {str(e)}")
        db.rollback()
        return False 