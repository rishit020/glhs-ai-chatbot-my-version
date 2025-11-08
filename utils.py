"""
Utility functions for session management and JSON loading.
"""
from datetime import datetime
from typing import Dict, List, Optional
import json
import os

##############################################################################
# Session Memory
##############################################################################
SESSION_MEMORY: Dict[str, Dict] = {}


def get_or_create_session(session_id: str) -> List[Dict[str, str]]:
    """Get existing session or create new one."""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {
            "messages": [],
            "last_updated": datetime.utcnow(),
        }
    else:
        SESSION_MEMORY[session_id]["last_updated"] = datetime.utcnow()
    return SESSION_MEMORY[session_id]["messages"]


def append_message(session_id: str, role: str, content: str):
    """Append a message to the session conversation."""
    conversation = get_or_create_session(session_id)
    conversation.append({"role": role, "content": content})


def clear_stale_sessions(timeout_minutes: int = 5):
    """Clear sessions that haven't been updated in the last N minutes."""
    now = datetime.utcnow()
    to_remove = []
    for sid, data in SESSION_MEMORY.items():
        last_updated = data["last_updated"]
        if (now - last_updated).total_seconds() > timeout_minutes * 60:
            to_remove.append(sid)
    for sid in to_remove:
        del SESSION_MEMORY[sid]


def load_json_data(file_path: str) -> dict:
    """Load JSON data from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' not found.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

