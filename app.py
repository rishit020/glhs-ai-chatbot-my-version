"""
Flask application for Green Level AI Counselor Chatbot.
"""
import os
import logging
import socket
from atexit import register
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler  # pyright: ignore[reportMissingImports]
from chatbot import get_chatbot
from utils import get_or_create_session, append_message, clear_stale_sessions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize chatbot on startup
try:
    chatbot = get_chatbot()
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    chatbot = None

# Initialize scheduler for stale session cleanup
# Only initialize once (prevents duplicate scheduler in Flask reloader)
scheduler = None

def initialize_scheduler():
    """Initialize and start the scheduler for stale session cleanup."""
    global scheduler
    if scheduler is None or not scheduler.running:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=clear_stale_sessions,
            trigger="interval",
            minutes=5,
            id='clear_stale_sessions',
            name='Clear stale sessions every 5 minutes'
        )
        scheduler.start()
        logger.info("Scheduler started: Stale session cleanup every 5 minutes")

# Initialize scheduler
initialize_scheduler()

# Register shutdown handler
@register
def shutdown_scheduler():
    """Shutdown scheduler on app termination."""
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages from the frontend."""
    if not chatbot:
        return jsonify({
            "error": "Chatbot not available. Please check server configuration.",
            "session_id": request.json.get("session_id", "")
        }), 500
    
    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")
        
        if not message:
            return jsonify({
                "error": "Message cannot be empty.",
                "session_id": session_id
            }), 400
        
        # Get conversation history (before adding current message)
        conversation = get_or_create_session(session_id)
        
        # Query chatbot with RAG (pass history without current message)
        response = chatbot.query_with_rag(
            question=message,
            conversation_history=conversation
        )
        
        # Add both user message and assistant response to history after getting response
        append_message(session_id, "user", message)
        append_message(session_id, "assistant", response)
        
        return jsonify({
            "response": response,
            "session_id": session_id
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "An error occurred processing your request.",
            "session_id": request.json.get("session_id", "") if request.json else ""
        }), 500


@app.route("/quick-action", methods=["POST"])
def quick_action():
    """Handle quick action button clicks."""
    if not chatbot:
        return jsonify({
            "error": "Chatbot not available.",
            "session_id": request.json.get("session_id", "")
        }), 500
    
    try:
        data = request.json or {}
        action = data.get("action", "").strip()
        session_id = data.get("session_id", "default")
        
        # Map actions to questions
        action_questions = {
            "graduation_requirements": "What are the graduation requirements at Green Level High School?",
            "course_planning": "Help me plan my courses for next year. What should I consider?",
            "college_prep": "What are some college preparation tips? Tell me about AP vs Honors courses.",
            "meet_counselor": "Who are the counselors at Green Level and how can I contact them?"
        }
        
        question = action_questions.get(action.lower(), "")
        if not question:
            return jsonify({
                "error": "Invalid action.",
                "session_id": session_id
            }), 400
        
        # Get conversation history (before adding current message)
        conversation = get_or_create_session(session_id)
        
        # Query chatbot (pass history without current message)
        response = chatbot.query_with_rag(
            question=question,
            conversation_history=conversation
        )
        
        # Add both user message and assistant response to history after getting response
        append_message(session_id, "user", question)
        append_message(session_id, "assistant", response)
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "question": question
        })
    
    except Exception as e:
        logger.error(f"Error in quick_action endpoint: {e}")
        return jsonify({
            "error": "An error occurred processing your request.",
            "session_id": request.json.get("session_id", "") if request.json else ""
        }), 500


def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables!")
        logger.error("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        exit(1)
    
    # Find available port (start with 5000, try next if in use)
    port = find_available_port(5000)
    if port != 5000:
        logger.info(f"Port 5000 is in use. Using port {port} instead.")
    
    logger.info(f"Starting Flask app on http://localhost:{port}")
    # Run the app
    app.run(host="0.0.0.0", port=port, debug=True)

