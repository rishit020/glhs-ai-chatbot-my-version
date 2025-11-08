// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Initialize session
let sessionId = localStorage.getItem('glhs_session_id') || generateSessionId();
localStorage.setItem('glhs_session_id', sessionId);

const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');
const quickButtons = document.querySelectorAll('.quick-btn');

// Remove welcome message on first interaction
let welcomeRemoved = false;

function removeWelcomeMessage() {
    if (!welcomeRemoved) {
        const welcome = document.querySelector('.welcome-message');
        if (welcome) {
            welcome.remove();
            welcomeRemoved = true;
        }
    }
}

// Add message to chat
function addMessage(text, isUser = false) {
    removeWelcomeMessage();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'counselor'}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    
    messageDiv.appendChild(bubble);
    chatContainer.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Show typing indicator
function showTyping() {
    typingIndicator.style.display = 'flex';
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Hide typing indicator
function hideTyping() {
    typingIndicator.style.display = 'none';
}

// Send message to backend
async function sendMessage(message) {
    if (!message.trim()) return;

    // Add user message to UI
    addMessage(message, true);
    messageInput.value = '';
    
    // Show typing indicator
    showTyping();
    sendButton.disabled = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });

        const data = await response.json();

        if (response.ok) {
            hideTyping();
            addMessage(data.response, false);
        } else {
            hideTyping();
            addMessage('Sorry, I encountered an error. Please try again.', false);
            console.error('Error:', data.error);
        }
    } catch (error) {
        hideTyping();
        addMessage('Sorry, I cannot connect to the server. Please check your connection.', false);
        console.error('Network error:', error);
    } finally {
        sendButton.disabled = false;
        messageInput.focus();
    }
}

// Handle quick action buttons
async function handleQuickAction(action) {
    removeWelcomeMessage();
    
    showTyping();
    sendButton.disabled = true;

    try {
        const response = await fetch('/quick-action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                action: action,
                session_id: sessionId
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Add the question to chat as user message
            if (data.question) {
                addMessage(data.question, true);
            }
            hideTyping();
            addMessage(data.response, false);
        } else {
            hideTyping();
            addMessage('Sorry, I encountered an error. Please try again.', false);
            console.error('Error:', data.error);
        }
    } catch (error) {
        hideTyping();
        addMessage('Sorry, I cannot connect to the server. Please check your connection.', false);
        console.error('Network error:', error);
    } finally {
        sendButton.disabled = false;
    }
}

// Event listeners
sendButton.addEventListener('click', () => {
    const message = messageInput.value.trim();
    if (message) {
        sendMessage(message);
    }
});

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (message) {
            sendMessage(message);
        }
    }
});

quickButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const action = btn.getAttribute('data-action');
        handleQuickAction(action);
    });
});

// Focus input on load
messageInput.focus();

