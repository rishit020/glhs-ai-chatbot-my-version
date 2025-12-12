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

// Simple markdown to HTML converter
function markdownToHtml(text) {
    if (!text) return '';
    
    // Split into lines for processing
    const lines = text.split('\n');
    const processedLines = [];
    let inList = false;
    let listItems = [];
    
    function closeList() {
        if (inList && listItems.length > 0) {
            processedLines.push('<ul>' + listItems.join('') + '</ul>');
            listItems = [];
            inList = false;
        }
    }
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        const trimmed = line.trim();
        
        // Headers
        if (trimmed.startsWith('### ')) {
            closeList();
            processedLines.push('<h3>' + trimmed.substring(4) + '</h3>');
            continue;
        }
        if (trimmed.startsWith('## ')) {
            closeList();
            processedLines.push('<h2>' + trimmed.substring(3) + '</h2>');
            continue;
        }
        if (trimmed.startsWith('# ')) {
            closeList();
            processedLines.push('<h2>' + trimmed.substring(2) + '</h2>');
            continue;
        }
        
        // Lists (unordered: *, -, +)
        const unorderedMatch = trimmed.match(/^[\*\-\+] (.+)$/);
        if (unorderedMatch) {
            if (!inList) {
                inList = true;
            }
            listItems.push('<li>' + unorderedMatch[1] + '</li>');
            continue;
        }
        
        // Lists (ordered: 1., 2., etc.)
        const orderedMatch = trimmed.match(/^\d+\. (.+)$/);
        if (orderedMatch) {
            if (!inList) {
                inList = true;
            }
            listItems.push('<li>' + orderedMatch[1] + '</li>');
            continue;
        }
        
        // Empty line - close list if we're in one
        if (trimmed === '') {
            closeList();
            processedLines.push('');
            continue;
        }
        
        // Regular text - close list if we're in one, then add paragraph
        closeList();
        processedLines.push('<p>' + trimmed + '</p>');
    }
    
    // Close any remaining list
    closeList();
    
    let html = processedLines.join('\n');
    
    // Process inline formatting (bold and italic)
    // Bold: **text** (process first to avoid conflicts)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic: *text* (only if not already part of bold)
    // Simple approach: match single asterisks that aren't adjacent to other asterisks
    html = html.replace(/([^*]|^)\*([^*]+?)\*([^*]|$)/g, '$1<em>$2</em>$3');
    
    // Markdown links: [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');
    
    return html;
}

// Add message to chat
function addMessage(text, isUser = false) {
    removeWelcomeMessage();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'counselor'}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    // Render markdown for counselor messages, plain text for user messages
    if (isUser) {
        bubble.textContent = text;
    } else {
        bubble.innerHTML = markdownToHtml(text);
    }
    
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

