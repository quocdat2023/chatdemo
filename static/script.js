// Application state
let conversations = [];
let currentConversationId = null;
let messages = [];
let isLoading = false;

// Translation dictionary
const translations = {
  en: {
    newConversation: 'New Conversation',
    noConversations: 'No conversations yet',
    startNewChat: 'Start a new chat to begin',
    message: 'message',
    messages: 'messages',
    analyzing: 'Analyzing...',
    legalReferences: 'Legal References',
    pageNumber: 'Page',
    matchScore: 'Match Score',
    relatedQuestions: 'Related Questions',
    errorMessage: 'I apologize, but I encountered an error while processing your request. Please try again.',
  },
  vi: {
    newConversation: 'Hội thoại mới',
    noConversations: 'Chưa có hội thoại nào',
    startNewChat: 'Bắt đầu một cuộc trò chuyện mới để bắt đầu',
    message: 'tin nhắn',
    messages: 'tin nhắn',
    analyzing: 'Đang phân tích...',
    legalReferences: 'Tài liệu pháp lý',
    pageNumber: 'Trang',
    matchScore: 'Độ khớp',
    relatedQuestions: 'Câu hỏi liên quan',
    errorMessage: 'Tôi xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.',
  },
};

// Mock suggested questions for initial screen
const suggestedQuestions = {
  en: [
    'What are the essential elements of a valid contract?',
    'How does civil law differ from criminal law?',
    'What protections does intellectual property law provide?',
    'What are the ways a contract can be terminated?',
  ],
  vi: [
    'Các yếu tố thiết yếu của một hợp đồng hợp lệ là gì?',
    'Luật dân sự khác với luật hình sự như thế nào?',
    'Luật sở hữu trí tuệ cung cấp những bảo vệ gì?',
    'Hợp đồng có thể được chấm dứt bằng những cách nào?',
  ],
};

// DOM elements
const sidebar = document.getElementById('sidebar');
const mobileOverlay = document.getElementById('mobileOverlay');
const menuBtn = document.getElementById('menuBtn');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const newChatBtn = document.getElementById('newChatBtn');
const conversationsList = document.getElementById('conversationsList');
const messagesContainer = document.getElementById('messagesContainer');
const messages_div = document.getElementById('messages');
const welcomeSection = document.getElementById('welcomeSection');
const suggestedQuestionsElement = document.getElementById('suggestedQuestions');
const inputForm = document.getElementById('inputForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const themeToggle = document.getElementById('themeToggle');
const languageSelect = document.getElementById('languageSelect');

// Theme management
function initializeTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
  const icon = themeToggle.querySelector('i');
  icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// Sidebar management
function openSidebar() {
  sidebar.classList.add('open');
  mobileOverlay.classList.add('show');
  document.body.style.overflow = 'hidden';
}

function closeSidebar() {
  sidebar.classList.remove('open');
  mobileOverlay.classList.remove('show');
  document.body.style.overflow = '';
}

// Conversation management
function createNewConversation() {
  const newConversation = {
    id: Date.now().toString(),
    title: translate('newConversation'),
    timestamp: new Date().toISOString(),
    messageCount: 0,
  };
  
  conversations.unshift(newConversation);
  currentConversationId = newConversation.id;
  messages = [];
  
  renderConversations();
  renderMessages();
  closeSidebar();
}

function selectConversation(conversationId) {
  currentConversationId = conversationId;
  messages = [];
  renderMessages();
  closeSidebar();
}

function deleteConversation(conversationId) {
  conversations = conversations.filter(conv => conv.id !== conversationId);
  if (currentConversationId === conversationId) {
    currentConversationId = null;
    messages = [];
    renderMessages();
  }
  renderConversations();
}

function updateConversationTitle(conversationId, newTitle, messageCount) {
  const conversation = conversations.find(conv => conv.id === conversationId);
  if (conversation) {
    if (conversation.messageCount === 0) {
      conversation.title = newTitle.slice(0, 50) + (newTitle.length > 50 ? '...' : '');
    }
    conversation.messageCount = messageCount;
    conversation.timestamp = new Date().toISOString();
    renderConversations();
  }
}

function renderConversations() {
  if (conversations.length === 0) {
    conversationsList.innerHTML = `
      <div class="empty-conversations">
        <i class="fas fa-comments"></i>
        <p data-translate="noConversations">${translate('noConversations')}</p>
        <small data-translate="startNewChat">${translate('startNewChat')}</small>
      </div>
    `;
    return;
  }

  conversationsList.innerHTML = conversations.map(conversation => `
    <div class="conversation-item ${conversation.id === currentConversationId ? 'active' : ''}" 
         data-conversation-id="${conversation.id}">
      <div class="conversation-title">${conversation.title}</div>
      <div class="conversation-meta">
        <span>${conversation.messageCount} ${conversation.messageCount === 1 ? translate('message') : translate('messages')}</span>
        <span>•</span>
        <span>${formatTimestamp(conversation.timestamp)}</span>
      </div>
      <button class="delete-conversation" data-conversation-id="${conversation.id}">
        <i class="fas fa-trash"></i>
      </button>
    </div>
  `).join('');

  conversationsList.querySelectorAll('.conversation-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (!e.target.closest('.delete-conversation')) {
        selectConversation(item.dataset.conversationId);
      }
    });
  });

  conversationsList.querySelectorAll('.delete-conversation').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(btn.dataset.conversationId);
    });
  });
}

// Message management
function addMessage(type, content, sources = null, relatedQuestions = null, isTyping = false) {
  const message = {
    id: Date.now().toString() + (isTyping ? '-typing' : ''),
    type,
    content,
    timestamp: new Date().toISOString(),
    sources,
    relatedQuestions,
    isTyping,
  };
  
  if (isTyping) {
    messages = messages.filter(msg => !msg.isTyping);
  }
  
  messages.push(message);
  renderMessages();
  
  return message.id;
}

function removeMessage(messageId) {
  messages = messages.filter(msg => msg.id !== messageId);
  renderMessages();
}

function renderMessages() {
  const hasMessages = messages.length > 0;
  const hasConversation = currentConversationId !== null;
  
  welcomeSection.style.display = hasMessages ? 'none' : 'block';
  
  if (!hasMessages && !hasConversation) {
    suggestedQuestionsElement.style.display = 'block';
    const currentLang = getCurrentLanguage();
    suggestedQuestionsElement.innerHTML = `
      <h3>${translate('relatedQuestions')}</h3>
      <div class="questions-grid">
        ${suggestedQuestions[currentLang].map(question => `
          <button class="question-btn" data-question="${question}">
            <i class="fas fa-question-circle"></i>
            <span>${question}</span>
          </button>
        `).join('')}
      </div>
    `;
  } else {
    suggestedQuestionsElement.style.display = 'none';
  }

  if (!hasMessages) {
    messages_div.innerHTML = '';
    return;
  }

  messages_div.innerHTML = messages.map(message => {
    if (message.isTyping) {
      return `
        <div class="message assistant fade-in">
          <div class="message-content">
            <div class="message-wrapper">
              <div class="message-avatar">
                <i class="fas fa-robot"></i>
              </div>
              <div class="message-bubble">
                <div class="typing-indicator">
                  <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                  </div>
                  <span>${translate('analyzing')}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      `;
    }

    const sourcesHtml = message.sources ? `
      <div class="message-sources">
        <div class="sources-title">${translate('legalReferences')}</div>
        ${message.sources.map(source => `
          <div class="source-item">
            <div class="source-header">
              <div class="source-info">
                <i class="fas fa-file-text"></i>
                <div class="source-details">
                  <div class="source-name">${source.documentName}</div>
                  <div class="source-excerpt">"${source.excerpt}"</div>
                  ${source.pageNumber ? `<div class="source-page">${translate('pageNumber')} ${source.pageNumber}</div>` : ''}
                </div>
              </div>
              <div class="source-meta">
                <div class="source-score">${Math.round(source.relevanceScore * 100)}% ${translate('matchScore')}</div>
                <i class="fas fa-external-link-alt"></i>
              </div>
            </div>
          </div>
        `).join('')}
      </div>
    ` : '';

    const relatedQuestionsHtml = message.relatedQuestions ? `
      <div class="related-questions">
        <div class="sources-title">${translate('relatedQuestions')}</div>
        <div class="questions-grid">
          ${message.relatedQuestions.map(question => `
            <button class="question-btn" data-question="${question}">
              <i class="fas fa-question-circle"></i>
              <span>${question}</span>
            </button>
          `).join('')}
        </div>
      </div>
    ` : '';

    return `
      <div class="message ${message.type} ${message.type === 'user' ? 'slide-in-right' : 'slide-in-left'}">
        <div class="message-content">
          <div class="message-wrapper">
            <div class="message-avatar">
              <i class="fas fa-${message.type === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-bubble">
              <div class="message-text">${message.content}</div>
              ${sourcesHtml}
              ${relatedQuestionsHtml}
              <div class="message-timestamp">
                <i class="fas fa-clock"></i>
                <span>${formatMessageTime(message.timestamp)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }).join('');

  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Message sending
async function sendMessage(content) {
  if (!content.trim() || isLoading) return;

  if (!currentConversationId) {
    createNewConversation();
  }

  addMessage('user', content);
  
  const messageCount = messages.filter(msg => !msg.isTyping).length;
  updateConversationTitle(currentConversationId, content, messageCount);

  isLoading = true;
  updateSendButton();

  const typingId = addMessage('assistant', '', null, null, true);

  try {
    const { response, sources, related_questions } = await generateResponse(content);
    
    removeMessage(typingId);
    
    addMessage('assistant', response, sources, related_questions);
    
  } catch (error) {
    removeMessage(typingId);
    addMessage('assistant', translate('errorMessage'));
  } finally {
    isLoading = false;
    updateSendButton();
  }
}

async function generateResponse(query) {
  try {
    const response = await fetch('http://127.0.0.1:5000/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: query }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    const formattedResponse = {
      response: data.final_response,
      sources: data.top_banan_documents.map(doc => ({
        documentId: doc.file,
        documentName: doc.file,
        excerpt: doc.text.slice(0, 150) + '...',
        relevanceScore: doc.distance,
        pageNumber: doc.pageNumber || null,
      })),
      related_questions: data.related_questions.map(q => q.question),
    };

    return formattedResponse;
  } catch (error) {
    console.error('Error fetching response from API:', error);
    throw error;
  }
}

// Utility functions
function formatTimestamp(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);
  
  if (diffInHours < 24) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } else if (diffInHours < 168) {
    return date.toLocaleDateString([], { weekday: 'short' });
  } else {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
}

function formatMessageTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
}

function updateSendButton() {
  const hasText = messageInput.value.trim().length > 0;
  sendBtn.disabled = !hasText || isLoading;
}

function adjustTextareaHeight() {
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

function getCurrentLanguage() {
  return languageSelect.value || 'vi';
}

function setLanguage(lang) {
  languageSelect.value = lang;
}

function translate(key) {
  const currentLang = getCurrentLanguage();
  return translations[currentLang][key] || translations.en[key] || key;
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
  initializeTheme();
  renderConversations();
  renderMessages();
  updateSendButton();
});

menuBtn.addEventListener('click', openSidebar);
closeSidebarBtn.addEventListener('click', closeSidebar);
mobileOverlay.addEventListener('click', closeSidebar);
newChatBtn.addEventListener('click', createNewConversation);

themeToggle.addEventListener('click', toggleTheme);

languageSelect.addEventListener('change', (e) => {
  setLanguage(e.target.value);
  renderConversations();
  renderMessages();
});

messageInput.addEventListener('input', () => {
  updateSendButton();
  adjustTextareaHeight();
});

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    inputForm.dispatchEvent(new Event('submit'));
  }
});

inputForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const content = messageInput.value.trim();
  if (content) {
    sendMessage(content);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    updateSendButton();
  }
});

document.addEventListener('click', (e) => {
  if (e.target.closest('.question-btn')) {
    const question = e.target.closest('.question-btn').dataset.question;
    sendMessage(question);
  }
});

window.addEventListener('resize', () => {
  if (window.innerWidth >= 1024) {
    closeSidebar();
  }
});