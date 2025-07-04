// Translation system
const translations = {
  en: {
    // Sidebar
    chatHistory: "Chat History",
    newConversation: "New Conversation",
    noConversations: "No conversations yet",
    startNewChat: "Start a new chat to begin",
    appName: "Legal AI Assistant",
    poweredBy: "Powered by RAG Technology",
    
    // Header
    appTitle: "Legal AI Assistant",
    appSubtitle: "Ask questions about legal topics and get expert insights",
    
    // Welcome
    welcomeTitle: "Welcome to Legal AI Assistant",
    welcomeText: "I'm here to help you understand legal concepts, analyze contracts, and provide insights on various areas of law. Ask me anything about legal topics and I'll provide detailed, professional guidance.",
    popularQuestions: "Popular Questions",
    
    // Questions
    question1: "What are the key elements of a valid contract?",
    question2: "Explain the difference between civil and criminal law",
    question3: "What is intellectual property law?",
    question4: "How does contract termination work?",
    question5: "What are fiduciary duties in corporate law?",
    question6: "Explain force majeure clauses",
    
    // Input
    inputPlaceholder: "Ask a question about legal topics...",
    inputHint: "Press Enter to send, Shift+Enter for new line",
    
    // Messages
    legalReferences: "Legal References",
    analyzing: "Analyzing your question...",
    pageNumber: "Page",
    matchScore: "match",
    
    // Time
    today: "Today",
    yesterday: "Yesterday",
    thisWeek: "This week",
    
    // Conversation meta
    messages: "messages",
    message: "message"
  },
  
  vi: {
    // Sidebar
    chatHistory: "Lịch sử trò chuyện",
    newConversation: "Cuộc trò chuyện mới",
    noConversations: "Chưa có cuộc trò chuyện nào",
    startNewChat: "Bắt đầu trò chuyện mới",
    appName: "Trợ lý AI Pháp lý",
    poweredBy: "Được hỗ trợ bởi công nghệ RAG",
    
    // Header
    appTitle: "Trợ lý AI Pháp lý",
    appSubtitle: "Đặt câu hỏi về các chủ đề pháp lý và nhận được thông tin chuyên sâu",
    
    // Welcome
    welcomeTitle: "Chào mừng đến với Trợ lý AI Pháp lý",
    welcomeText: "Tôi ở đây để giúp bạn hiểu các khái niệm pháp lý, phân tích hợp đồng và cung cấp thông tin chi tiết về các lĩnh vực khác nhau của pháp luật. Hãy hỏi tôi bất cứ điều gì về các chủ đề pháp lý và tôi sẽ cung cấp hướng dẫn chi tiết, chuyên nghiệp.",
    popularQuestions: "Câu hỏi phổ biến",
    
    // Questions
    question1: "Các yếu tố chính của một hợp đồng hợp lệ là gì?",
    question2: "Giải thích sự khác biệt giữa luật dân sự và luật hình sự",
    question3: "Luật sở hữu trí tuệ là gì?",
    question4: "Việc chấm dứt hợp đồng hoạt động như thế nào?",
    question5: "Nghĩa vụ tín thác trong luật doanh nghiệp là gì?",
    question6: "Giải thích các điều khoản bất khả kháng",
    
    // Input
    inputPlaceholder: "Đặt câu hỏi về các chủ đề pháp lý...",
    inputHint: "Nhấn Enter để gửi, Shift+Enter để xuống dòng",
    
    // Messages
    legalReferences: "Tài liệu tham khảo pháp lý",
    analyzing: "Đang phân tích câu hỏi của bạn...",
    pageNumber: "Trang",
    matchScore: "khớp",
    
    // Time
    today: "Hôm nay",
    yesterday: "Hôm qua",
    thisWeek: "Tuần này",
    
    // Conversation meta
    messages: "tin nhắn",
    message: "tin nhắn"
  }
};

// Translation utility functions
function getCurrentLanguage() {
  return localStorage.getItem('language') || 'vi';
}

function setLanguage(lang) {
  localStorage.setItem('language', lang);
  updateTranslations();
}

function translate(key) {
  const lang = getCurrentLanguage();
  return translations[lang][key] || translations.en[key] || key;
}

function updateTranslations() {
  const elements = document.querySelectorAll('[data-translate]');
  elements.forEach(element => {
    const key = element.getAttribute('data-translate');
    element.textContent = translate(key);
  });
  
  // Update placeholders
  const placeholderElements = document.querySelectorAll('[data-translate-placeholder]');
  placeholderElements.forEach(element => {
    const key = element.getAttribute('data-translate-placeholder');
    element.placeholder = translate(key);
  });
}

// Initialize translations when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  const savedLanguage = getCurrentLanguage();
  const languageSelect = document.getElementById('languageSelect');
  if (languageSelect) {
    languageSelect.value = savedLanguage;
  }
  updateTranslations();
});