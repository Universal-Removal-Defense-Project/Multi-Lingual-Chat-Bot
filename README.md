# Multi-Lingual-Chat-Bot
AI-powered multilingual chatbot supporting 8+ languages with persistent conversation history, language-aware responses, and scalable architecture for immigrant-serving nonprofit organizations.

Universal Multi-Lingual Access Chatbot

The Universal Multi-Lingual Access Chatbot is an open-source AI assistant designed to provide accessible communication across languages and cultures. Built using Python, Streamlit, and Groq's ultra-fast LLM infrastructure, the application enables users to communicate with an AI assistant in their preferred language while maintaining conversation history and a simple user-friendly interface.

This project serves as the foundation for future language-access initiatives developed by the Universal Removal Defense Project (URDP), supporting the organization's vision of ensuring that language should never be a barrier to information, services, or access to justice.

Core Features
Multi-language AI conversations
Support for English, Spanish, French, German, Arabic, Chinese, Japanese, Hindi, and additional languages
Persistent multi-chat conversation history
Groq-powered Llama 3.3 70B integration
Modern Streamlit user interface
Lightweight deployment architecture
Free API option through Groq
Mobile-responsive design
Conversation export capability
Scalable foundation for future legal-access tools
Technology Stack
Python 3.x
Streamlit
Groq API
Llama 3.3 70B
Session State Management
JSON Storage
GitHub Actions (future)
Docker (future)
Long-Term Vision

Future versions may include:

Voice-to-text support
Real-time translation
Document translation assistance
Immigration intake assistance
Language accessibility tools
Multi-user authentication
Case management integration
AI-powered knowledge retrieval
Secure cloud deployment

---

## Getting Started

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Groq API key
Get a free key at https://console.groq.com/keys, then either:
```bash
export GROQ_API_KEY="gsk_your_key_here"
```
or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in.

### 3. Run
```bash
streamlit run ui.py
```
The app opens at http://localhost:8501. Pick a language in the sidebar and start chatting.

## Project Structure
```
backend.py            # Groq calls + prompt building (pure logic, no Streamlit)
storage.py            # Conversation data model + JSON persistence
styles.py             # Centralised theme CSS (light/dark + mobile)
ui.py                 # Streamlit app (the only module that imports streamlit)
assets/               # Optional: drop logo.png here to show a brand logo
.streamlit/
    config.toml       # Theme / server config
    secrets.toml      # Your API key (gitignored; see .example)
requirements.txt
```
Design note: AI logic, persistence, and UI are kept in separate modules so later
milestones (history, sidebar, streaming, theming) extend the code instead of
rewriting it.

## Milestone Status
- **M1 — Core Chatbot MVP:** complete (project structure, Groq backend, chat UI, language selector)
- **M2 — Conversation Management:** complete (persistent history, new chat, sidebar, rename/delete)
- **M3 — UI/UX Enhancement:** complete (URDP branding, light/dark toggle, streaming, mobile styling)
