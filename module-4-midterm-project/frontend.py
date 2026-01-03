##############
# frontend.py
##############
"""
Streamlit Frontend for GenAI RAG Pipeline Demo
-----------------------------------------------
A Python-based web interface to interact with the FastAPI backend.
"""

import streamlit as st
import httpx
import time
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="GenAI RAG Pipeline Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8001",
    help="URL of the FastAPI backend server"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # User ID input
    user_id_input = st.text_input(
        "User ID (optional)",
        value=st.session_state.user_id or "",
        help="Optional user identifier for tracking"
    )
    if user_id_input:
        st.session_state.user_id = user_id_input
    
    # API Status Check
    st.markdown("---")
    st.subheader("🔌 API Status")
    
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
                api_status = "connected"
            else:
                st.error("❌ API Error")
                api_status = "error"
    except Exception as e:
        st.error(f"❌ Cannot connect to API\n{str(e)}")
        api_status = "disconnected"
    
    # Metrics link
    st.markdown("---")
    st.subheader("📊 Metrics")
    st.markdown(f"[View Prometheus Metrics]({API_BASE_URL}/metrics)")
    
    # Clear history button
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 GenAI RAG Pipeline Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise-grade RAG pipeline with guardrails & observability</div>', unsafe_allow_html=True)

# Check API connection before allowing queries
if api_status == "disconnected":
    st.warning("⚠️ Cannot connect to the API server. Please ensure the FastAPI server is running.")
    st.info("💡 Start the server with: `python api_server.py` or `uvicorn api_server:app --host 0.0.0.0 --port 8000`")
    st.stop()

# query input
st.markdown("### 💬 Ask a query")
query = st.text_area(
    "Enter your query:",
    height=100,
    placeholder="e.g., What is the capital of France?",
    key="query_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    submit_button = st.button("🚀 Submit", type="primary", use_container_width=True)

# Process query
if submit_button and query.strip():
    with st.spinner("🔄 Processing your query..."):
        try:
            start_time = time.time()
            
            # Make API request
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{API_BASE_URL}/ask",
                    json={
                        "query": query.strip(),
                        "user_id": st.session_state.user_id
                    }
                )
                response.raise_for_status()
                result = response.json()
            
            elapsed_time = time.time() - start_time
            
            # Store in conversation history
            st.session_state.conversation_history.append({
                "query": query.strip(),
                "response": result["response"],
                "request_id": result["request_id"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "response_time": f"{elapsed_time:.2f}s"
            })
            
            # Display answer
            st.markdown("---")
            st.markdown("### ✅ Response")
            st.markdown(f'<div class="answer-box">{result["response"]}</div>', unsafe_allow_html=True)
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Request ID", result["request_id"][:8] + "...")
            with col2:
                st.metric("Response Time", f"{elapsed_time:.2f}s")
            with col3:
                if st.session_state.user_id:
                    st.metric("User ID", st.session_state.user_id)
            
            st.success("✅ Response received successfully!")
            
        except httpx.HTTPStatusError as e:
            st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            st.error(f"❌ Connection Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

elif submit_button and not query.strip():
    st.warning("⚠️ Please enter a query before submitting.")

# Conversation History
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    
    # Show history in reverse order (newest first)
    for i, entry in enumerate(reversed(st.session_state.conversation_history)):
        with st.expander(f"💬 {entry['query'][:50]}... | {entry['timestamp']}", expanded=(i == 0)):
            st.markdown("**Query:**")
            st.write(entry['query'])
            st.markdown("**Response:**")
            st.write(entry['response'])
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Request ID: {entry['request_id']}")
            with col2:
                st.caption(f"Response Time: {entry['response_time']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>GenAI RAG Pipeline Demo | Built with Streamlit & FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)