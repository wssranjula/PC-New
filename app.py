import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import os
import time
import logging
import traceback
import hashlib
import uuid
from pathlib import Path
from functools import lru_cache
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories for temporary and cache files
TEMP_DIR = Path("./temp")
CACHE_DIR = Path("./cache")
TEMP_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Load environment variables
load_dotenv()

# Safe way to get API key - will check env vars if not in secrets
def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("API key not found in secrets or environment variables")
            raise ValueError("API key not found. Please set GOOGLE_API_KEY in your environment or secrets.")
        return api_key

# Initialize LLM with error handling
@st.cache_resource
def init_llm():
    try:
        api_key = get_api_key()
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",
            temperature=0,
            max_tokens=None,
            timeout=30,  # Set explicit timeout
            max_retries=3,
            api_key=api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        return None

# Import prompt
try:
    from prompt import prompt, cp
except ImportError:
    logger.error("Could not import prompt.py")
    prompt = "You are an expert compliance analyst. Analyze the document against company policies and provide a detailed report."
    cp = "You are a helpful AI assistant."

# Load and cache document content
@lru_cache(maxsize=10)
def load_master_data(docx_path):
    """Load and cache policy document content"""
    try:
        if not os.path.exists(docx_path):
            logger.error(f"File not found: {docx_path}")
            return "Policy document not found."
        
        loader = Docx2txtLoader(docx_path)
        data = loader.load()
        return "\n".join([doc.page_content for doc in data])
    except Exception as e:
        logger.error(f"Error loading document {docx_path}: {str(e)}")
        return f"Error loading policy document: {str(e)}"

# PDF processing with chunking for better memory usage
def process_pdf(file_path, chunk_size=5):
    """Process PDF in chunks to avoid memory issues with large files"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Process in chunks to avoid memory issues
        all_content = []
        for i in range(0, len(pages), chunk_size):
            chunk = pages[i:i+chunk_size]
            chunk_content = "\n".join(page.page_content for page in chunk)
            all_content.append(chunk_content)
        
        return "\n".join(all_content)
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

# Cache for document analysis to avoid duplicate processing
def get_cache_key(policy_content, document_content):
    """Generate a cache key based on contents"""
    content_hash = hashlib.md5(f"{policy_content[:1000]}{document_content[:1000]}".encode()).hexdigest()
    return content_hash

def check_cache(cache_key):
    """Check if analysis is cached"""
    cache_file = CACHE_DIR / f"{cache_key}.txt"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return f.read()
    return None

def save_to_cache(cache_key, content):
    """Save analysis to cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.txt"
        with open(cache_file, "w") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Error saving to cache: {str(e)}")

# Define analysis chain with fallback and retry
def run_analysis_chain(company_policy, document_content):
    """Run the analysis chain with error handling and retries"""
    llm = init_llm()
    if not llm:
        return "Error: Could not initialize language model."
    
    # Check cache first
    cache_key = get_cache_key(company_policy, document_content)
    cached_result = check_cache(cache_key)
    if cached_result:
        logger.info("Using cached analysis result")
        return cached_result
    
    analysis_chain = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "Legislation/Regulatory Content: {company_policy}"),
        ("human", "Document Content: {document_content}"),
        ("human", "Provide a Professional Detailed Compliance Report that analyzes how well the document aligns with the legislative requirements. Include specific references to sections of legislation, identify areas of compliance and non-compliance, and suggest improvements where appropriate.")
    ]) | llm | StrOutputParser()
    
    # Try analysis with retry logic
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            report = analysis_chain.invoke({
                "company_policy": company_policy,
                "document_content": document_content
            })
            
            # Cache successful result
            save_to_cache(cache_key, report)
            return report
            
        except Exception as e:
            logger.error(f"Analysis attempt {attempt+1} failed: {str(e)}")
            if attempt == max_retries:
                return f"Error analyzing document. Please try again later. Error details: {str(e)}"
            time.sleep(2)  # Wait before retry

# Define chat chain with error handling and mode-specific prompts
def run_chat_chain(company_policy, report_content, user_question):
    """Run the chat chain with error handling and mode-specific prompts"""
    llm = init_llm()
    if not llm:
        return "Error: Could not initialize language model."
    
    try:
        # Determine if we're in legislation-only mode or document analysis mode
        if not report_content or report_content == "No report generated yet.":
            # Legislation-only mode prompt
            system_prompt = """
            You are an expert assistant specialized in helping users understand legislation and regulatory requirements.
            
            The legislative content is: {company_policy}
            
            Provide concise and accurate answers, referencing specific sections of the legislation when relevant.
            Your answers should be informative, clear, and cite specific parts of the legislation to support your response.
            If you don't know the answer, say so rather than making up information.
            """
        else:
            # Document analysis mode prompt
            system_prompt = """
            You are an expert assistant specialized in helping users understand legislation and compliance reports.
            
            The legislative content is: {company_policy}
            
            The compliance report content is: {report_content}
            
            Provide concise and accurate answers, referencing specific sections of the legislation or report when relevant.
            Your answers should be informative, clear, and cite specific parts of the legislation or report to support your response.
            If discussing compliance issues, be precise about how the document aligns or doesn't align with specific requirements.
            """
        
        chat_chain = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_question}")
        ]) | llm | StrOutputParser()
        
        return chat_chain.invoke({
            "company_policy": company_policy,
            "report_content": report_content,
            "user_question": user_question
        })
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return f"I'm sorry, I encountered an error processing your question. Please try again or rephrase your question. Error: {str(e)[:100]}..."

# Main app with performance optimizations
def main():
    # Streamlit App Configuration
    st.set_page_config(
        page_title="Legislation Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply CSS for better performance (static rather than dynamic)
    st.markdown("""
    <style>
    .main {background: linear-gradient(to bottom, #1a1a1a, #2d2d2d); padding: 20px;}
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px;
        transition: background-color 0.3s; width: 100%; font-size: 14px;
    }
    .stButton>button:hover {background-color: #45a049;}
    
    .stTextInput>div>div>input, div[data-testid="stTextInput"] input, .stTextInput input {
        border-radius: 8px !important; 
        background-color: #2d2d2d !important; 
        color: #e0e0e0 !important; 
        border: 1px solid #4CAF50 !important;
        padding: 20px !important; 
        font-size: 18px !important; 
        width: 100% !important; 
        height: 120px !important;
        min-height: 120px !important;
        line-height: 1.5 !important;
    }
    
    [data-testid="stTextInput"] {
        margin: 15px 0 !important;
    }
    
    .chat-container {
        background-color: #2d2d2d; border: 2px solid #4CAF50; border-radius: 8px;
        padding: 15px; height: 800px; overflow-y: auto; margin-bottom: 20px;
    }
    .chat-message {
        padding: 12px 18px; border-radius: 8px; margin: 10px 0; max-width: 85%;
        font-size: 15px; line-height: 1.5; color: #e0e0e0;
    }
    .user-message {background-color: #4CAF50; color: white;}
    .assistant-message {background-color: #3a3a3a;}
    .stMarkdown h1 {color: #4CAF50; margin-bottom: 20px;}
    .stMarkdown h2 {color: #66BB6A; margin-top: 20px;}
    .footer {font-size: 12px; color: #888; text-align: center; padding: 20px 0;}
    .section-container {background-color: #252525; padding: 20px; border-radius: 8px; height: 100%;}
    .stProgress > div > div > div > div {background-color: #4CAF50 !important;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📑 Legislation Analyzer")
    st.markdown("Ask questions about legislation or upload a document to analyze for compliance.")

    # Define file paths with error handling
    POLICY_FILES = {
        "AML": "./cleaned_document.docx",
        "ASIC": "./asic1cleaned.docx",
        "ARPA": "./ARPA1.docx",
        "Credit": "./credit.docx",
        "Work place health and safety": "./wplhs.docx",
        "Treasury Law": "./tlcleaned.docx"
        # "Privacy": "./privacycleaned.docx",
        # "Fraud Prevention": "./fraud_prevention.docx",
        # "Regulatory Compliance": "./regulatory_compliance.docx"
    }

    # Modified app flow with checkbox for mode selection
    with st.container():
        # First Dropdown: Industry Selection
        industry = st.selectbox(
            "Select Industry",
            ["Financial Services", "Healthcare", "Technology"]
        )

        # Define options for the second dropdown based on industry
        second_dropdown_options = {
            "Financial Services": ["AML", "ARPA", "ASIC", "Work place health and safety", "Privacy", 
                                   "Treasury Law"],
            "Healthcare": ["AML", "ARPA", "ASIC", "Work place health and safety", "Privacy", 
                           "Treasury Law"],
            "Technology": ["AML", "ARPA", "ASIC", "Work place health and safety", "Privacy", 
                           "Treasury Law"]
        }

        # Second Dropdown: Subcategory Selection
        subcategory = st.selectbox(
            "Select Policy Area",
            second_dropdown_options[industry],
            help="Select a specific policy area to focus on."
        )
        
        # Add checkbox for mode selection
        if 'analysis_mode' not in st.session_state:
            st.session_state['analysis_mode'] = "legislation_only"
            
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["Ask questions about legislation only", "Upload document and analyze against legislation"],
            index=0,
            horizontal=True,
            key="mode_selector"
        )
        
        # Update session state based on selection
        st.session_state['analysis_mode'] = "legislation_only" if analysis_mode == "Ask questions about legislation only" else "document_analysis"
    
    # Load selected policy with error handling
    try:
        data_path = POLICY_FILES.get(subcategory, POLICY_FILES["AML"])
        
        # Check if file exists
        if not os.path.exists(data_path):
            st.error(f"Policy file for {subcategory} not found. Using default policy.")
            logger.error(f"Policy file not found: {data_path}")
            data_path = next((p for p in POLICY_FILES.values() if os.path.exists(p)), None)
            
            if not data_path:
                st.error("No policy files found. Please check the application configuration.")
                return
        
        # Load policy content
        company_policy = load_master_data(data_path)
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! You can ask me anything about the company policy, or upload a document to analyze it against the policy."}]
        
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = str(uuid.uuid4())
        
        # Equal column layout
        col1, col2 = st.columns(2, gap="medium")
        
        # Left Column - Document Upload and Analysis
        with col1:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("Document Analysis")
            
            uploaded_file = st.file_uploader(
                "Upload a PDF", 
                type=["pdf"], 
                help="Upload a document to analyze against company policy",
                accept_multiple_files=False,
                key="pdf_uploader"
            )
            
            if uploaded_file is not None:
                # Generate unique filename to avoid conflicts
                temp_file_path = TEMP_DIR / f"{st.session_state.session_id}_{uploaded_file.name}"
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                file_size = os.path.getsize(temp_file_path) / (1024 * 1024)  # Size in MB
                
                # Warn if file is large
                if file_size > 10:
                    st.warning(f"Large file detected ({file_size:.1f} MB). Processing may take longer.")
                
                if st.button("Analyze Document 📊", key="analyze"):
                    progress = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Process PDF with progress updates
                        status_text.text("Reading document...")
                        progress.progress(10)
                        
                        # Use ThreadPoolExecutor for non-blocking file processing
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(process_pdf, temp_file_path)
                            
                            # Simulate progress while waiting
                            for i in range(10, 50):
                                time.sleep(0.05)
                                progress.progress(i)
                            
                            document_content = future.result()
                        
                        progress.progress(50)
                        status_text.text("Analyzing document...")
                        
                        # Run analysis
                        report = run_analysis_chain(company_policy, document_content)
                        
                        # Update session state
                        st.session_state['report'] = report
                        st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! I've generated the compliance report. Feel free to ask me any questions about it or the company policy."}]
                        
                        # Finish progress
                        for i in range(50, 101):
                            time.sleep(0.02)
                            progress.progress(i)
                        
                        status_text.text("Analysis complete!")
                        st.toast("Analysis complete!", icon="✅")
                        
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            
                    except Exception as e:
                        logger.error(f"Error during analysis: {str(e)}\n{traceback.format_exc()}")
                        status_text.text("Error during analysis")
                        st.error(f"Error analyzing document: {str(e)}")
                        progress.progress(100)
                        
                        # Clean up temp file on error
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
            
            # Display report with expandable view
            if 'report' in st.session_state:
                st.markdown("#### Analysis Report")
                
                # Show preview with expand option
                report_preview = st.session_state['report'][:200] + "..." if len(st.session_state['report']) > 200 else st.session_state['report']
                st.markdown(f"{report_preview}", unsafe_allow_html=True)
                
                with st.expander("View Full Report"):
                    st.markdown(st.session_state['report'])
                
                # Download option
                st.download_button(
                    label="Download Report 📤",
                    data=st.session_state['report'],
                    file_name=f"compliance_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    key="download_report"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Right Column - Chat Interface
        with col2:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.subheader("Chat with Policy & Report")
            
            # Render chat messages
            chat_html = ""
            for msg in st.session_state['chat_history']:
                timestamp = time.strftime("%I:%M %p")
                role_class = "user" if msg["role"] == "user" else "assistant"
                chat_html += f'<div style="display: flex; justify-content: {"flex-end" if msg["role"] == "user" else "flex-start"};"><div class="chat-message {role_class}-message" role="log" aria-label="{role_class} message">{msg["content"]} <span style="font-size: 12px; color: #888;">{timestamp}</span></div></div>'
            
            st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)
            
            # Chat input section with clearing functionality
            with st.container():
                # Initialize session states for input management
                if 'clear_input' not in st.session_state:
                    st.session_state['clear_input'] = False
                
                # Process message from previous submission if exists
                if st.session_state.get('clear_input', False):
                    # Get the stored message
                    user_message = st.session_state.get('temp_input', '')
                    # Reset the clear flag
                    st.session_state['clear_input'] = False
                    # Clear the input field
                    st.session_state.chat_input = ""
                    
                    if user_message:
                        # Add user message to chat history
                        st.session_state['chat_history'].append({"role": "user", "content": user_message})
                        
                        with st.spinner("Thinking..."):
                            # Get report content if available
                            report_content = st.session_state.get('report', "No report generated yet.")
                            
                            # Run chat chain
                            response = run_chat_chain(company_policy, report_content, user_message)
                            
                            # Add assistant response
                            st.session_state['chat_history'].append({"role": "assistant", "content": response})
                        
                        # Clear the temp message
                        st.session_state.pop('temp_input', None)
                        st.toast("Message sent!", icon="✉️")
                        st.rerun()
                
                # User input field
                user_input = st.text_area(
                    "Ask a question:", 
                    placeholder="e.g., 'What does the policy say about data security?'",
                    key="chat_input",
                    height=120
                )
                
                # Buttons for send and clear
                col_send, col_clear = st.columns(2)
                with col_send:
                    if st.button("Send 📩", key="send_chat", disabled=not user_input):
                        if user_input:
                            # Store the current input before it gets cleared
                            st.session_state['temp_input'] = user_input
                            # Set flag to clear input on next rerun
                            st.session_state['clear_input'] = True
                            st.rerun()
                
                with col_clear:
                    # Initialize clear flag if not present
                    if 'clear_chat_clicked' not in st.session_state:
                        st.session_state['clear_chat_clicked'] = False
                        
                    # Check if we need to clear the chat (from previous click)
                    if st.session_state.get('clear_chat_clicked', False):
                        # Reset the flag
                        st.session_state['clear_chat_clicked'] = False
                        # Reset the chat history
                        st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! You can ask me anything about the company policy, or upload a document to analyze it against the policy."}]
                        st.toast("Chat cleared!", icon="🧹")
                        st.rerun()
                    
                    # Button that just sets the flag
                    if st.button("Clear 🗑️", key="clear_chat"):
                        st.session_state['clear_chat_clicked'] = True
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown('<div class="footer">Powered by SSR | © 2025</div>', unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")
        st.markdown("Please refresh the page and try again. If the problem persists, contact support.")

if __name__ == "__main__":
    main()