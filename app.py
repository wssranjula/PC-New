import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import os
import time

# Load environment variables
load_dotenv()
api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key
)

from prompt import prompt, cp

# Load master data
def load_master_data(docx_path):
    loader = Docx2txtLoader(docx_path)
    data = loader.load()
    return "\n".join([doc.page_content for doc in data])

# Define analysis chain
analysis_chain = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("human", "Company Policy: {company_policy}"),
    ("human", "Document Content: {document_content}"),
    ("human", "Provide a Professional Detailed Report.")
]) | llm | StrOutputParser()

# Define chat chain
chat_chain = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assistant specialized in helping users understand company policies and compliance reports. The company policy content is: {company_policy}. If a report exists, its content is: {report_content}. Provide concise and accurate answers, referencing specific sections of the policy or report when relevant. If no report exists, assist based on the policy alone."),
    ("human", "{user_question}")
]) | llm | StrOutputParser()

# Streamlit App Configuration
st.set_page_config(page_title="Policy Analyzer", layout="wide")
st.markdown("""
   <style>
    .main {background: linear-gradient(to bottom, #1a1a1a, #2d2d2d); padding: 20px;}
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px;
        transition: background-color 0.3s; width: 100%; font-size: 14px;
    }
    .stButton>button:hover {background-color: #45a049;}
    
    /* More specific selector to ensure these styles are applied */
    .stTextInput>div>div>input, div[data-testid="stTextInput"] input, .stTextInput input {
        border-radius: 8px !important; 
        background-color: #2d2d2d !important; 
        color: #e0e0e0 !important; 
        border: 1px solid #4CAF50 !important;
        padding: 20px !important; 
        font-size: 18px !important; 
        width: 100% !important; 
        height: 70px !important;
        min-height: 70px !important;
        line-height: 1.5 !important;
    }
    
    /* Additional styling for the container */
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
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📑 Company Policy Analyzer")
st.markdown("Upload a PDF to analyze against company policies or chat with the policy document.")

# Industry Dropdown
# Industry Dropdowns
with st.container():
    # First Dropdown: Industry Selection
    industry = st.selectbox(
        "Select Industry",
        ["Financial Services", "Healthcare", "Technology"]
    )

    # Define options for the second dropdown based on industry
    second_dropdown_options = {
        "Financial Services": ["AML",  "ARPA","ASIC", "Work place health and safety","Privacy","Treasury Law","Fraud Prevention", "Credit","Regulatory Compliance"],
        "Healthcare": ["AML",  "ARPA","ASIC", "Work place health and safety","Privacy","Treasury Law","Fraud Prevention", "Credit","Regulatory Compliance"],
        "Technology": ["AML",  "ARPA","ASIC", "Work place health and safety","Privacy","Treasury Law","Fraud Prevention", "Credit","Regulatory Compliance"]
    }

    # Second Dropdown: Subcategory Selection
    if industry == "Financial Services":
        subcategory = st.selectbox(
            "Select Policy Area",
            second_dropdown_options[industry],
            help="Select a specific policy area to focus on."
        )
    else:
        subcategory = st.selectbox(
            "Select Policy Area",
            second_dropdown_options[industry]
        )
    
    # Optional: Display selected subcategory (for debugging or UI purposes)
    if industry == "Financial Services" and subcategory:
        st.write(f"Selected Policy Area: {subcategory}")

# Main content with equal columns
AML_DATA_PATH = "./cleaned_document.docx"
ASIC_DATA_PATH = "./asic1cleaned.docx"
ARPA_DATA_PATH = "./ARPA1.docx"
CREDIT_DATA_PATH = "./credit.docx"
WORK_PLACE_HEALTH_SAFETY_PATH = "./wplhs.docx"
TREASURY_LAW_DTA_PATH = "./tlcleaned.docx"
PRIVACY_DATA_PATH = "./privacycleaned.docx"


if not os.path.exists(AML_DATA_PATH):
    st.error(f"Master data file not found at: {AML_DATA_PATH}")
else:
        # Set the data path based on the selected subcategory
    if subcategory == "ARPA":
        data_path = ARPA_DATA_PATH
    # elif subcategory == "Credit":
    #     data_path = CREDIT_DATA_PATH
    elif subcategory == "ASIC":
        data_path = ASIC_DATA_PATH
    elif subcategory == "AML":
        data_path = AML_DATA_PATH
    elif subcategory == "Work place health and safety":
        data_path = WORK_PLACE_HEALTH_SAFETY_PATH
    elif subcategory == "Treasury Law":
        data_path = TREASURY_LAW_DTA_PATH
    elif subcategory == "Privacy":
        data_path = PRIVACY_DATA_PATH
    else :
         data_path = AML_DATA_PATH
    print(data_path)
    company_policy = load_master_data(data_path)
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! You can ask me anything about the company policy, or upload a document to analyze it against the policy."}]
    
    # Equal column layout
    col1, col2 = st.columns(2, gap="medium")
    
    # Left Column - Document Upload and Analysis
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Document Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF", 
            type=["pdf"], 
            help="Drag and drop supported!",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            progress = st.progress(0)
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load_and_split()
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
            document_content = "\n".join(page.page_content for page in pages)
            
            if st.button("Analyze Document 📊", key="analyze"):
                with st.spinner("Analyzing document..."):
                    try:
                        report = analysis_chain.invoke({
                            "company_policy": company_policy,
                            "document_content": document_content
                        })
                        st.session_state['report'] = report
                        st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! I've generated the compliance report. Feel free to ask me any questions about it or the company policy."}]
                        st.toast("Analysis complete!", icon="✅")
                    except Exception as e:
                        st.toast(f"Error: {e}", icon="❌")
                os.remove("temp.pdf")
        
        if 'report' in st.session_state:
            st.markdown("#### Analysis Report")
            st.markdown(f"{st.session_state['report'][:200]}... [Expand]", unsafe_allow_html=True)
            with st.expander("View Full Report"):
                st.markdown(st.session_state['report'])
            st.download_button(
                label="Download Report 📤",
                data=st.session_state['report'],
                file_name="compliance_report.md",
                mime="text/markdown"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Column - Chat Interface
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Chat with Policy & Report")
        
        chat_html = ""
        for msg in st.session_state['chat_history']:
            timestamp = time.strftime("%I:%M %p")
            chat_html += f'<div style="display: flex; justify-content: {"flex-end" if msg["role"] == "user" else "flex-start"};"><div class="chat-message {msg["role"]}-message" role="log" aria-label="{msg["role"]} message">{msg["content"]} <span style="font-size: 12px; color: #888;">{timestamp}</span></div></div>'
        st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)
        
        # Chat input section
        with st.container():
            user_input = st.text_input(
                "Ask a question:", 
                placeholder="e.g., 'What does the policy say about data security?'",
                key="chat_input"
            )
            col_send, col_clear = st.columns(2)
            with col_send:
                if st.button("Send 📩", key="send_chat", disabled=not user_input):
                    if user_input:
                        st.session_state['chat_history'].append({"role": "user", "content": user_input})
                        with st.spinner("Thinking..."):
                            report_content = st.session_state.get('report', "No report generated yet.")
                            response = chat_chain.invoke({
                                "company_policy": company_policy,
                                "report_content": report_content,
                                "user_question": user_input
                            })
                            st.session_state['chat_history'].append({"role": "assistant", "content": response})
                        st.toast("Message sent!", icon="✉️")
                        st.rerun()
            with col_clear:
                if st.button("Clear 🗑️", key="clear_chat"):
                    st.session_state['chat_history'] = [{"role": "assistant", "content": "Hello! You can ask me anything about the company policy, or upload a document to analyze it against the policy."}]
                    st.toast("Chat cleared!", icon="🧹")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Powered by SSR | © 2025</div>', unsafe_allow_html=True)