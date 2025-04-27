import os
import streamlit as st
import torch

# Disable eager execution for PyTorch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

def local_css():
    st.markdown("""
        <style>
        /* Modern Container Styles */
        .main {
            padding: 30px;
            max-width: 1200px;
            margin: 0 auto;
            background: linear-gradient(to bottom right, #ffffff, #f8f9fa);
        }
        
        /* Logo Styles */
        .logo-container {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            margin: 0 auto;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* Header Styles */
        .stTitle {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 48px !important;
            color: #2e7d32 !important;
            padding-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            letter-spacing: 1px;
            background: -webkit-linear-gradient(45deg, #2e7d32, #1b5e20);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Welcome Message */
        .big-font {
            font-family: 'Segoe UI', sans-serif;
            font-size: 28px !important;
            color: #1e88e5 !important;
            margin-bottom: 30px;
            line-height: 1.4;
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(145deg, #ffffff, #f0f7ff);
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border-left: 5px solid #2e7d32;
        }
        
        /* Scheme List */
        .scheme-list {
            font-family: 'Segoe UI', sans-serif;
            font-size: 22px !important;
            color: #333333 !important;
            margin: 25px 0;
            line-height: 1.8;
            padding: 25px;
            border-radius: 15px;
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        
        .scheme-list ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .scheme-list li {
            margin: 12px 0;
            padding-left: 30px;
            position: relative;
        }
        
        .scheme-list li:before {
            content: '🌱';
            position: absolute;
            left: 0;
        }
        
        /* Chat Messages */
        .chat-message {
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 20px;
            font-size: 18px;
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.6;
            animation: fadeIn 0.5s ease-in;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .chat-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            color: #1565c0;
            margin-left: 20px;
            box-shadow: 0 4px 15px rgba(25, 118, 210, 0.1);
            border-left: 4px solid #1565c0;
        }
        
        .bot-message {
            background: linear-gradient(145deg, #e8f5e9, #c8e6c9);
            color: #2e7d32;
            margin-right: 20px;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.1);
            border-right: 4px solid #2e7d32;
        }
        
        /* Input Field */
        .stTextInput>div>div>input {
            font-size: 20px !important;
            padding: 20px !important;
            border-radius: 15px !important;
            border: 2px solid #e0e0e0 !important;
            background: white !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #2e7d32 !important;
            box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.2) !important;
            transform: translateY(-2px);
        }
        
        /* Footer */
        .footer {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px !important;
            color: #666666 !important;
            text-align: center;
            padding: 30px;
            margin-top: 40px;
            border-top: 2px solid #f0f0f0;
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            position: relative;
        }
        
        .footer::after {
            content: '🌾';
            position: absolute;
            right: 20px;
            bottom: 20px;
            font-size: 24px;
            animation: wave 2s infinite;
        }
        
        @keyframes wave {
            0%, 100% { transform: rotate(0deg); }
            50% { transform: rotate(10deg); }
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm(huggingface_repo_id, HF_TOKEN):
    client = InferenceClient(
        model=huggingface_repo_id,
        token=HF_TOKEN
    )
    
    llm = HuggingFaceEndpoint(
        client=client,
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_length=512
    )
    return llm

def main():
    st.set_page_config(
        page_title="Farmer's Scheme Assistant",
        page_icon="🌾",
        layout="centered"
    )
    
    # Apply custom CSS
    local_css()
    
    # Add logo and title
    st.markdown("""
        <div class="logo-container">
            <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48cGF0aCBmaWxsPSIjMmU3ZDMyIiBkPSJNNjQsMTkyVjQ0OEg0NDhWMTkySDY0Wk0zODQsMzg0SDEyOFYyNTZIMzg0VjM4NFoiLz48cGF0aCBmaWxsPSIjMmU3ZDMyIiBkPSJNMjI0LDIyNEgxNjBWMjg4SDIyNFYyMjRaTTM1MiwyMjRIMjg4VjI4OEgzNTJWMjI0WiIvPjxwYXRoIGZpbGw9IiMyZTdkMzIiIGQ9Ik0yNTYsNjRMMTkyLDEyOEgzMjBMMjU2LDY0WiIvPjwvc3ZnPg==" class="logo" alt="Crop Logo">
        </div>
    """, unsafe_allow_html=True)
    
    st.title("🌾 Farmer's Scheme Assistant")
    
    # Welcome message with enhanced styling
    st.markdown(
        '<div class="big-font">Welcome to your Agricultural Schemes Assistant! '
        'I can help you with information about various government schemes and programs for farmers. '
        'Feel free to ask any questions about agricultural support initiatives.</div>',
        unsafe_allow_html=True
    )
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.markdown(
            f'<div class="chat-message {"user-message" if message["role"] == "user" else "bot-message"}">'
            f'{message["content"]}</div>',
            unsafe_allow_html=True
        )

    # Chat input
    prompt = st.chat_input("Ask me about agricultural schemes...")

    if prompt:
        # Add user message to chat
        st.markdown(
            f'<div class="chat-message user-message">{prompt}</div>',
            unsafe_allow_html=True
        )
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the farmer's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide only information from the given context.

        Context: {context}
        Question: {question}

        Answer the question directly and professionally, focusing on helping farmers.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            # Get vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the knowledge base")
                return

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': PromptTemplate(
                        template=CUSTOM_PROMPT_TEMPLATE,
                        input_variables=["context", "question"]
                    )
                }
            )

            # Get response
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            # Display assistant's response
            st.markdown(
                f'<div class="chat-message bot-message">{result}</div>',
                unsafe_allow_html=True
            )
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Enhanced footer
    st.markdown(
        '<div class="footer">Powered by AI to help Indian farmers access information '
        'about government schemes and agricultural programs. Together we grow! 🌱</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
