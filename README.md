# 🌾 Farmer's Scheme Assistant

A sophisticated AI-powered chatbot using RAG (Retrieval Augmented Generation) to help Indian farmers access information about government schemes and agricultural programs.

## 🎯 Features

- **Interactive Chat Interface**: User-friendly conversational interface
- **Intelligent Responses**: Powered by Mistral-7B-Instruct LLM
- **PDF Knowledge Base**: Processes and learns from PDF documents
- **Vector Search**: Efficient information retrieval using FAISS
- **Responsive Design**: Modern UI with smooth animations
- **RAG Architecture**: Context-aware responses using document retrieval

## 🔄 RAG Architecture

This project implements Retrieval Augmented Generation (RAG) to provide accurate, context-aware responses about agricultural schemes.

### How RAG Works in Our System

1. **Document Processing**
   - PDF documents containing scheme information are loaded
   - Documents are split into smaller chunks (500 tokens with 50 token overlap)
   - Each chunk is converted into vector embeddings using sentence-transformers

2. **FAISS Vector Store**
   - FAISS (Facebook AI Similarity Search) creates an efficient index of document embeddings
   - Enables fast similarity search across millions of vectors
   - Stores embeddings in `vectorstore/db_faiss` for persistence

3. **Query Processing**
   ```mermaid
   graph LR
   A[User Query] --> B[Query Embedding]
   B --> C[FAISS Search]
   C --> D[Top 3 Relevant Chunks]
   D --> E[Context + Query]
   E --> F[LLM Response]
   ```

4. **Response Generation**
   - User question is embedded using the same model
   - FAISS finds the most relevant document chunks
   - Retrieved context + question is sent to Mistral-7B
   - LLM generates response based only on provided context

### RAG Components

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   PDF Documents  │     │  Vector Storage   │     │    User Query    │
│    Processing    │ --> │      (FAISS)     │ <-- │    Processing    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  ↓                         ↓
                         ┌──────────────────┐     ┌──────────────────┐
                         │  Context Window  │ --> │   LLM Response   │
                         │    Generation    │     │   Generation     │
                         └──────────────────┘     └──────────────────┘
```

### FAISS Implementation

```python
# Embedding Configuration
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store Creation
db = FAISS.from_documents(text_chunks, embedding_model)

# Similarity Search
retriever = db.as_retriever(search_kwargs={'k': 3})
```

### Key Benefits of RAG

- **Accuracy**: Responses are grounded in actual scheme documents
- **Up-to-date**: Easy to update by adding new PDFs
- **Verifiable**: Sources can be traced back to official documents
- **Efficient**: FAISS enables fast retrieval from large document collections
- **Memory-efficient**: Only relevant context is sent to LLM

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Language Model**: Mistral-7B-Instruct v0.3
- **Embeddings**: Hugging Face (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain
- **Python Version**: 3.11+

## 📋 Prerequisites

- Python 3.11 or higher
- Hugging Face API token
- PDF documents containing scheme information

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/agriculture-bot.git
   cd agriculture-bot
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   HF_TOKEN=your_huggingface_token_here
   ```

5. **Prepare knowledge base**
   ```bash
   # Place your PDF files in the data directory
   python create_memory_for_llm.py
   ```

6. **Run the application**
   ```bash
   streamlit run agribot.py
   ```

## 📁 Project Structure

```
agriculture-bot/
├── agribot.py           # Main application file
├── create_memory_for_llm.py  # Vector store creation
├── data/                # PDF documents directory
├── vectorstore/         # FAISS vector store
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## 🔧 Configuration

The application can be configured through the following environment variables:

- `HF_TOKEN`: Hugging Face API token
- `CUSTOM_PROMPT_TEMPLATE`: Custom prompt template for the LLM
- `DB_FAISS_PATH`: Path to FAISS vector store

## 🚀 Usage

1. Start the application
2. Upload relevant PDF documents containing scheme information
3. Ask questions about agricultural schemes and programs
4. Receive accurate, context-aware responses

## 📚 Knowledge Base

The assistant's knowledge comes from PDF documents containing information about:
- Government agricultural schemes
- Farming programs
- Support initiatives
- Agricultural policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://www.langchain.com/) for document processing
- [Hugging Face](https://huggingface.co/) for ML models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

---
Made with ❤️ for Indian Farmers