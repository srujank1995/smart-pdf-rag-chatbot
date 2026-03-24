import pdfplumber
import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# ---------------- LOAD ENV ----------------
load_dotenv()

# Load keys
GROQ_API_KEY = SecretStr(os.getenv("GROQ_API_KEY") or "")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set HuggingFace token
if HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Safety check
if not GROQ_API_KEY.get_secret_value():
    st.error("❌ GROQ API key not found. Check your .env file.")
    st.stop()

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart PDF Chatbot", layout="wide")
st.title("📄 Smart PDF Chatbot (Groq Powered 🚀)")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []

# ---------------- FUNCTIONS ----------------

def extract_text_from_pdf(file):
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text.append(content)
    return "\n".join(text)


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # stable for all systems
    )
    return FAISS.from_texts(chunks, embeddings)


def create_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an intelligent assistant answering questions from a document.

Instructions:
- Answer ONLY using the provided context
- If unsure, say "Not available in document"
- Keep answers structured and clear
- Use bullet points where useful

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ---------------- MAIN LOGIC ----------------

if uploaded_file:
    file_hash = hash(uploaded_file.name)

    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("⚙️ Processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error("❌ No readable text found in PDF")
                st.stop()

            chunks = split_text(text)

            st.session_state.vector_store = create_vector_store(chunks)
            st.session_state.chain, st.session_state.retriever = create_chain(
                st.session_state.vector_store
            )
            st.session_state.file_hash = file_hash
            st.session_state.chat_history = []

        st.success("✅ Document processed successfully!")

    # ---------------- CHAT ----------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("💬 Ask something about your document...")

    if user_question:
        with st.spinner("🤖 Thinking..."):
            try:
                    docs = st.session_state.retriever.invoke(user_question)
                    response = st.session_state.chain.invoke(user_question)
            except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
                    st.stop()

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", response))

        # Show sources
        with st.expander("📚 Sources"):
            for i, doc in enumerate(docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:300] + "...")

    # ---------------- DISPLAY CHAT ----------------

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)