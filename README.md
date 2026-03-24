# 📄 Smart PDF RAG Chatbot (Groq Powered 🚀)

An intelligent **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to upload PDF documents and ask questions in natural language. The app extracts, processes, and retrieves relevant information to generate accurate, context-aware answers.

---

## 🚀 Features

* 📂 Upload and process PDF documents
* 💬 Chat with your documents in real-time
* 🧠 Context-aware answers using RAG architecture
* ⚡ Fast inference powered by **Groq LLM**
* 🔍 Semantic search using **FAISS vector database**
* 🤖 HuggingFace embeddings for document understanding
* 📚 Source document preview for transparency
* 🗂 Chat history support
* 🔐 Secure API key management using `.env`

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (LLaMA models)
* **Embeddings:** HuggingFace (sentence-transformers)
* **Vector Store:** FAISS
* **Framework:** LangChain

---

## 📸 Demo

> Upload a PDF → Ask questions → Get accurate answers instantly!

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/srujank1995/smart-pdf-rag-chatbot
cd smart-pdf-rag-chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
conda create -p venv python=3.10 -y
conda activate ./venv
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

---

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
smart-pdf-rag-chatbot/
│── app.py
│── requirements.txt
│── .env
│── README.md
```

---

## 🧠 How It Works

1. 📄 PDF is uploaded
2. ✂️ Text is extracted & split into chunks
3. 🔎 Embeddings are generated using HuggingFace
4. 📦 Stored in FAISS vector database
5. ❓ User asks a question
6. 🎯 Relevant chunks are retrieved
7. 🤖 Groq LLM generates the final answer

---

## ⚡ Models Used

| Component  | Model                   |
| ---------- | ----------------------- |
| LLM        | llama-3.3-70b-versatile |
| Embeddings | all-MiniLM-L6-v2        |

---

## 🛠️ Future Enhancements

* 📑 Multi-PDF support
* 🌐 Deploy on cloud (Streamlit Cloud / AWS)
* 🧾 PDF highlighting for answers
* 🧠 Chat memory (context retention)
* 🔊 Voice-based interaction
* 📊 Analytics dashboard

---

## ⚠️ Common Issues

### ❌ Model Download Stuck

* Clear HuggingFace cache
* Add `HF_TOKEN` in `.env`

### ❌ Groq Model Error

* Use updated models:

  * `llama-3.3-70b-versatile`
  * `llama-3.1-8b-instant`

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 💡 Author

Developed by **[Srujan Kinjawadekar]**

---

## 🌟 Support

If you like this project, please ⭐ the repository!

