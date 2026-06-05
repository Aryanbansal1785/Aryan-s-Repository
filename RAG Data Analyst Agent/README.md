# 🔍 DataLens – AI Data Analyst Agent

An AI-powered data analyst that lets you upload any CSV, Excel, or PDF file and ask questions about it in plain English. Built with RAG (Retrieval-Augmented Generation) and AI Agents.


## Demo

![DataLens Demo](demo.gif)



## Features

- **Multi-file support** — Upload CSV, Excel (.xlsx), or PDF files
- **Auto-summary** — Instantly summarizes your dataset the moment you upload it
- **Natural language queries** — Ask questions like "which region had the highest profit?" without writing any SQL
- **Auto-generated charts** — Automatically visualizes results as bar charts when relevant
- **Transparent SQL** — Shows the SQL query it generated so you can verify the logic
- **Conversation memory** — Remembers context across multiple questions in the same session
- **PDF Q&A** — Chat with any PDF document using RAG



## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5 Turbo |
| Embeddings | OpenAI Embeddings |
| Vector Store | ChromaDB |
| Agent & RAG | LangChain |
| Database | SQLite via SQLAlchemy |
| Charts | Plotly |
| UI | Streamlit |
| Language | Python 3.9+ |

---

## Architecture

```
User uploads file
       ↓
file_handler.py detects file type
       ↓
CSV/Excel ──→ db_setup.py ──→ SQLite DB ──→ SQL Agent ──→ Answer + Chart
PDF       ──→ rag_setup.py ──→ ChromaDB  ──→ RAG Agent ──→ Answer
```


## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/aryanbansal1785/Aryan-s-Repository.git
cd Aryan-s-Repository
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```


## Project Structure

```
DataLens/
├── src/
│   ├── agent.py          # SQL Agent + RAG Agent + Auto Summary
│   ├── db_setup.py       # CSV/Excel → SQLite loader
│   ├── rag_setup.py      # PDF → ChromaDB indexer
│   └── file_handler.py   # File type detection and routing
├── app.py                # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Example Questions

**For CSV/Excel files:**
- "Give me a summary of this data"
- "Which region had the highest sales?"
- "Who are the top 5 customers by profit?"
- "What product category has the lowest margins?"
- "Show me monthly sales trends"

**For PDF files:**
- "What is this document about?"
- "Summarize the key findings"
- "What does it say about revenue growth?"


## Future Improvements

- [ ] Support for multiple file uploads at once
- [ ] Export answers and charts as PDF report
- [ ] Support for database connections (PostgreSQL, MySQL)
- [ ] Add line and pie chart options
- [ ] Deploy with user authentication


## Author

**Aryan Bansal**  
[LinkedIn](https://linkedin.com/in/aryanbansal1785) • [GitHub](https://github.com/aryanbansal1785)

---

