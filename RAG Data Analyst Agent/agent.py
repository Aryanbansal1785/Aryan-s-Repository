import os
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#Auto Summary

def generate_auto_summary(columns: list) -> str:
    engine = create_engine("sqlite:///db/data.db")
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM uploaded_data"))
        row_count = result.fetchone()[0]
    
    prompt = f"""
You are a data analyst. A dataset was just uploaded with {row_count} rows and these columns: {', '.join(columns)}.

Give a brief, friendly 3-4 sentence summary of:
1. What this dataset appears to be about
2. What kinds of questions can be answered from it
3. One interesting analysis you'd suggest

Keep it concise and conversational.
"""
    return llm.invoke(prompt).content


#SQL Agent with Memory
def ask_sql_agent(question: str, columns: list, chat_history: list = []) -> tuple:
    engine = create_engine("sqlite:///db/data.db")

    schema_context = f"The table is called 'uploaded_data' and has these columns: {', '.join(columns)}"
    
    # Build memory context from chat history
    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for msg in chat_history[-6:]:  # last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
You are a data analyst. {schema_context}

{history_text}

The user asked: "{question}"

Write a valid SQLite SQL query to answer this question.
Return ONLY the SQL query, nothing else.
"""
    sql_query = llm.invoke(prompt).content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        rows = result.fetchall()
        col_names = list(result.keys())

    result_text = "\n".join([str(dict(zip(col_names, row))) for row in rows[:20]])

    explanation_prompt = f"""
The user asked: "{question}"
The SQL query was: {sql_query}
The result was: {result_text}

Give a clear, concise answer in plain English. Be conversational and insightful.
"""
    explanation = llm.invoke(explanation_prompt).content
    
    # Return chart data too
    chart_data = {"columns": col_names, "rows": [list(r) for r in rows[:20]]}
    return explanation, sql_query, chart_data


#RAG Agent with Memory
def ask_rag_agent(question: str, chat_history: list = []) -> tuple:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )

    full_question = f"{history_text}\nCurrent question: {question}" if history_text else question
    result = qa_chain.invoke({"query": full_question})
    return result["result"], None, None
