import streamlit as st
import pandas as pd
import plotly.express as px
from file_handler import handle_file
from agent import ask_sql_agent, ask_rag_agent, generate_auto_summary
import tempfile
import os

#Page Config 

st.set_page_config(
    page_title="DataLens – AI Data Analyst",
    page_icon="🔍",
    layout="wide"
)

#Custom CSS
#Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f5f0e8; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #ede8dc; border-right: 1px solid #d4cfc4; }
    /* Chat messages */
    [data-testid="stChatMessage"] { background-color: #ede8dc; border-radius: 12px; margin-bottom: 8px; }
    /* Input box */
    [data-testid="stChatInput"] { background-color: #ede8dc; border: 1px solid #d4cfc4; border-radius: 12px; }
    /* Metric cards */
    [data-testid="stMetric"] { background-color: #ede8dc; border: 1px solid #d4cfc4; border-radius: 12px; padding: 16px; }
    /* Success/info boxes */
    .stAlert { border-radius: 12px; }
    /* Headings */
    h1, h2, h3 { color: #1a1a1a; }
    /* Expander */
    .streamlit-expanderHeader { background-color: #ede8dc; border-radius: 8px; }
    /* File uploader */
    [data-testid="stFileUploader"] { background-color: #ede8dc; border: 1px dashed #d4cfc4; border-radius: 12px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

#Session State

for key, default in {
    "file_type": None,
    "columns": None,
    "chat_history": [],
    "auto_summary": None,
    "chart_data": None,
    "rows": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

#Sidebar 

with st.sidebar:
    st.markdown("##  DataLens")
    st.caption("AI-powered Data Analyst")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your file",
        type=["csv", "xlsx", "xls", "pdf"],
        help="Supports CSV, Excel, and PDF files"
    )

    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Processing..."):
            result = handle_file(tmp_path)
            st.session_state.file_type = result[0]
            st.session_state.columns = result[1] if result[0] == "sql" else None
            st.session_state.rows = result[2] if result[0] == "sql" else 0
            st.session_state.chat_history = []
            st.session_state.chart_data = None

            if st.session_state.file_type == "sql":
                with st.spinner("Generating summary..."):
                    st.session_state.auto_summary = generate_auto_summary(
                        st.session_state.columns
                    )
            else:
                st.session_state.auto_summary = "PDF indexed and ready! Ask me anything about it."

        st.success("File ready!")

    if st.session_state.file_type == "sql" and st.session_state.columns:
        st.divider()
        st.markdown("**Columns**")
        for col in st.session_state.columns:
            st.caption(f"• {col}")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.chart_data = None
        st.rerun()

# Main Area 

if not st.session_state.file_type:
    # Landing screen
    st.markdown("# DataLens")
    st.markdown("### Your AI-powered Data Analyst")
    st.markdown(" ")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(" **CSV & Excel**\nAsk questions in plain English — DataLens writes the SQL and explains the result")
    with col2:
        st.info(" **PDF Reports**\nUpload any PDF and chat with it — extract insights instantly")
    with col3:
        st.info(" **Auto Charts**\nDataLens automatically visualizes your data when relevant")

    st.markdown(" ")
    st.markdown("#####  Upload a file from the sidebar to get started")

else:
    # Metrics row
    if st.session_state.file_type == "sql":
        st.markdown("#  DataLens")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(" File Type", "Tabular Data")
        with col2:
            st.metric(" Rows", f"{st.session_state.rows:,}")
        with col3:
            st.metric(" Columns", len(st.session_state.columns))
        st.divider()
    else:
        st.markdown("# DataLens")
        st.metric(" File Type", "PDF Document")
        st.divider()

    #Auto summary 
    if st.session_state.auto_summary and not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.write(st.session_state.auto_summary)

    #Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sql"):
                with st.expander("🔍 SQL Query"):
                    st.code(msg["sql"], language="sql")
            if msg.get("chart"):
                chart = msg["chart"]
                if chart and len(chart["columns"]) >= 2:
                    try:
                        df_chart = pd.DataFrame(chart["rows"], columns=chart["columns"])
                        num_cols = df_chart.select_dtypes(include="number").columns.tolist()
                        cat_cols = df_chart.select_dtypes(exclude="number").columns.tolist()
                        if num_cols and cat_cols:
                            fig = px.bar(
                                df_chart,
                                x=cat_cols[0],
                                y=num_cols[0],
                                color_discrete_sequence=["#6366f1"],
                                template="plotly_dark"
                            )
                            fig.update_layout(
                                plot_bgcolor="#1a1d27",
                                paper_bgcolor="#1a1d27",
                                margin=dict(t=20, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass

    # Chat input
    question = st.chat_input("Ask anything about your data...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.file_type == "sql":
                    answer, sql, chart_data = ask_sql_agent(
                        question,
                        st.session_state.columns,
                        st.session_state.chat_history
                    )
                else:
                    answer, sql, chart_data = ask_rag_agent(
                        question,
                        st.session_state.chat_history
                    )

            st.write(answer)

            if sql:
                with st.expander("🔍 SQL Query"):
                    st.code(sql, language="sql")

            # Auto chart
            if chart_data and len(chart_data["columns"]) >= 2:
                try:
                    df_chart = pd.DataFrame(chart_data["rows"], columns=chart_data["columns"])
                    num_cols = df_chart.select_dtypes(include="number").columns.tolist()
                    cat_cols = df_chart.select_dtypes(exclude="number").columns.tolist()
                    if num_cols and cat_cols:
                        fig = px.bar(
                            df_chart,
                            x=cat_cols[0],
                            y=num_cols[0],
                            color_discrete_sequence=["#6366f1"],
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            plot_bgcolor="#1a1d27",
                            paper_bgcolor="#1a1d27",
                            margin=dict(t=20, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "chart": chart_data
        })
