import streamlit as st
from agent import create_agent

st.set_page_config(page_title="Enterprise Data Copilot", layout="wide")

st.title("🤖 Enterprise Data Copilot")

query = st.text_input("Ask a question about company data or policy")

if query:
    with st.spinner("Thinking..."):
        agent = create_agent()
        response = agent.run(query)
        st.success(response)
