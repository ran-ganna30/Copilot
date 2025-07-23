import os
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.tools.python.tool import PythonREPLTool
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
llm = OpenAI(temperature=0)

# Load CSV Tool
def query_csv_tool():
    df = pd.read_csv("data/sales_data.csv")

    def run(query):
        return str(df.query(query).to_string())

    return Tool(name="Sales CSV Tool", func=run, description="Query sales data")

# Load PDF QA Tool
def build_pdf_qa_tool():
    loader = PyPDFLoader("documents/policy.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    def run(query):
        docs = db.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain.run(input_documents=docs, question=query)

    return Tool(name="PDF QA Tool", func=run, description="Answer questions from company policy PDF")

# Initialize agent
def create_agent():
    tools = [query_csv_tool(), build_pdf_qa_tool(), PythonREPLTool()]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent
