import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Streamlit app
st.title("Document QA System")
st.write("Upload a PDF document and ask questions about its content.")

# Input OpenAI API key
openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")

# Upload PDF document
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and openai_api_key:
    with st.spinner('Processing...'):
        # Load documents
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        loader = PyMuPDFLoader("uploaded_document.pdf")
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create and save vector store
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever()

        # Create prompt
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

# Question: 
{question} 
# Context: 
{context} 

# Answer:"""
        )

        # Create LLM
        llm = OpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

        # Create chain
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        st.success('Document processed successfully!')

        # Run chain
        question = st.text_input("Ask a question about the document:")

        if question:
            with st.spinner('Generating answer...'):
                response = qa_chain({"query": question})
                st.write("### Answer")
                st.write(response['result'])
else:
    st.warning("Please upload a PDF document and enter your OpenAI API key.")
