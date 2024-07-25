import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Streamlit app
st.title("Document QA System")
st.write("Upload a PDF document and ask questions about its content.")

# Input OpenAI API key
openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")

# Upload PDF document
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and openai_api_key:
    with st.spinner('Processing...'):
        # 단계 1: 문서 로드(Load Documents)
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        loader = PyMuPDFLoader("uploaded_document.pdf")
        docs = loader.load()

        # 단계 2: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)

        # 단계 3: 임베딩(Embedding) 생성
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # 단계 4: DB 생성(Create DB) 및 저장
        # 벡터스토어를 생성합니다.
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

        # 단계 5: 검색기(Retriever) 생성
        # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
        retriever = vectorstore.as_retriever()

        # 단계 6: 프롬프트 생성(Create Prompt)
        # 프롬프트를 생성합니다.
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
        )

        # 단계 7: 언어모델(LLM) 생성
        # 모델(LLM) 을 생성합니다.
        llm = OpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

        # 단계 8: 체인(Chain) 생성
        chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            llm=llm,
            chain_type="stuff",
            prompt=prompt
        )

        st.success('Document processed successfully!')

        # 체인 실행(Run Chain)
        # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
        question = st.text_input("Ask a question about the document:")

        if question:
            with st.spinner('Generating answer...'):
                response = chain.run({"query": question})
                st.write("### Answer")
                st.write(response)
else:
    st.warning("Please upload a PDF document and enter your OpenAI API key.")
