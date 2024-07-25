import streamlit as st
import fitz  # PyMuPDF
from langchain import OpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import json

# 앱 타이틀
st.title('논문 PDF 기반 Q&A 시스템')

# API 키 입력
api_key = st.text_input('OpenAI API 키를 입력하세요:', type='password')

if api_key:
    # PDF 파일 업로드
    uploaded_file = st.file_uploader("논문 PDF 파일을 업로드하세요.", type="pdf")

    # OpenAI 설정
    llm = OpenAI(api_key=api_key)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    def extract_text_and_create_embeddings(pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        # 텍스트를 분할하여 임베딩
        texts = []
        for doc in documents:
            split_text = split_document_text(doc.page_content)
            texts.extend(split_text)
        
        vectorstore = FAISS.from_texts(texts, embeddings)
        return vectorstore

    def split_document_text(text, max_tokens=2048):
        # 텍스트를 최대 토큰 길이에 맞게 분할
        tokens = text.split()
        chunks = []
        chunk = []
        chunk_size = 0

        for token in tokens:
            token_size = len(token)
            if chunk_size + token_size > max_tokens:
                chunks.append(' '.join(chunk))
                chunk = [token]
                chunk_size = token_size
            else:
                chunk.append(token)
                chunk_size += token_size

        if chunk:
            chunks.append(' '.join(chunk))
        
        return chunks

    if uploaded_file is not None:
        # PDF 파일에서 텍스트 추출 및 임베딩 생성
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        vectorstore = extract_text_and_create_embeddings("temp.pdf")
        
        # 질문 입력
        st.header('질문 입력')
        user_question = st.text_area('질문을 입력하세요:')
        
        if st.button('질문에 답하기'):
            # 질문을 임베딩하여 리트리버 사용
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            
            # 질문에 대한 답변 얻기
            result = qa_chain.run(user_question)
            
            # 결과 출력
            st.header('답변')
            st.write(result)

            # 결과를 JSON 포맷으로 출력
            st.header('결과 (JSON 포맷)')
            try:
                result_json = json.loads(result)
                st.json(result_json)
            except json.JSONDecodeError:
                st.write("JSON 형식이 올바르지 않습니다.")
                st.write(result)
