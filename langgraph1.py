import streamlit as st
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np

# 앱 타이틀
st.title('논문 기반 Q&A 시스템')

# API 키 입력
api_key = st.text_input('OpenAI API 키를 입력하세요:', type='password')

# PDF 파일 업로드
uploaded_file = st.file_uploader("논문 PDF 파일을 업로드하세요.", type="pdf")

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def get_embeddings(texts, api_key):
    openai.api_key = api_key
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        embeddings = [embedding['embedding'] for embedding in response['data']]
        return embeddings
    except openai.error.InvalidRequestError as e:
        st.error(f"Invalid request error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    return index

if api_key and uploaded_file:
    # PDF 파일에서 텍스트 추출
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_text = extract_text_from_pdf("temp.pdf")

    # 텍스트 길이 제한 처리 (OpenAI API는 매우 긴 텍스트를 처리할 수 없음)
    if len(pdf_text) > 5000:
        pdf_text = pdf_text[:5000]
        st.warning("PDF 텍스트가 너무 길어 일부만 사용됩니다.")

    # 텍스트 임베딩 생성
    text_embeddings = get_embeddings([pdf_text], api_key)

    if text_embeddings:
        # FAISS 인덱스 생성
        index = create_faiss_index(text_embeddings)

        # 질문 입력
        st.header('질문 입력')
        user_question = st.text_area('질문을 입력하세요:')

        if st.button('질문에 답하기'):
            # 질문 임베딩 생성
            question_embedding = get_embeddings([user_question], api_key)

            if question_embedding:
                # 가장 유사한 텍스트 검색
                D, I = index.search(np.array(question_embedding).astype(np.float32), 1)
                closest_text = pdf_text.split('\n')[I[0][0]]

                # 결과 출력
                st.header('답변')
                st.write(closest_text)
