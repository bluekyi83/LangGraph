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

def split_text(text, max_tokens=2048):
    words = text.split()
    chunks, chunk, length = [], [], 0
    for word in words:
        length += len(word) + 1
        if length > max_tokens:
            chunks.append(' '.join(chunk))
            chunk, length = [word], len(word) + 1
        else:
            chunk.append(word)
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def get_embeddings(texts, api_key):
    openai.api_key = api_key
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
        return [embedding['embedding'] for embedding in response['data']]
    except openai.error.InvalidRequestError as e:
        st.error(f"Invalid request error: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype(np.float32))
    return index

def summarize_text(text, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Please summarize the following text in 5 sentences."},
                {"role": "user", "content": text}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message['content'].strip()
    except openai.error.InvalidRequestError as e:
        st.error(f"Invalid request error: {e}")
        return "요약 실패: 요청 오류 발생"
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "요약 실패: 일반 오류 발생"

if api_key and uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_text = extract_text_from_pdf("temp.pdf")
    text_chunks = split_text(pdf_text)
    text_embeddings = get_embeddings(text_chunks, api_key)

    if text_embeddings:
        index = create_faiss_index(text_embeddings)

        st.header('질문 입력')
        user_question = st.text_area('질문을 입력하세요:')

        if st.button('질문에 답하기'):
            question_embedding = get_embeddings([user_question], api_key)
            if question_embedding:
                D, I = index.search(np.array(question_embedding).astype(np.float32), 1)
                closest_chunk = text_chunks[I[0][0]]
                summary = summarize_text(closest_chunk, api_key)

                st.header('답변')
                st.write(summary)
