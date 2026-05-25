from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

@st.cache_resource
def build_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_texts(_chunks, embeddings)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask Your PDF 🔍")

    pdf = st.file_uploader("Upload Your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
            chunk_size=1000,
            chunk_overlap=400,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        info_base = build_vectorstore(chunks)
        retriever = info_base.as_retriever(search_kwargs={"k": 10})

        llm = Ollama(model="llama3",num_gpu=0)

        prompt = ChatPromptTemplate.from_template("""
        Answer the user's question based only on the provided context.
        If you don't know the answer, just say that you don't know.

        <context>
        {context}
        </context>

        Question: {input}
        """)

        user_question = st.text_input("Ask a question about your PDF")

        if user_question:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": user_question})
            st.write(response["answer"])                                 

if __name__ == "__main__":
    main()