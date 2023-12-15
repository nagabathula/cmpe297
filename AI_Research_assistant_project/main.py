import streamlit as st
import testing 
import json 
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os
#load api key lib
from dotenv import load_dotenv
import base64
import joblib

# Placeholder functions for your logic
def search_documents(query):
    # Implement your search logic here
    return testing.search(query)
    ##["Document 1", "Document 2", "Document 3"]

def summarize_document(pdf):
    
    if pdf is not None:
        st.write(pdf.name)

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()
        
        print(text)

        # Langchain text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Store pdf name
        store_name = pdf.name[:-4]

        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         vectorstore = pickle.load(f)
        # else:
        #     # Embedding (OpenAI methods)
        #     embeddings = OpenAIEmbeddings()

        #     # Store the chunks part in db (vector)
        #     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(vectorstore, f)
        
                   # Embedding (OpenAI methods)
        embeddings = OpenAIEmbeddings()

            # Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        # if os.path.exists(f"{store_name}.joblib"):
        #     # Load vectorstore using joblib
        #     vectorstore = joblib.load(f"{store_name}.joblib")
        # else:
        #     # Embedding (OpenAI methods)
        #     embeddings = OpenAIEmbeddings()

        #     # Store the chunks part in db (vector)
        #     vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        #     # Save vectorstore using joblib
        #     joblib.dump(vectorstore, f"{store_name}.joblib")

        # Accept user questions/query
        query = st.text_input("Ask questions about related your upload pdf file")

        if query:
            docs = vectorstore.similarity_search(query=query, k=5)
            print("After docs")

            # OpenAI rank LNV process
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            print("After chain")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write(response)


# Main app layout
def main():
    st.title("AI-Powered Research Assistant")

    tab1, tab2 = st.tabs(["Document Search", "PDF Upload and Chatbot"])

    with tab1:
        st.header("Search for Documents")
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            results = search_documents(query)
            st.write("Search Results:")
            data = json.loads(results)

            # Display paper information as HTML blocks
            for paper in data['papers']:
                paper_link = f"https://arxiv.org/abs/{paper['id']}"
                
                st.markdown(f"<h4><a href='{paper_link}' target='_blank'>{paper['title']}</a></h4>", unsafe_allow_html=True)
                st.write(f"Authors: {paper['authors']}")
                st.write(f"Abstract: {paper['abstract']}")
                st. write(f"Year: {int(paper['year'])}")
                st.write(f"Search Relevance Score: {paper['score']}")
            


    with tab2:
        st.header("Upload PDF and Summarize")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
        if uploaded_file is not None:
           summarize_document(uploaded_file)
           
if __name__ == "__main__":
    main()

