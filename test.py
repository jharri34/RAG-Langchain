#! .venv/Scripts/python.exe
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from  langchain_community.document_loaders import DirectoryLoader


DATA_PATH = "data"

def load_docs():
    docs_loader = DirectoryLoader(DATA_PATH)
    
    return docs_loader.load()

def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {(docs)} docs into {len(chunks)} chunks")
    document =  chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks

    

def main():

   #st.subheader('Raw data')
   #st.write("Here's our first attempt at using data to create a table:")
   #with st.sidebar:
   #    st.write("This code will be printed to the sidebar.")
           

   docs = []
   docs = load_docs()

   print(docs)
   #chunks = split_docs(docs)
   
   

   
   


if __name__=="__main__":
    main()