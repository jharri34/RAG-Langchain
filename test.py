#! .venv/Scripts/python.exe
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from  langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer The question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""



vector_store =  None

def get_chroma_datebase():
    return vector_store

def create_chroma_database():
    vector_store  =  Chroma(
    persist_directory = CHROMA_PATH, 
    collection_name="foo",

    embedding_function=get_embedding_function(),
    # other params... 
)
    return vector_store
def search_chroma_database(query, k):
    results = vector_store.similarity_search(query,k)
    return results
def delete_chroma_database(ids):
    vector_store.delete(ids=["3"])
def update_chroma_database(document :Document):
    vector_store.update_documents(ids=[f"{document.metadata["id"]}"],documents=[document])

def add_chroma_database(chunks: list[Document]):
    index = 0
    last_id = None
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_id = f"{source}:{page}:{index}"
        if current_id == last_id:
            last_id = current_id
            index += 1
        else :
            index = 0

        
        print(f"current id: {current_id}")
        chunk.metadata["id"] = current_id
        print(f"adding {current_id}")
        vector_store.add_documents(documents=chunk,ids=current_id)
        print(f"added {current_id}")



def get_embedding_function():
    return OllamaEmbeddings(
    model="llama3"
)


def load_docs():
    docs_loader = DirectoryLoader(DATA_PATH)
    
    return docs_loader.load()

def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

    

def main():

   #st.subheader('Raw data')
   #st.write("Here's our first attempt at using data to create a table:")
   #with st.sidebar:
   #    st.write("This code will be printed to the sidebar.")
           

   docs = []
   docs = load_docs()

   #print(docs)
   chunks = split_docs(docs)
   create_chroma_database()
   add_chroma_database(chunks)
   

   
   


if __name__=="__main__":
    main()