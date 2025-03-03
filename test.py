#! .venv/Scripts/python.exe
import streamlit as st
import asyncio
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from  langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


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
    return create_chroma_database()


def create_chroma_database():
    vector_store  =  Chroma(
    persist_directory = CHROMA_PATH, 
    collection_name="foo",

    embedding_function=get_embedding_function(),
    # other params... 
)
    return vector_store
def search_chroma_database(query, k):
    vector_store = get_chroma_datebase()
    print(f"searching: {query} and getting top: {k} results")
    results = vector_store.similarity_search(query=query,k=k)
    return results
def delete_chroma_database(ids):
    vector_store.delete(ids=["3"])
def update_chroma_database(document :Document):
    vector_store.update_documents(ids=[f"{document.metadata["id"]}"],documents=[document])
def assign_ids(chunks: list[Document]):
    index = 0
    last_id = None
    for chunk in chunks:
        source = chunk.metadata.get("source")
        # print(f"page number is:{chunk}")
        current_id = f"{source}"
        if current_id == last_id:
            index += 1
        else :
            index = 0
        last_id = current_id
        current_id = f"{current_id}:{index}"
        chunk.metadata["id"] = current_id
    return chunks
def add_chroma_database(chunks: list[Document]):
    chunks_with_ids = assign_ids(chunks)
    # ("Assigned ids.print....(((())))((((()))))")
    vector_store = get_chroma_datebase()
    existing_chunks = vector_store.get(include=[])
    existing_ids = set(existing_chunks["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks)>0:
        print (f"👉 Adding new chunks: {len(new_chunks)}")
        vector_store.add_documents(new_chunks,ids=[chunk.metadata["id"] for chunk in new_chunks])
        print(f"added {len(new_chunks)} new chunks to the database ")       
    else:
        print("✅ No new documents to add")
    




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

def query_chroma_database(query:str):
    results = search_chroma_database(query,k=5)
    print(f"these are the top {5} results from your query {query}:\n\n---\n\n {results}" )
    if len(results) == 0:
        print(f"Unable to find results matching query:{query}")
        return
    context= "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)
    print(f"this is the prompt {prompt}")
    model =  OllamaLLM(model="llama3")
    response = model.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(f"this is your response{formatted_response}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The query text.")
    args = parser.parse_args()
    query = args.query
    st.subheader('RAG + Chroma+ LangChain')
    st.write("Here's our first attempt at using data to create a table:")
    #with st.sidebar:
    #    st.write("This code will be printed to the sidebar.")
           
    if query != "":
        docs = []
        docs = load_docs()

        #print(docs)
        chunks = split_docs(docs)
        create_chroma_database()
        add_chroma_database(chunks)
        query_chroma_database(query)
    else:
        print(f"query is {query}")
        return
    
   

   
   


if __name__=="__main__":
    main()