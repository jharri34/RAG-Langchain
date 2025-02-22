#! .venv/Scripts/python.exe
import argparse
import os
import sys
import streamlit as st
import pandas as pd
from platform import python_version
from langchain.document_loaders import PyPDFDirectoryLoader



DATA_PATH = "data\- CIA - Human Resource Exploitation Training Manual - aka Honduras Manual - a1-g11 (Torture) (1983).pdf"

def load_docs():
    docs_loader = PyPDFLoader(DATA_PATH)
    
    return docs_loader.load()

def main():

   #st.subheader('Raw data')
   #st.write("Here's our first attempt at using data to create a table:")
   #with st.sidebar:
   #    st.write("This code will be printed to the sidebar.")
           

   docs = []
   print(load_docs)
   docs = load_docs
   
   


if __name__=="__main__":
    main()