import os
import sys

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseChunker import BaseChunker
import streamlit as st

from langchain_core.documents import Document

class SemanticChunker(BaseChunker):
    def __init__(self) -> None:
        st.header(body=f"Fixed Size Chunker")
        self.input_chunk_size = st.number_input("INPUT CHUNK SIZE", value=100)
        self.input_chunk_overlap = 10 # st.number_input("INPUT CHUNK OVERLAP", value=10)
        super().__init__()

    def create_chunks(self) -> list[Document]:
        self.splitter = RecursiveCharacterTextSplitter(
            # chunk_size=self.input_chunk_size,
            # chunk_overlap=self.input_chunk_overlap
        )
        chunks = self.splitter.create_documents(self.input_txt)
       
        return chunks

SemanticChunker()