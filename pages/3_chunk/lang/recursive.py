import os
import sys

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from commondata import CHUNK_OVERLAP, CHUNK_SIZE
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseChunker import BaseChunker
import streamlit as st

from langchain_core.documents import Document

class RecursiveChunker(BaseChunker):
    def __init__(self) -> None:
        st.header(body=f"Recursive Size Chunker")
        self.input_chunk_size = st.number_input("INPUT CHUNK SIZE", value=CHUNK_SIZE)
        self.input_chunk_overlap = st.number_input("INPUT CHUNK OVERLAP", value=CHUNK_OVERLAP)
        # self.separators = st.text_area("Enter strings, separated by commas:", "'\n\n', '\n', ' ")

        super().__init__()

    def create_chunks(self) -> list[Document]:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.input_chunk_size,
            chunk_overlap=self.input_chunk_overlap,
            # separators=[s.strip() for s in self.separators.split(',') if s.strip()]
        )
        chunks = self.splitter.create_documents([self.input_txt])


        st.info(f'CREATED {len(chunks)} CHUNKS')
        
        return chunks

RecursiveChunker()