import sys
import os

from langchain_text_splitters import CharacterTextSplitter

from commondata import DUMMY_TEXT
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseChunker import BaseChunker

import streamlit as st

from langchain_core.documents import Document

class FixedSizeChunker(BaseChunker):
    def __init__(self) -> None:
        st.header(body=f"Fixed Size Chunker")
        super().__init__()

    def display_components(self):
        """Add chunker-specific UI components"""
        self.input_chunk_size = st.number_input("INPUT CHUNK SIZE", value=35)
        self.input_chunk_overlap = st.number_input("INPUT CHUNK OVERLAP", value=5)

    def create_chunks(self) -> list[Document]:
        self.splitter = CharacterTextSplitter(
            chunk_size=self.input_chunk_size,
            separator='', # important
            strip_whitespace=True,
            chunk_overlap=self.input_chunk_overlap
        )

        chunks = self.splitter.create_documents([self.input_txt])
        st.info(f'CREATED {len(chunks)} CHUNKS')
        return chunks

# Instantiate the class to trigger the UI
FixedSizeChunker()