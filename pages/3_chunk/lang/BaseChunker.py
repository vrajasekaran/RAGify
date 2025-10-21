#https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
import streamlit as st
import pandas as pd
from abc import ABC, abstractmethod
import streamlit as st

from commondata import DUMMY_TEXT

from langchain_core.documents import Document

class BaseChunker:
    @abstractmethod
    def create_chunks(self) -> list[Document]:
        pass

    def __init__(self) -> None:
        self.input_txt = st.text_area(label="INPUT TEXT", value=DUMMY_TEXT)
        st.info(f"TOTAL CHARS: {len(self.input_txt)}")
        
        # Call the abstract method for chunker-specific components
        self.display_components()
        
        # Create chunks and display them
        chunks = self.create_chunks()
        self.display_chunks(chunks)
    
    @abstractmethod
    def display_components(self):
        """Override this method to add chunker-specific UI components"""
        pass
    
    def display_chunks(self, chunks: list[Document]) -> None:
        """Common method to display chunks information and data"""
        st.info(f"TOTAL CHUNKS: {len(chunks)}")
        
        # Convert chunks to a more readable format for display
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "Chunk #": i + 1,
                "Content": chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content,
                "Length": len(chunk.page_content),
                "Metadata": str(chunk.metadata) if chunk.metadata else "None"
            })
        
        st.dataframe(chunk_data)
        st.balloons()

        