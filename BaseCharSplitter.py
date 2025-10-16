import streamlit as st
from langchain_text_splitters import TextSplitter

from commondata import DUMMY_TEXT

class BaseCharSplitter:

    def __init__(self, splitter: TextSplitter) -> None:
        st.header(body=f"{splitter.__class__.__name__}")
        input_txt = st.text_area(label="INPUT TEXT", value=DUMMY_TEXT)
        st.info(f"TOTAL CHARS: {len(input_txt)}")
        input_chunk_size = st.number_input("INPUT CHUNK SIZE", value=100)
        input_chunk_overlap = st.number_input("INPUT CHUNK OVERLAP", value=10)

        self.splitter = splitter 
        self.splitter._chunk_size = input_chunk_size
        self.splitter._chunk_overlap = input_chunk_overlap

        chunks = self.splitter.split_text(input_txt)
        st.info(f"TOTAL CHUNKS: {len(chunks)}")
        st.dataframe(chunks)

# for chunk in chunks:
#     st.write(chunk)
#     st.write("\n\n=========")
