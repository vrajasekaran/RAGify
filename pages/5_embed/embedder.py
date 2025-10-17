import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import streamlit as st

load_dotenv()

input_text = st.text_area(
    label="Enter Text",
    value="Input Text to generate embeddings"
)

if (st.button(
    label="GENERATE EMBEDDINGS"
)):
    openai_embeddings = OpenAIEmbeddings().embed_query(input_text)
    st.write(openai_embeddings)