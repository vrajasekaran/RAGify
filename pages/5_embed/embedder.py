import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import pandas as pd

load_dotenv()

input_text = st.text_area(
    label="Enter Text",
    value="Input Text to generate embeddings"
)

if (st.button(
    label="GENERATE OPENAI EMBEDDINGS"
)):
    openai_embeddings = OpenAIEmbeddings().embed_query(input_text)
    pd.DataFrame(openai_embeddings)
    st.write(openai_embeddings)


if (st.button(
    label="GENERATE NOMIC EMBEDDINGS"
)):
    
    openai_embeddings = OpenAIEmbeddings().embed_query(input_text)
    pd.DataFrame(openai_embeddings)
    st.write(openai_embeddings)