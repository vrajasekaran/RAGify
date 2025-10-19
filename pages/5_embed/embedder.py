import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import pandas as pd
from langchain_ollama import OllamaEmbeddings

    
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
    nomic_embedings = OllamaEmbeddings(model="nomic-embed-text").embed_query(input_text)
    pd.DataFrame(nomic_embedings)
    st.write(nomic_embedings)

if (st.button(
    label="GENERATE EMBEDDING GEMMA EMBEDDINGS"
)):
    nomic_embedings = OllamaEmbeddings(model="embeddinggemma").embed_query(input_text)
    pd.DataFrame(nomic_embedings)
    st.write(nomic_embedings)
