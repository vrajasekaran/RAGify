from transformers import BertModel, AutoTokenizer
import pandas as pd
import streamlit as st

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizerBert = AutoTokenizer.from_pretrained(model_name)

input_txt = st.text_area(label="Enter your Text")

#sentence = "When will you be Back?"

if (st.button('TOKENIZE WITH BERTMODEL')):
    tokens = tokenizerBert.tokenize(input_txt)   
    st.write(tokens)


import tiktoken

if (st.button('TOKENIZE WITH TIKTOKEN')):
    enc = tiktoken.get_encoding("o200k_base")
    st.write(enc.encode(input_txt))
    st.write(enc.decode(enc.encode(input_txt)))


if (st.button('TOKENIZE WITH GEMMA')):
    tokenizerGemma = AutoTokenizer.from_pretrained("gemma-2-2b")
    st.write(tokenizerGemma.tokenize(input_txt))

# from tiktoken._educational import *

# Train a BPE tokeniser on a small amount of text
# enc = train_simple_encoding()

# # Visualise how the GPT-4 encoder encodes text
# st.write(enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base"))
# st.write(enc.encode("hello world aaaaaaaaaaaa"))