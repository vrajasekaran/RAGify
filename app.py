import streamlit as st

st.set_page_config(
    page_title = "RAGify"
)

st.write("RAGify anything")

pages = {
    "Tokenizer":[
        st.Page("pages/1_tokenizer/tokenizer.py", title="Tokenizers"),
    ],
    "Text Splitters":[
        st.Page("pages/2_splitters/chara.py", title="CharacterTextSplitter"),
        st.Page("pages/2_splitters/rec_char.py", title="RecursiveCharacterTextSplitter"),
        st.Page("pages/2_splitters/md_text.py", title="MarkdownTextSplitter"),
        st.Page("pages/2_splitters/pdf.py", title="Unstructured.io partitions")
    ],
    "Chunkers":[
        st.Page("pages/3_chunk/title.py", title="Unstructured.io Chunk By Title"),
        st.Page("pages/3_chunk/basic.py", title="Unstructured.io - Basic Chunk By Elements")
    ],
    "Image Summarizer":[
        st.Page("pages/4_image_summarizer/image_sum.py", title="Image Summarizer through LLM")
    ],
    "RAG RAG":[
        st.Page("pages/final/final.py", title="RAG RAG"),
        st.Page("pages/final/retrieve.py", title="RAG Retrieve")
    ]
}

pg = st.navigation(pages)
pg.run()
