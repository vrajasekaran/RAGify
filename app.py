import streamlit as st

st.set_page_config(
    page_title = "RAGify"
)

st.write("RAGify anything")

pages = {
    "Tokenizer":[
        st.Page("pages/1_tokenizer.py", title="Tokenizers"),
    ],
    "Text Splitters":[
        st.Page("pages/splitters/CHAR.py", title="CharacterTextSplitter"),
        st.Page("pages/splitters/REC_CHAR.py", title="RecursiveCharacterTextSplitter"),
        st.Page("pages/splitters/MD_TEXT.py", title="MarkdownTextSplitter"),
        st.Page("pages/splitters/PDF.py", title="Unstructured.io partitions")
    ]
}

pg = st.navigation(pages)
pg.run()
