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

    "Chunkers - LangChain":[
        st.Page("pages/3_chunk/lang/fixed.py", title="Fixed Size Chunking"),
        st.Page("pages/3_chunk/lang/semantic.py", title="Semantic Chunking"),
        
    ],
    "Chunkers - Unstructured.io":[
        st.Page("pages/3_chunk/un/title.py", title="Unstructured.io Chunk By Title"),
        st.Page("pages/3_chunk/un/basic.py", title="Unstructured.io - Basic Chunk By Elements")
    ],
    "Image Summarizer":[
        st.Page("pages/4_image_summarizer/image_sum.py", title="Image Summarizer through LLM")
    ],
    "Embedder":[
        st.Page("pages/5_embed/embedder.py", title="Generate Embeddings"),
        st.Page("pages/5_embed/embed_vis.py", title="Visualize Embeddings")
    ],
    "RAG RAG":[
        st.Page("pages/final/final.py", title="RAG RAG"),
        st.Page("pages/final/retrieve.py", title="RAG Retrieve")
    ]
}

pg = st.navigation(pages)
pg.run()
