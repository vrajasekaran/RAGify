import json
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
import pandas as pd
import streamlit as st
import os
import base64
from rich import print 
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

input_pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)
st.pdf(input_pdf)

elements = partition_pdf(
    file=input_pdf,
    # filename=input_pdf,

    strategy="hi_res",
    hi_res_model_name="yolox",
    extract_image_block_types=['Table', 'Image'],
    extract_images_in_pdf=True,
    extract_image_block_to_payload=True,
    infer_table_structure=True
)

st.info(f"TOTAL partitions: {len(elements)}")
st.info(f"TOTAL Image partitions: {len([e for e in elements if e.category == 'Image'])}")
st.info(f"TOTAL Table partitions: {len([e for e in elements if e.category == 'Table'])}")
# st.dataframe({elements})
# for element in elements:
#     # st.dataframe(element)
#     st.write(element.category)
#     st.write(element.to_dict())
#     if (element.category == "Image") and hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
#         image_data = base64.b64decode(element.metadata.image_base64)
#         st.image(image_data)


# from unstructured.chunking.basic import chunk_elements

# chunks = chunk_elements(
#     elements=elements,
#     max_characters=1000,
#     new_after_n_chars=800
# )


from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(
    elements=elements,
    max_characters=500,
    new_after_n_chars=200,
    combine_text_under_n_chars=50
)

st.dataframe(chunks)
st.info(f"TOTAL chunks: {len(chunks)}")
st.info(f"TOTAL Image chunks: {len([e for e in chunks if e.category == 'Image'])}")
st.info(f"TOTAL Table chunks: {len([e for e in chunks if e.category == 'Table'])}")
# st.dataframe({elements})
for i, element in enumerate(chunks):
    # st.dataframe(element)
    with st.expander(f"processing chunk: {i}", expanded=False):
        st.write(element.category)
        st.write(element.to_dict())
        if (element.category == "Image") and hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
            image_data = base64.b64decode(element.metadata.image_base64)
            st.image(image_data)


def log_msg(msg: str):
    print(msg)
    st.write(msg)
    # yield

def split_content_by_type(chunk):
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    if(hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements')):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == 'Table':
                st.write('Table identified')
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
                st.html(table_html)

            elif element_type == 'Image':
                st.write('Image identified')
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
                    image_data = base64.b64decode(element.metadata.image_base64)
                    st.image(image_data)

    content_data['types'] = list(set(content_data['types']))
    return content_data


def create_summary(text: str, tables: list[str], images: list[str]):
   

    try:
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # llm = ChatOllama(model="qwen2.5vl", temperature=0)
        llm = ChatOllama(model="gemma3", temperature=0)
        # llm = ChatOllama(model="gpt-oss", temperature=0)

        # llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """

        if tables:
            prompt_text += "TABLES: \n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:
                """
        message_content = [{"type":"text", "text": prompt_text }]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        st.write(message_content)
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        st.write(f"Summary failed: {e}")
        
    

def summarize_chunks(chunks) -> list[Document]:
    """"""
    st.write(f'Started Summarizing chunks, TOTAL: {len(chunks)}')
    docs = []

    for i, chunk in enumerate(chunks):
        with st.expander(f"Summarizing chunk: {i}", expanded=False):
            content_data = split_content_by_type(chunk)
            st.write(content_data)
            
            if content_data['tables'] or content_data['images']:
                try:
                    st.write('creating AI summary')
                    enhanced_content = create_summary(
                        content_data['text'],
                        content_data['tables'],
                        content_data['images']
                    )
                except Exception as e:
                    enhanced_content = content_data['text']

            else:
                enhanced_content = content_data['text']

            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps({
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "images_base64": content_data['images']
                    })
                }
            )

            st.write(doc)
            docs.append(doc)

    return docs

def create_vector_store(docs):
    # embeddding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    embeddding_model = OllamaEmbeddings(model="embeddinggemma")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddding_model,
        persist_directory="./gemma",
        # collection_metadata=

    )
    return vector_store



st.info("summarize_chunks")
summarized_docs = summarize_chunks(chunks)
st.info("creating vector store")
vs = create_vector_store(summarized_docs)
# pd.DataFrame(vs)
st.info("Finished Successfully!!!")

