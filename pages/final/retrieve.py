
import base64
import json
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

def test():
    embeddding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = Chroma(
        persist_directory=".",
        embedding_function=embeddding_model
    )

    query = st.text_area(label="Enter Query")
    retriever = vs.as_retriever(search_kwargs={"k":10})
    matched_chunks = retriever.invoke(query)
    for i, matched_chunk in enumerate(matched_chunks):
        with st.expander(f"Matched {i}"):
            st.write(matched_chunk)

    st.write(generate_final_answer(matched_chunks, query))

def generate_final_answer(matched_chunks, query):
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        prompt_text = f"""
        Based on the following documents, please answer this question: {query}
        
        CONTENT TO ANALYZE:

        """
        for i, matched_chunk in enumerate(matched_chunks):
            prompt_text += f"-------Document {i+1}---------\n"

            if "original_content" in matched_chunk.metadata:
                original_data = json.loads(matched_chunk.metadata["original_content"])

                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"

            prompt_text += "\n"
        
        prompt_text += """
        Please provide a clear, comprehensive answer using the text, tables, and images above. 
        If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."
        
        ANSWER:
        """

        message_content = [{"type": "text", "text": prompt_text}]

        for matched_chunk in matched_chunks:
            if "original_content" in matched_chunk.metadata:
                original_data = json.loads(matched_chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])

                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    })
                    image_data = base64.b64decode(image_base64)
                    st.image(image_data)

        st.write(message_content)

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        st.write(response)
        return response.content
    
    except Exception as e:
        st.error(f"Error: {e}")
    

test()