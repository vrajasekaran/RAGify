import pandas as pd
import streamlit as st
import os
import base64
from unstructured.partition.pdf import partition_pdf

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
for element in elements:
    # st.dataframe(element)
    st.write(element.category)
    st.write(element.to_dict())
    if (element.category == "Image") and hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
        image_data = base64.b64decode(element.metadata.image_base64)
        st.image(image_data)