import streamlit as st
from langchain_text_splitters import CharacterTextSplitter

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseCharSplitter import BaseCharSplitter

# Create an instance of BaseCharSplitter with CharacterTextSplitter
BaseCharSplitter(CharacterTextSplitter())
