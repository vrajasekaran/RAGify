import streamlit as st
from langchain_text_splitters import CharacterTextSplitter

from pages.splitters.BaseCharSplitter import BaseCharSplitter

# Create an instance of BaseCharSplitter with CharacterTextSplitter
BaseCharSplitter(CharacterTextSplitter())
