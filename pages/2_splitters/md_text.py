from langchain_text_splitters import MarkdownTextSplitter

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseCharSplitter import BaseCharSplitter

BaseCharSplitter(MarkdownTextSplitter())