import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import chromadb
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, Field, PrivateAttr
from ragatouille import RAGPretrainedModel
import logging
import re
import warnings
# from retrive import RetrievalToolInput
class RetrievalToolInput(BaseModel):
    query: str = Field(..., description="The query to retrieve information for")
    weights: Optional[Dict[str, float]] = Field(default={"vector": 0.25, "bm25": 0.25, "colbert": 0.3, "context": 0.2})
