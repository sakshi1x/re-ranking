from asyncio.log import logger
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



def populate_sample_data():
        directory = "./chroma_db"
        top_k = 5
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="https://jo3m4y06rnnwhaz.askbhunte.com")
        client = chromadb.PersistentClient(path=directory)
        collection_name = "ml_tools_collection"
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            logger.info(f"No existing collection to delete: Collection {collection_name} does not exist.")
        collection = client.create_collection(name=collection_name)
        
        sample_documents = [
            {"id": "1", "content": "TensorFlow is an open-source platform for machine learning and deep learning, widely used for neural network implementations.", "metadata": {"source": "TensorFlow"}},
            {"id": "2", "content": "PyTorch offers dynamic computation graphs and is preferred for research in deep learning and neural networks.", "metadata": {"source": "PyTorch"}},
            {"id": "3", "content": "Scikit-learn provides simple and efficient tools for data mining and data analysis, focusing on traditional machine learning algorithms.", "metadata": {"source": "Scikit-learn"}},
            {"id": "4", "content": "Python is the primary language used for data science and machine learning due to its extensive ecosystem.", "metadata": {"source": "Python"}},
            {"id": "5", "content": "AWS SageMaker is a fully managed service for building, training, and deploying machine learning models in the cloud.", "metadata": {"source": "AWS"}},
            {"id": "6", "content": "Keras is a high-level neural networks API that runs on top of TensorFlow, making deep learning more accessible.", "metadata": {"source": "Keras"}},
            {"id": "7", "content": "XGBoost is an optimized distributed gradient boosting library designed for efficiency and performance.", "metadata": {"source": "XGBoost"}},
            {"id": "8", "content": "Google Cloud AI Platform provides tools for the entire ML workflow from data preparation to prediction.", "metadata": {"source": "GCP"}},
            {"id": "9", "content": "Pandas is essential for data manipulation and analysis in Python machine learning workflows.", "metadata": {"source": "Pandas"}},
            {"id": "10", "content": "Hugging Face Transformers provides state-of-the-art natural language processing models and pipelines.", "metadata": {"source": "HuggingFace"}},
        ]
        
        documents = [doc["content"] for doc in sample_documents]
        embeddings = embeddings.embed_documents(documents)
        metadatas = [{"source": doc["metadata"]["source"], "id": doc["id"]} for doc in sample_documents]
        ids = [doc["id"] for doc in sample_documents]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully populated {len(sample_documents)} documents into collection.")

  