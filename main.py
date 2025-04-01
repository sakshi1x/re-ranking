import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from ingest import populate_sample_data
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
from rerank import RerankingTool  # Assuming this is your generic RerankingTool from the previous response

logger = logging.getLogger(__name__)

async def test_tool():
    """
    Test the RerankingTool with a variety of queries and evaluate its accuracy.
    """
    # Populate the database with generic data
    populate_sample_data(
        directory="./chroma_db",
        collection_name="generic_collection",
        embedding_url="https://jo3m4y06rnnwhaz.askbhunte.com"  # Adjust as needed
    )

    # Initialize the reranking tool
    tool = RerankingTool(
        persist_directory="./chroma_db",
        top_k=5,
        collection_name="generic_collection",
        embedding_url="https://jo3m4y06rnnwhaz.askbhunte.com"
    )

    # Test queries spanning multiple domains
    queries = [
        "best strategies for customer retention",
        "cloud computing benefits",
        "machine learning for market prediction",
        "sustainable farming techniques",
        "telemedicine advantages",
        "blockchain in finance",
        "online learning platforms",
        "renewable energy solutions",
        "AI in autonomous vehicles",
        "supply chain optimization methods",
    ]
    weights = {"vector": 0.25, "bm25": 0.25, "colbert": 0.3, "context": 0.2}

    with open("eval.txt", "w") as f:
        f.write("=== Starting Retrieval Tests ===\n")

        for query in queries:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\nTesting Query: {query}\nDate: {timestamp}\n")
            print(f"\nTesting Query: {query}\nDate: {timestamp}")

            results = await tool._run(query, weights)

            f.write("\n=== Results ===\n" + str(results) + "\n")
            print("\n=== Results ===\n", results)

        # Test cases for accuracy evaluation
        test_cases = [
            {"name": "Customer Retention", "query": "best strategies for customer retention", "expected": ["loyalty programs", "personalized offers"]},
            {"name": "Cloud Computing", "query": "cloud computing benefits", "expected": ["scalability", "cost reduction"]},
            {"name": "Market Prediction", "query": "machine learning for market prediction", "expected": ["machine learning", "predict"]},
            {"name": "Sustainable Farming", "query": "sustainable farming techniques", "expected": ["crop yield", "environment"]},
            {"name": "Telemedicine", "query": "telemedicine advantages", "expected": ["healthcare", "rural"]},
            {"name": "Blockchain", "query": "blockchain in finance", "expected": ["secure", "transactions"]},
            {"name": "Online Learning", "query": "online learning platforms", "expected": ["education", "flexible"]},
            {"name": "Renewable Energy", "query": "renewable energy solutions", "expected": ["solar", "climate"]},
            {"name": "Autonomous Vehicles", "query": "AI in autonomous vehicles", "expected": ["artificial intelligence", "safety"]},
            {"name": "Supply Chain", "query": "supply chain optimization methods", "expected": ["costs", "delivery"]},
        ]

        f.write("\n=== Starting Accuracy Evaluation ===\n")
        print("\n=== Starting Accuracy Evaluation ===")

        await tool.evaluate_accuracy(test_cases)

if __name__ == "__main__":
    asyncio.run(test_tool())