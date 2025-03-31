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
# from retrive import RetrievalToolInput
from rerank import AdvancedRerankingTool

async def test_tool():
    populate_sample_data()
    tool = AdvancedRerankingTool()
    queries = [
        "What is the best tool for machine learning?",
        "deep learning framework",
        "cloud ML deployment services",
        "Python libraries for data science",
        "neural network implementation tools",
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

        test_cases = [
            {"name": "General ML tools", "query": "best tool for machine learning", "expected": ["machine learning", "tool", "algorithms"]},
            {"name": "DL frameworks", "query": "deep learning framework", "expected": ["deep learning", "framework", "neural networks"]},
            {"name": "Cloud ML services", "query": "cloud ML deployment", "expected": ["cloud", "deployment", "ML"]},
            {"name": "Python data tools", "query": "Python data analysis", "expected": ["Python", "data analysis", "tool"]},
            {"name": "NN APIs", "query": "neural networks API", "expected": ["neural networks", "API"]},
        ]

        # Evaluate accuracy with the test cases
        f.write("\n=== Starting Accuracy Evaluation ===\n")
        print("\n=== Starting Accuracy Evaluation ===")

        await tool.evaluate_accuracy(test_cases)
        # f.write(str(accuracy_results) + "\n")
        # print("\n=== Accuracy Evaluation Results ===\n", accuracy_results)

if __name__ == "__main__":
    asyncio.run(test_tool())
