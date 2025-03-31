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
from retrive import RetrievalToolInput  # Assuming this is defined elsewhere

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




class AdvancedRerankingTool(BaseTool):
    name: str = "AdvancedRerankingTool"
    description: str = "Enhanced retrieval with advanced reranking capabilities for any dataset."
    args_schema: type = RetrievalToolInput

    _persist_directory: str = PrivateAttr()
    _embeddings: OllamaEmbeddings = PrivateAttr()
    _client: chromadb.PersistentClient = PrivateAttr()
    _colbert_reranker: RAGPretrainedModel = PrivateAttr()
    _top_k: int = PrivateAttr()
    _collection_name: str = PrivateAttr()

    def __init__(self, persist_directory: str = "./chroma_db", top_k: int = 5, collection_name: str =  "ml_tools_collection"):
        super().__init__()
        self._persist_directory = persist_directory
        self._top_k = top_k
        self._collection_name = collection_name
        try:
            self._embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="https://jo3m4y06rnnwhaz.askbhunte.com")
            self._client = chromadb.PersistentClient(path=persist_directory)
            self._colbert_reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            logger.info("Initialized retrieval components successfully.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def _preprocess_query(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'[^\w\s?]', '', query)
        return query

    async def _retrieve_vector(self, query: str) -> List[Document]:
        collection = self._client.get_or_create_collection(self._collection_name)
        query_embedding = self._embeddings.embed_query(query)
        total_docs = len(collection.get()["ids"])
        n_results = min(self._top_k * 2, total_docs) if total_docs > 0 else self._top_k
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        docs = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            vector_score = max(0, 1 - dist)
            docs.append(Document(page_content=doc, metadata={
                **meta,
                "vector_score": vector_score,
                "bm25_score": 0.0,
                "colbert_score": 0.0,
                "context_score": 0.0,
                "retrieval_method": "vector"
            }))
        return docs

    async def _retrieve_bm25(self, query: str) -> List[Document]:
        collection = self._client.get_or_create_collection(self._collection_name)
        all_results = collection.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(all_results["documents"], all_results["metadatas"])
        ]
        
        if not all_docs:
            return []

        try:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = min(self._top_k * 2, len(all_docs))
            results = bm25_retriever.invoke(query)
            
            docs = []
            for i, doc in enumerate(results):
                bm25_score = 1.0 - (i / len(results))
                # Dynamic score boosting based on query intent
                query_lower = query.lower()
                content_lower = doc.page_content.lower()
                if "tool" in query_lower and any(kw in content_lower for kw in ["tool", "machine learning", "algorithms"]):
                    bm25_score = min(1.0, bm25_score + 0.4)
                if "framework" in query_lower and any(kw in content_lower for kw in ["framework", "neural networks"]):
                    bm25_score = min(1.0, bm25_score + 0.6)
                if "deep learning" in query_lower and any(kw in content_lower for kw in ["deep learning", "neural networks"]):
                    bm25_score = min(1.0, bm25_score + 0.5)
                if "cloud" in query_lower and "cloud" in content_lower:
                    bm25_score = min(1.0, bm25_score + 0.4)
                docs.append(Document(page_content=doc.page_content, metadata={
                    **doc.metadata,
                    "bm25_score": bm25_score,
                    "vector_score": 0.0,
                    "colbert_score": 0.0,
                    "context_score": 0.0,
                    "retrieval_method": "bm25"
                }))
            return docs
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []

    def _colbert_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        doc_texts = [doc.page_content for doc in docs]
        reranked = self._colbert_reranker.rerank(query, doc_texts, k=len(docs))
        scores = np.array([result["score"] for result in reranked])
        if scores.max() > scores.min():
            normalized = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized = np.ones_like(scores) * 0.5
        for i, result in enumerate(reranked):
            docs[result["result_index"]].metadata["colbert_score"] = normalized[i]
        return docs

    def _contextual_similarity(self, query: str, docs: List[Document]) -> List[Document]:
        query_embedding = np.array(self._embeddings.embed_query(query))
        query_terms = set(self._preprocess_query(query).split())
        for doc in docs:
            doc_embedding = np.array(self._embeddings.embed_query(doc.page_content))
            cosine_sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-6)
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            doc.metadata["context_score"] = 0.6 * cosine_sim + 0.4 * overlap  # Adjusted weights for broader context
        return docs

    def _normalize_scores(self, docs: List[Document]) -> List[Document]:
        for method in ["vector", "bm25", "colbert", "context"]:
            scores = np.array([doc.metadata.get(f"{method}_score", 0) for doc in docs])
            if scores.std() > 0:
                normalized = (scores - scores.mean()) / (scores.std() + 1e-6)
                normalized = 1 / (1 + np.exp(-normalized))  # Sigmoid normalization
                for doc, norm_score in zip(docs, normalized):
                    doc.metadata[f"{method}_score"] = norm_score
        return docs

    def _adjust_weights(self, weights: Dict[str, float], query: str) -> Dict[str, float]:
        terms = set(self._preprocess_query(query).split())
        term_count = len(terms)
        embedding = np.array(self._embeddings.embed_query(query))
        semantic_density = np.linalg.norm(embedding) / len(query)
        
        # More dynamic weight adjustment based on query complexity and intent
        if term_count <= 3:  # Short, specific queries
            if "tool" in query.lower() or "service" in query.lower():
                return {"vector": 0.3, "bm25": 0.4, "colbert": 0.2, "context": 0.1}
            return {"vector": 0.25, "bm25": 0.3, "colbert": 0.35, "context": 0.1}
        elif term_count >= 7 or semantic_density > 0.1:  # Long, complex queries
            if "learning" in query.lower() or "framework" in query.lower():
                return {"vector": 0.2, "bm25": 0.35, "colbert": 0.3, "context": 0.15}
            return {"vector": 0.35, "bm25": 0.2, "colbert": 0.25, "context": 0.2}
        return weights

    def _advanced_fusion(self, docs: List[Document], weights: Dict[str, float], query: str) -> List[Document]:
        method_ranks = {}
        for method in ["vector", "bm25", "colbert", "context"]:
            sorted_docs = sorted(docs, key=lambda x: x.metadata.get(f"{method}_score", 0), reverse=True)
            method_ranks[method] = {doc.page_content: 1 / (i + 1) for i, doc in enumerate(sorted_docs)}
        
        for doc in docs:
            rrf_score = sum(weights[m] * method_ranks[m].get(doc.page_content, 0) for m in weights)
            doc.metadata["final_score"] = min(1.0, rrf_score * 1.5)  # Amplify scores for clearer differentiation
        return sorted(docs, key=lambda x: x.metadata["final_score"], reverse=True)[:self._top_k]

    async def _run(self, query: str, weights: Dict[str, float]) -> str:
        query = self._preprocess_query(query)
        if not query:
            return "Empty query provided."
        
        weights = self._adjust_weights(weights, query)
        
        vector_docs = await self._retrieve_vector(query)
        bm25_docs = await self._retrieve_bm25(query)
        
        content_to_doc = {}
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in content_to_doc:
                content_to_doc[doc.page_content] = Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": doc.metadata.get("source", "Unknown"),
                        "id": doc.metadata.get("id", "N/A"),
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "colbert_score": 0.0,
                        "context_score": 0.0
                    }
                )
            existing_doc = content_to_doc[doc.page_content]
            existing_doc.metadata["vector_score"] = max(existing_doc.metadata["vector_score"], doc.metadata["vector_score"])
            existing_doc.metadata["bm25_score"] = max(existing_doc.metadata["bm25_score"], doc.metadata["bm25_score"])

        combined_docs = list(content_to_doc.values())
        if not combined_docs:
            return "No results found in local data."
        
        combined_docs = self._colbert_rerank(query, combined_docs)
        combined_docs = self._contextual_similarity(query, combined_docs)
        combined_docs = self._normalize_scores(combined_docs)
        final_docs = self._advanced_fusion(combined_docs, weights, query)

        output = []
        query_terms = set(self._preprocess_query(query).split())
        for i, doc in enumerate(final_docs, 1):
            score = doc.metadata["final_score"]
            reason = "Highly relevant" if score > 0.75 else "Relevant" if score > 0.5 else "Moderately relevant"
            doc_terms = set(doc.page_content.lower().split())
            matched_terms = query_terms & doc_terms
            if doc.metadata["vector_score"] > 0.7 or doc.metadata["colbert_score"] > 0.7:
                reason += ", strong semantic match"
            if doc.metadata["context_score"] > 0.7:
                reason += ", strong contextual match"
            if matched_terms:
                reason += f", matched terms: {', '.join(sorted(matched_terms))}"
            output.append(
                f"{i}. [Score: {score:.3f}, ID: {doc.metadata['id']}, Source: {doc.metadata['source']}]\n"
                f"Content: {doc.page_content}\n"
                f"Why: {reason}\n"
                f"Breakdown: Vector={doc.metadata['vector_score']:.3f}, "
                f"BM25={doc.metadata['bm25_score']:.3f}, "
                f"ColBERT={doc.metadata['colbert_score']:.3f}, "
                f"Context={doc.metadata['context_score']:.3f}"
            )
        return "\n\n".join(output) if output else "No results found."

    async def evaluate_accuracy(self, test_cases: List[Dict[str, List[str]]]) -> None:
        """Evaluate accuracy with customizable test cases."""
        weights = {"vector": 0.25, "bm25": 0.25, "colbert": 0.3, "context": 0.2}
        total_precision, total_recall, total_f1 = 0, 0, 0
        
        for case in test_cases:
            results = await self._run(case["query"], weights)
            retrieved_docs = results.split("\n\n")
            retrieved_contents = []
            for doc in retrieved_docs:
                content_match = re.search(r"Content: (.+?)(?=\nWhy:|\nBreakdown:|$)", doc, re.DOTALL)
                if content_match:
                    retrieved_contents.append(content_match.group(1).strip().lower())
            
            # Flatten expected terms for flexible matching
            expected_terms = set()
            for term in case["expected"]:
                expected_terms.update(term.lower().split())
            
            retrieved_set = set()
            for i, content in enumerate(retrieved_contents[:self._top_k]):
                content_terms = set(content.split())
                if expected_terms & content_terms:
                    retrieved_set.add(i)
            
            expected_set = set(range(min(len(case["expected"]), self._top_k)))
            precision = len(retrieved_set & expected_set) / max(len(retrieved_set), 1)
            recall = len(retrieved_set & expected_set) / max(len(expected_set), 1)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nTest Case: {case['name']}\nQuery: {case['query']}\nExpected Terms: {case['expected']}\nRetrieved Contents: {retrieved_contents[:3]}\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        num_cases = len(test_cases)
        print(f"\n=== Overall Metrics ===\nAverage Precision: {total_precision / num_cases:.3f}\nAverage Recall: {total_recall / num_cases:.3f}\nAverage F1 Score: {total_f1 / num_cases:.3f}")

