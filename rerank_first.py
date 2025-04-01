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

# Assuming RetrievalToolInput is defined elsewhere
from retrive import RetrievalToolInput

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class RerankingTool(BaseTool):
    """
    A generic reranking tool for advanced retrieval and ranking of documents across any domain.
    Combines vector-based search, BM25, ColBERT reranking, and contextual similarity with flexible configurations.
    """

    name: str = "GenericRerankingTool"
    description: str = "Performs enhanced document retrieval and ranking for any dataset or domain."
    args_schema: type = RetrievalToolInput

    # Private attributes
    _persist_directory: str = PrivateAttr()
    _embeddings: OllamaEmbeddings = PrivateAttr()
    _client: chromadb.PersistentClient = PrivateAttr()
    _colbert_reranker: RAGPretrainedModel = PrivateAttr()
    _top_k: int = PrivateAttr()
    _collection_name: str = PrivateAttr()
    _default_weights: Dict[str, float] = PrivateAttr()

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        top_k: int = 5,
        collection_name: str = "generic_collection",
        embedding_model: str = "nomic-embed-text",
        embedding_url: str = "",
        default_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the reranking tool with customizable configurations.
        """
        super().__init__()
        self._persist_directory = persist_directory
        self._top_k = top_k
        self._collection_name = collection_name
        self._default_weights = default_weights or {"vector": 0.25, "bm25": 0.25, "colbert": 0.3, "context": 0.2}

        try:
            self._embeddings = OllamaEmbeddings(model=embedding_model, base_url=embedding_url)
            self._client = chromadb.PersistentClient(path=persist_directory)
            self._colbert_reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            logger.info("Initialized retrieval components successfully.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError(f"Initialization failed: {e}")

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query with generic cleaning suitable for any domain.
        """
        query = query.strip().lower()
        query = re.sub(r'[^\w\s?!]', '', query)  # Keep words, spaces, and basic punctuation
        return " ".join(query.split())  # Remove extra whitespace

    async def _retrieve_vector(self, query: str) -> List[Document]:
        """
        Retrieve documents using vector similarity with embeddings.
        """
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
                **(meta or {}),
                "vector_score": vector_score,
                "bm25_score": 0.0,
                "colbert_score": 0.0,
                "context_score": 0.0,
                "retrieval_method": "vector"
            }))
        return docs

    async def _retrieve_bm25(self, query: str) -> List[Document]:
        """
        Retrieve documents using BM25 ranking.
        """
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
                bm25_score = 1.0 - (i / len(results))  # Simple positional scoring
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
        """
        Rerank documents using ColBERT for fine-tuned semantic similarity.
        """
        doc_texts = [doc.page_content for doc in docs]
        reranked = self._colbert_reranker.rerank(query, doc_texts, k=len(docs))
        scores = np.array([result["score"] for result in reranked])

        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6) if scores.max() > scores.min() else np.ones_like(scores) * 0.5
        for i, result in enumerate(reranked):
            docs[result["result_index"]].metadata["colbert_score"] = normalized[i]
        return docs

    def _contextual_similarity(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Calculate contextual similarity generically based on embeddings and term overlap.
        """
        query_embedding = np.array(self._embeddings.embed_query(query))
        query_terms = set(self._preprocess_query(query).split())

        for doc in docs:
            doc_embedding = np.array(self._embeddings.embed_query(doc.page_content))
            cosine_sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-6)
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            doc.metadata["context_score"] = 0.7 * cosine_sim + 0.3 * overlap  # Adjusted for generality
        return docs

    def _normalize_scores(self, docs: List[Document]) -> List[Document]:
        """
        Normalize scores across all methods for consistent comparison.
        """
        for method in ["vector", "bm25", "colbert", "context"]:
            scores = np.array([doc.metadata.get(f"{method}_score", 0) for doc in docs])
            if scores.std() > 0:
                normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
                for doc, norm_score in zip(docs, normalized):
                    doc.metadata[f"{method}_score"] = norm_score
        return docs

    def _adjust_weights(self, weights: Dict[str, float], query: str) -> Dict[str, float]:
        """
        Dynamically adjust weights based on query characteristics, avoiding domain-specific assumptions.
        """
        terms = set(self._preprocess_query(query).split())
        term_count = len(terms)
        embedding = np.array(self._embeddings.embed_query(query))
        semantic_density = np.linalg.norm(embedding) / max(len(query), 1)

        if term_count <= 3:  # Short queries favor precision
            return {"vector": 0.3, "bm25": 0.35, "colbert": 0.25, "context": 0.1}
        elif term_count >= 7 or semantic_density > 0.15:  # Long or dense queries favor semantics
            return {"vector": 0.35, "bm25": 0.2, "colbert": 0.3, "context": 0.15}
        return weights  # Default to provided weights for balanced queries

    def _advanced_fusion(self, docs: List[Document], weights: Dict[str, float], query: str) -> List[Document]:
        """
        Combine scores using a generic ranking fusion strategy.
        """
        method_ranks = {}
        for method in ["vector", "bm25", "colbert", "context"]:
            sorted_docs = sorted(docs, key=lambda x: x.metadata.get(f"{method}_score", 0), reverse=True)
            method_ranks[method] = {doc.page_content: 1 / (i + 1) for i, doc in enumerate(sorted_docs)}

        for doc in docs:
            rrf_score = sum(weights[m] * method_ranks[m].get(doc.page_content, 0) for m in weights)
            doc.metadata["final_score"] = min(1.0, rrf_score * 1.2)  # Slightly reduced amplification

        return sorted(docs, key=lambda x: x.metadata["final_score"], reverse=True)[:self._top_k]

    async def _run(self, query: str, weights: Optional[Dict[str, float]] = None) -> str:
        """
        Execute the full retrieval and reranking process.
        """
        query = self._preprocess_query(query)
        if not query:
            return "Empty query provided."

        weights = self._adjust_weights(weights or self._default_weights, query)

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
        """
        Evaluate accuracy using precision, recall, and F1 metrics.
        """
        weights = self._default_weights
        total_precision, total_recall, total_f1 = 0, 0, 0

        for case in test_cases:
            results = await self._run(case["query"], weights)
            retrieved_docs = results.split("\n\n")
            retrieved_contents = [
                re.search(r"Content: (.+?)(?=\nWhy:|\nBreakdown:|$)", doc, re.DOTALL).group(1).strip().lower()
                for doc in retrieved_docs if re.search(r"Content: (.+?)(?=\nWhy:|\nBreakdown:|$)", doc, re.DOTALL)
            ]

            expected_terms = set(" ".join(case["expected"]).lower().split())
            retrieved_set = set()
            for i, content in enumerate(retrieved_contents[:self._top_k]):
                content_terms = set(content.split())
                if expected_terms & content_terms:
                    retrieved_set.add(i)

            expected_set = set(range(min(len(case["expected"]), self._top_k)))
            precision = len(retrieved_set & expected_set) / max(len(retrieved_set), 1)
            recall = len(retrieved_set & expected_set) / max(len(expected_set), 1)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        print(f"Average Precision: {total_precision / len(test_cases):.3f}")
        print(f"Average Recall: {total_recall / len(test_cases):.3f}")
        print(f"Average F1: {total_f1 / len(test_cases):.3f}")