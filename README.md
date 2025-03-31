

# RerankingTool

A versatile retrieval and reranking system built with ChromaDB, BM25, ColBERT, and contextual similarity, designed to rank documents for any dataset with adaptive scoring.

## Overview

`RerankingTool` is a Python-based tool that combines multiple retrieval and reranking techniques to deliver ranked document results for user queries. It supports ingestion of arbitrary document sets and dynamically adjusts weights to optimize relevance.

### Key Features
- **Vector Retrieval**: Uses ChromaDB with `OllamaEmbeddings` for semantic search.
- **BM25 Retrieval**: Keyword-based ranking with custom boosting logic.
- **ColBERT Reranking**: Contextual reranking with the `colbertv2.0` model.
- **Contextual Similarity**: Combines cosine similarity and term overlap for refined scoring.
- **Fusion**: Merges results using reciprocal rank fusion (RRF) with adaptive weights (e.g., vector: 0.25, BM25: 0.25, ColBERT: 0.3, context: 0.2).
- **Dynamic Evaluation**: Generates test cases from ingested documents to compute precision, recall, and F1 scores.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sakshi-rumsan/ranking_model.git
   cd ..
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Requires Python 3.9+. Key dependencies include `chromadb`, `langchain`, `ragatouille`, and `numpy`.

3. Ensure the Ollama API (``) is accessible or replace with a local embedding model.

## Current Performance
- **Dataset**: Tested with 10 documents.
- **Metrics**: Average Precision: 0.723, Recall: 0.867, F1: 0.731.
- **Scalability**: Suitable for small-to-medium datasets (100s-1000s of documents); bottlenecks exist for larger scales.

## Scalability Considerations
- **Strengths**: Efficient vector search with ChromaDB, lightweight fusion.
- **Limitations**: 
  - API-based embeddings (`OllamaEmbeddings`) limit throughput.
  - In-memory BM25 indexing scales poorly.
  - ColBERT and contextual similarity become compute-intensive with more documents.

## Next Steps
- **Optimization**: Precompute embeddings, persist BM25 index, and limit ColBERT reranking for scalability to larger datasets (e.g., millions of documents).
- **Monitoring**: Integrate with LangChain Smith for active performance tracking (precision, recall) and real-time refinement.
- **Enhancements**: Add caching, parallel processing, and distributed deployment options.

## Contributing
Feel free to submit issues or pull requests! Focus areas:
- Scalability improvements.
- Alternative embedding models.
- Enhanced query intent detection.

## License
MIT License - see [LICENSE](LICENSE) for details.

---
