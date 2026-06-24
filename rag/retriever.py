"""
CARA - Greenfield Medical Centre AI Receptionist
retriever.py - Hybrid retrieval using Dense + BM25 + RRF
"""

import os
import sys
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "chroma_db")
TOP_K_RETRIEVAL = 15   # how many chunks to retrieve from each method before RRF
TOP_K_FINAL = 5        # final chunks to return after RRF
RRF_K = 60             # standard RRF constant

# Map intent to collection name
INTENT_TO_COLLECTION = {
    "appointments": "appointments",
    "triage": "triage",
    "prescription": "prescription",
    "hours": "hours",
    "general": None   # None means search all collections
}


class EMMARetriever:
    """
    Hybrid retriever combining Dense semantic search
    and BM25 keyword search, fused with RRF.
    """

    def __init__(self):
        print("  [Retriever] Loading embedding model ...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("  [Retriever] Connecting to ChromaDB ...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Load all 4 collections
        self.collections = {
            "appointments": self.chroma_client.get_collection("appointments"),
            "triage": self.chroma_client.get_collection("triage"),
            "prescription": self.chroma_client.get_collection("prescription"),
            "hours": self.chroma_client.get_collection("hours"),
        }

        # Build BM25 index for each collection at startup
        print("  [Retriever] Building BM25 indexes ...")
        self.bm25_indexes = {}
        self.collection_texts = {}
        self.collection_ids = {}
        self.collection_metadatas = {}

        for name, col in self.collections.items():
            data = col.get()
            texts = data["documents"]
            ids = data["ids"]
            metadatas = data["metadatas"]

            self.collection_texts[name] = texts
            self.collection_ids[name] = ids
            self.collection_metadatas[name] = metadatas

            # Tokenize for BM25
            tokenized = [text.lower().split() for text in texts]
            self.bm25_indexes[name] = BM25Okapi(tokenized)

        print("  [Retriever] Ready.\n")

    # ─────────────────────────────────────────
    # MAIN RETRIEVE METHOD
    # ─────────────────────────────────────────
    def retrieve(self, query: str, intent: str) -> list[dict]:
        """
        Main retrieval method.

        Args:
            query: combined retrieval query (summary + last 6 turns + current message)
            intent: classified intent from intent_classifier

        Returns:
            List of top 3 chunks with text, score, metadata
        """

        # Determine which collections to search
        collection_name = INTENT_TO_COLLECTION.get(intent, None)

        if collection_name:
            # Search specific collection
            collections_to_search = [collection_name]
        else:
            # General intent — search all collections
            collections_to_search = list(self.collections.keys())

        print(f"  [Retriever] Intent: {intent} → searching: {collections_to_search}")

        # Collect results across all relevant collections
        all_results = []
        for col_name in collections_to_search:
            results = self._hybrid_search(query, col_name)
            all_results.extend(results)

        if not all_results:
            print("  [Retriever] No results found")
            return []

        # If searching multiple collections — re-rank across all
        if len(collections_to_search) > 1:
            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

        # Return top K final
        top_results = all_results[:TOP_K_FINAL]

        print(f"  [Retriever] Returning {len(top_results)} chunks:")
        for i, r in enumerate(top_results):
            print(f"    {i+1}. score={r['score']:.3f} | {r['text'][:80]}...")

        return top_results

    # ─────────────────────────────────────────
    # HYBRID SEARCH FOR ONE COLLECTION
    # ─────────────────────────────────────────
    def _hybrid_search(self, query: str, collection_name: str) -> list[dict]:
        """
        Runs Dense + BM25 on one collection, fuses with RRF.

        Returns list of dicts with text, score, metadata
        """

        # ── DENSE RETRIEVAL ──────────────────
        query_embedding = self.model.encode(query).tolist()
        collection = self.collections[collection_name]
        total_chunks = collection.count()
        n_results = min(TOP_K_RETRIEVAL, total_chunks)

        dense_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        dense_docs = dense_results["documents"][0]
        dense_ids = dense_results["ids"][0]
        dense_distances = dense_results["distances"][0]

        # Convert cosine distance to similarity score
        # ChromaDB returns distance (lower = more similar)
        # Convert: similarity = 1 - distance
        dense_scores = [1 - d for d in dense_distances]

        # ── BM25 RETRIEVAL ───────────────────
        tokenized_query = query.lower().split()
        bm25 = self.bm25_indexes[collection_name]
        all_texts = self.collection_texts[collection_name]
        all_ids = self.collection_ids[collection_name]
        all_metadatas = self.collection_metadatas[collection_name]

        bm25_scores_raw = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores_raw)[::-1][:TOP_K_RETRIEVAL]

        bm25_ids = [all_ids[i] for i in top_bm25_indices]
        bm25_docs = [all_texts[i] for i in top_bm25_indices]
        bm25_metadatas_list = [all_metadatas[i] for i in top_bm25_indices]

        # ── RRF FUSION ───────────────────────
        # Collect all unique chunk ids
        all_chunk_ids = list(set(dense_ids + bm25_ids))
        rrf_scores = {}

        for chunk_id in all_chunk_ids:
            score = 0.0

            # Dense rank contribution
            if chunk_id in dense_ids:
                rank = dense_ids.index(chunk_id) + 1
                score += 1.0 / (rank + RRF_K)

            # BM25 rank contribution
            if chunk_id in bm25_ids:
                rank = bm25_ids.index(chunk_id) + 1
                score += 1.0 / (rank + RRF_K)

            rrf_scores[chunk_id] = score

        # Sort by RRF score descending
        ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = ranked_ids[:TOP_K_FINAL]

        # ── BUILD RESULT LIST ────────────────
        results = []
        for chunk_id in top_ids:
            # Get text and metadata
            if chunk_id in dense_ids:
                idx = dense_ids.index(chunk_id)
                text = dense_docs[idx]
                metadata = dense_results["metadatas"][0][idx]
                # Use cosine similarity as score
                cosine_score = dense_scores[idx]
            elif chunk_id in bm25_ids:
                idx = bm25_ids.index(chunk_id)
                text = bm25_docs[idx]
                metadata = bm25_metadatas_list[idx]
                # BM25 only result — use lower score
                cosine_score = 0.35
            else:
                continue

            results.append({
                "text": text,
                "score": cosine_score,
                "rrf_score": rrf_scores[chunk_id],
                "metadata": metadata,
                "collection": collection_name
            })

        return results


# ─────────────────────────────────────────
# SINGLETON — one instance shared across app
# ─────────────────────────────────────────
_retriever_instance = None

def get_retriever() -> EMMARetriever:
    """Returns singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EMMARetriever()
    return _retriever_instance


# ─────────────────────────────────────────
# TEST — run directly to verify
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Retriever...\n")
    retriever = get_retriever()

    test_cases = [
        ("I need appointment with Dr Patel tomorrow morning", "appointments"),
        ("I have chest pain and difficulty breathing", "triage"),
        ("I need my repeat prescription for blood pressure tablets", "prescription"),
        ("What time does the surgery open on Saturday", "hours"),
        ("I have angina and want to see a doctor today", "triage"),
        ("How do I register as a new patient", "hours"),
    ]

    for query, intent in test_cases:
        print("=" * 60)
        print(f"Query : {query}")
        print(f"Intent: {intent}")
        chunks = retriever.retrieve(query, intent)
        print(f"\nTop {len(chunks)} chunks retrieved:")
        for i, chunk in enumerate(chunks):
            print(f"\n  Chunk {i+1} (score={chunk['score']:.3f}):")
            print(f"  {chunk['text'][:150]}...")
        print()