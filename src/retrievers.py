"""Bốn phương pháp Retrieval cho Day 07: TF-IDF, Okapi BM25, Dense (Qdrant + EmbeddingGemma-2) và Hybrid RRF.

Tuân thủ hợp đồng `Retriever.search(query, *, filters=None, top_k=3)` trong `starter/agentic.py`.
Tuyệt đối không sử dụng nhãn `gold` trong bất kỳ bước suy luận filter hay xếp hạng nào.
"""
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY06_ROOT = REPO_ROOT.parent / "K5-L3-Day06-DataIngestionEngineering"
sys.path.insert(0, str(REPO_ROOT / "starter"))
from baseline import TfidfIndex, tokenize  # noqa: E402

ALLOWED_FILTER_KEYS = {"category", "year", "section", "document_id", "source_id", "page", "source"}

STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "for",
    "of", "and", "or", "as", "by", "with", "from", "that", "this", "these", "those", "it",
    "its", "what", "how", "why", "when", "where", "which", "who", "whom", "does", "do", "did",
    "can", "could", "would", "should", "may", "might", "an", "a", "into", "over", "under",
    "between", "during", "while", "than", "then", "both", "each", "other", "such", "their",
    "they", "them", "we", "our", "authors", "paper", "model", "models",
}


def matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Kiểm tra một chunk có thỏa mãn toàn bộ điều kiện lọc metadata hay không."""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    for k, v in filters.items():
        if v is None:
            continue
        if k not in ALLOWED_FILTER_KEYS:
            raise ValueError(f"Unsupported filter key: {k}")
        if k in ("source_id", "document_id"):
            actual = chunk.get("source_id") or meta.get("document_id")
        elif k in ("page", "source"):
            actual = chunk.get(k)
        else:
            actual = meta.get(k)
        if actual != v:
            return False
    return True


def infer_query_filters(
    query: str, explicit_filters: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str]:
    """Suy luận metadata filter đáng tin cậy từ nội dung câu hỏi (không bao giờ đọc nhãn gold).

    Quy tắc:
    1. Giữ nguyên các trường trong `explicit_filters` nếu người dùng truyền vào.
    2. Chỉ gắn thêm `year` hoặc `category` khi câu hỏi chứa từ khóa định danh duy nhất
       cho một nhóm bài báo cụ thể mà không nhắc tới nhóm bài báo khác (tránh lọc nhầm câu hỏi so sánh liên bài báo).
    """
    inferred: dict[str, Any] = dict(explicit_filters) if explicit_filters else {}
    reasons: list[str] = []
    if explicit_filters:
        reasons.append(f"user_supplied={explicit_filters}")

    q_lower = unicodedata.normalize("NFC", query).casefold()

    # Suy luận năm xuất bản nếu câu hỏi nêu đích danh một năm duy nhất trong {2017, 2020, 2022}
    years_mentioned = [y for y in (2017, 2020, 2022) if re.search(rf"\b{y}\b", q_lower)]
    if len(years_mentioned) == 1 and "year" not in inferred:
        inferred["year"] = years_mentioned[0]
        reasons.append(f"year_in_query={years_mentioned[0]}")

    # Kiểm tra dấu hiệu đặc trưng của từng bài báo
    arch_markers = (
        "transformer", "vaswani", "multi-head", "self-attention", "positional encoding",
        "dmodel", "dff", "wmt 2014", "en-de", "en-fr", "label smoothing", "byte-pair",
    )
    rag_markers = (
        "rag-sequence", "rag-token", "lewis", "non-parametric memory", "parametric memory",
        "bart generator", "jeopardy", "ms-marco", "msmarco", "dpr", "realm",
    )
    agent_markers = (
        "react", "cot-sc", "alfworld", "webshop", "yao", "inner monologue", "hotpotqa",
        "acting-only",
    )

    has_arch = any(m in q_lower for m in arch_markers)
    has_rag = any(m in q_lower for m in rag_markers)
    has_agent = any(m in q_lower for m in agent_markers)

    # Chỉ lọc theo category khi câu hỏi thuộc đúng 1 miền duy nhất
    domain_count = int(has_arch) + int(has_rag) + int(has_agent)
    if domain_count == 1 and "category" not in inferred:
        if has_arch:
            inferred["category"] = "architecture"
            reasons.append("domain_marker=architecture(Transformer)")
        elif has_rag:
            inferred["category"] = "rag"
            reasons.append("domain_marker=rag(RAG)")
        elif has_agent:
            inferred["category"] = "agent"
            reasons.append("domain_marker=agent(ReAct)")

    if not inferred:
        return None, "no_filter"
    return inferred, "; ".join(reasons)


def tokenize_technical(text: str) -> list[str]:
    """Tách từ giữ cả từ đơn chuẩn hóa, các thuật ngữ gạch nối (rag-sequence, cot-sc) và ký hiệu kỹ thuật."""
    norm = unicodedata.normalize("NFC", text).casefold()
    # Chuẩn hóa một số biến thể dính chữ hoặc ký hiệu trong PDF
    norm = norm.replace("→", " ").replace("−", "-")
    base_tokens = [t for t in re.findall(r"\w+", norm) if t not in STOPWORDS]
    # Bắt thêm các cụm từ nối bằng dấu gạch ngang như rag-sequence, cot-sc, react-im, en-de
    hyphen_tokens = re.findall(r"\b[a-z0-9]+-[a-z0-9]+\b", norm)
    return base_tokens + hyphen_tokens


class TfidfRetriever:
    """Method 1: Lexical TF-IDF Retriever (mở rộng từ `starter/baseline.py`)."""

    def __init__(self, chunks: list[dict[str, Any]], use_filters: bool = False) -> None:
        self.chunks = chunks
        self.use_filters = use_filters
        self._index = TfidfIndex(chunks)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        if top_k < 1:
            raise ValueError("top_k phải >= 1")

        active_filters = filters if self.use_filters else None
        q_vec = self._index._vector(Counter(tokenize(query)))
        results: list[dict[str, Any]] = []
        for chunk, doc_vec in zip(self.chunks, self._index.vectors):
            if active_filters and not matches_filter(chunk, active_filters):
                continue
            score = sum(val * doc_vec.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, "score": round(float(score), 6), "strategy": "tfidf"})
        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


class BM25Retriever:
    """Method 2: Lexical Okapi BM25 Retriever với chuẩn hóa độ dài và tách thuật ngữ kỹ thuật."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        use_filters: bool = False,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.use_filters = use_filters

        self.doc_tokens = [tokenize_technical(c["text"]) for c in chunks]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0

        n_docs = len(chunks)
        df = Counter(term for counts in self.doc_freqs for term in counts)
        self.idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        if top_k < 1:
            raise ValueError("top_k phải >= 1")

        active_filters = filters if self.use_filters else None
        q_counts = Counter(tokenize_technical(query))
        results: list[dict[str, Any]] = []

        for chunk, tf_map, dl in zip(self.chunks, self.doc_freqs, self.doc_lens):
            if active_filters and not matches_filter(chunk, active_filters):
                continue
            score = 0.0
            norm_denom = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            for term, q_freq in q_counts.items():
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self.idf.get(term, 0.0)
                term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + norm_denom))
                score += term_score * (1.0 + 0.1 * (q_freq - 1))
            if score > 0:
                results.append({**chunk, "score": round(float(score), 6), "strategy": "bm25"})

        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


class DenseQdrantRetriever:
    """Method 3: Dense Vector Retrieval kết nối trực tiếp QdrantPersistentStore và EmbeddingGemma-2 từ Day 06."""

    _SHARED_EMBEDDER: Any = None
    _SHARED_CACHE: dict[str, list[float]] = {}
    _CACHE_LOADED: bool = False

    def __init__(
        self,
        qdrant_path: Path | None = None,
        cache_path: Path | None = None,
        model_name: str = "google/embeddinggemma-2",
        strategy_key: str = "B",
        dimensions: int = 768,
        use_filters: bool = True,
        auto_infer_filters: bool = False,
    ) -> None:
        self.qdrant_path = qdrant_path or (DAY06_ROOT / "results" / "qdrant_db")
        self.cache_path = cache_path or (DAY06_ROOT / "results" / "embedding_cache.json")
        self.model_name = model_name
        self.strategy_key = strategy_key
        self.dimensions = dimensions
        self.use_filters = use_filters
        self.auto_infer_filters = auto_infer_filters

        if str(DAY06_ROOT) not in sys.path:
            sys.path.insert(0, str(DAY06_ROOT))
        from starter.src.store import QdrantPersistentStore  # type: ignore

        self._store_cls = QdrantPersistentStore
        if not DenseQdrantRetriever._CACHE_LOADED and self.cache_path.exists():
            try:
                DenseQdrantRetriever._SHARED_CACHE = json.loads(
                    self.cache_path.read_text(encoding="utf-8")
                )
                DenseQdrantRetriever._CACHE_LOADED = True
            except Exception:
                DenseQdrantRetriever._SHARED_CACHE = {}

    def _get_embedder(self):
        if DenseQdrantRetriever._SHARED_EMBEDDER is None:
            import torch
            from starter.src.embeddings import MyEmbedder  # type: ignore

            embedder = MyEmbedder(
                model_name=self.model_name, batch_size=8, local_files_only=True
            )
            # Chuyển trọng số trên CPU từ bfloat16 sang float32 để tăng tốc độ mã hóa truy vấn mới gấp ~50 lần
            embedder._model = embedder._model.to(torch.float32)
            DenseQdrantRetriever._SHARED_EMBEDDER = embedder
        return DenseQdrantRetriever._SHARED_EMBEDDER

    def embed_query(self, query: str) -> list[float]:
        """Lấy vector truy vấn từ cache SHA-256 hoặc mã hóa trực tiếp bằng `google/embeddinggemma-2`."""
        q_clean = query.strip()
        cache_key = hashlib.sha256(f"{self.model_name}::{q_clean}".encode("utf-8")).hexdigest()
        if cache_key in DenseQdrantRetriever._SHARED_CACHE:
            vec = DenseQdrantRetriever._SHARED_CACHE[cache_key]
            if len(vec) == self.dimensions:
                return vec

        embedder = self._get_embedder()
        vec = embedder.embed_query(q_clean)
        DenseQdrantRetriever._SHARED_CACHE[cache_key] = vec
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(DenseQdrantRetriever._SHARED_CACHE), encoding="utf-8"
            )
        except Exception:
            pass
        return vec

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        if top_k < 1:
            raise ValueError("top_k phải >= 1")

        active_filters = filters if self.use_filters else None
        if self.use_filters and self.auto_infer_filters:
            active_filters, _ = infer_query_filters(query, active_filters)

        # Chuyển đổi khóa source_id sang document_id cho tương thích với Qdrant schema của Day 06
        qdrant_filters: dict[str, Any] | None = None
        if active_filters:
            qdrant_filters = {}
            for k, v in active_filters.items():
                if v is None:
                    continue
                qdrant_filters["document_id" if k == "source_id" else k] = v

        q_vec = self.embed_query(query)
        store = self._store_cls(storage_path=self.qdrant_path, dimensions=self.dimensions)
        try:
            raw_hits = store.search(
                strategy=self.strategy_key,
                query_vector=q_vec,
                top_k=top_k,
                filters=qdrant_filters,
            )
        finally:
            try:
                store.close()
            except Exception:
                pass

        results: list[dict[str, Any]] = []
        for h in raw_hits:
            results.append(
                {
                    "id": h["chunk_id"],
                    "source_id": h["document_id"],
                    "source": h["source"],
                    "page": int(h["page"]),
                    "text": h["text"],
                    "metadata": h.get("metadata", {}),
                    "score": round(float(h["score"]), 6),
                    "strategy": "dense",
                }
            )
        return results[:top_k]


class HybridRRFRetriever:
    """Method 4: Hybrid Retrieval kết hợp Dense (Qdrant + Gemma-2) và Lexical (BM25 + TF-IDF)
    bằng Reciprocal Rank Fusion (RRF) + chuẩn hóa điểm Min-Max + suy luận Metadata Filter an toàn.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        dense_retriever: DenseQdrantRetriever,
        bm25_retriever: BM25Retriever | None = None,
        tfidf_retriever: TfidfRetriever | None = None,
        rrf_k: int = 5,
        candidate_pool: int = 25,
        dense_weight: float = 0.65,
        bm25_weight: float = 0.25,
        tfidf_weight: float = 0.10,
        use_filters: bool = True,
        auto_infer_filters: bool = True,
    ) -> None:
        self.chunks = chunks
        self.dense = dense_retriever
        self.bm25 = bm25_retriever or BM25Retriever(chunks, use_filters=True)
        self.tfidf = tfidf_retriever or TfidfRetriever(chunks, use_filters=True)
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self.use_filters = use_filters
        self.auto_infer_filters = auto_infer_filters
        self.last_filter_info: dict[str, Any] = {}

    @staticmethod
    def _minmax_normalize(items: list[dict[str, Any]]) -> dict[str, float]:
        if not items:
            return {}
        scores = [float(x["score"]) for x in items]
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return {x["id"]: 1.0 for x in items}
        return {x["id"]: (float(x["score"]) - lo) / (hi - lo) for x in items}

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("search query phải là chuỗi không rỗng")
        if top_k < 1:
            raise ValueError("top_k phải >= 1")

        active_filters = filters if self.use_filters else None
        reason = "user_supplied" if active_filters else "no_filter"
        if self.use_filters and self.auto_infer_filters:
            active_filters, reason = infer_query_filters(query, active_filters)
        self.last_filter_info = {"effective_filters": active_filters, "reason": reason}

        pool_k = max(top_k * 4, self.candidate_pool)
        dense_hits = self.dense.search(query, filters=active_filters, top_k=pool_k)
        bm25_hits = self.bm25.search(query, filters=active_filters, top_k=pool_k)
        tfidf_hits = self.tfidf.search(query, filters=active_filters, top_k=pool_k)

        # Nếu bộ lọc suy luận quá chặt làm rỗng kết quả, tự động nới lỏng về không lọc
        if not dense_hits and not bm25_hits and active_filters:
            dense_hits = self.dense.search(query, filters=None, top_k=pool_k)
            bm25_hits = self.bm25.search(query, filters=None, top_k=pool_k)
            tfidf_hits = self.tfidf.search(query, filters=None, top_k=pool_k)
            self.last_filter_info["fallback_unfiltered"] = True

        dense_rank = {c["id"]: idx + 1 for idx, c in enumerate(dense_hits)}
        bm25_rank = {c["id"]: idx + 1 for idx, c in enumerate(bm25_hits)}
        tfidf_rank = {c["id"]: idx + 1 for idx, c in enumerate(tfidf_hits)}

        dense_norm = self._minmax_normalize(dense_hits)
        bm25_norm = self._minmax_normalize(bm25_hits)
        tfidf_norm = self._minmax_normalize(tfidf_hits)

        merged_chunks: dict[str, dict[str, Any]] = {}
        for lst in (dense_hits, bm25_hits, tfidf_hits):
            for c in lst:
                if c["id"] not in merged_chunks:
                    merged_chunks[c["id"]] = dict(c)

        scored: list[dict[str, Any]] = []
        for cid, chunk_obj in merged_chunks.items():
            r_d = dense_rank.get(cid, pool_k + 50)
            r_b = bm25_rank.get(cid, pool_k + 50)
            r_t = tfidf_rank.get(cid, pool_k + 50)

            rrf_score = (
                self.dense_weight / (self.rrf_k + r_d)
                + self.bm25_weight / (self.rrf_k + r_b)
                + self.tfidf_weight / (self.rrf_k + r_t)
            )
            norm_fusion = (
                self.dense_weight * dense_norm.get(cid, 0.0)
                + self.bm25_weight * bm25_norm.get(cid, 0.0)
                + self.tfidf_weight * tfidf_norm.get(cid, 0.0)
            )
            # Kết hợp RRF (kháng nhiễu ngoại lai của điểm BM25) và Score Fusion chuẩn hóa Min-Max
            final_score = 0.85 * (rrf_score * self.rrf_k) + 0.15 * norm_fusion

            enriched_meta = {
                **chunk_obj.get("metadata", {}),
                "dense_rank": dense_rank.get(cid),
                "bm25_rank": bm25_rank.get(cid),
                "tfidf_rank": tfidf_rank.get(cid),
                "dense_score": next((x["score"] for x in dense_hits if x["id"] == cid), None),
                "bm25_score": next((x["score"] for x in bm25_hits if x["id"] == cid), None),
            }
            scored.append(
                {
                    **chunk_obj,
                    "metadata": enriched_meta,
                    "score": round(float(final_score), 6),
                    "strategy": "hybrid_rrf",
                }
            )

        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return scored[:top_k]
