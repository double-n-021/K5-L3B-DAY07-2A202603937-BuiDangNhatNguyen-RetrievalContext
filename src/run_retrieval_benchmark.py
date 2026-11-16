"""Chạy đánh giá và so sánh 4 phương pháp Retrieval cùng các cấu hình Context Engineering cho Day 07.

Chỉ chạy trên split được yêu cầu (mặc định `--split dev`). Tuyệt đối không dùng tập `test` để chọn cấu hình.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
sys.path.insert(0, str(REPO_ROOT))

from baseline import hit, validate  # noqa: E402
from src.context_engineering import pack_context_smart  # noqa: E402
from src.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseQdrantRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
    infer_query_filters,
)


def check_complete_context(context: str, question: dict[str, Any]) -> bool:
    """Kiểm tra xem ngữ cảnh đã đóng gói có chứa đầy đủ tất cả các phần bằng chứng bắt buộc hay không."""
    req_parts = question.get("required_evidence_parts") or [g["evidence"] for g in question["gold"]]
    return all(part in context for part in req_parts)


def evaluate_configuration(
    config_name: str,
    description: str,
    retriever: Any,
    chunks_by_id: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    use_user_filter: bool = False,
    auto_infer_filter: bool = False,
    enable_neighbor: bool = False,
    index_seconds: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Đánh giá một cấu hình Retrieval + Context trên danh sách câu hỏi và trả về summary + traces."""
    traces: list[dict[str, Any]] = []

    for q in questions:
        t0 = time.perf_counter()
        user_filter = q.get("filter") if use_user_filter else None
        effective_filter, filter_reason = (
            infer_query_filters(q["question"], user_filter)
            if auto_infer_filter
            else (user_filter, "user_supplied" if user_filter else "no_filter")
        )

        # Lấy kết quả truy xuất
        raw_results = retriever.search(q["question"], filters=user_filter, top_k=top_k)
        # Luôn giữ lát cắt top-3 để tính Hit@1 và Evidence Hit@3 công bằng theo đúng BENCHMARK.md
        top3_results = raw_results[:3]

        context, context_ids = pack_context_smart(
            raw_results,
            budget=budget,
            all_chunks_by_id=chunks_by_id,
            enable_neighbor_expansion=enable_neighbor,
            neighbor_anchors=2,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        marked_top3 = [{**c, "hit": hit(c, q["gold"])} for c in top3_results]
        marked_all = [{**c, "hit": hit(c, q["gold"])} for c in raw_results]

        hit1 = bool(marked_top3 and marked_top3[0]["hit"])
        hit3 = any(c["hit"] for c in marked_top3)
        rr3 = 0.0
        for idx, c in enumerate(marked_top3):
            if c["hit"]:
                rr3 = 1.0 / (idx + 1)
                break

        # Kiểm tra bằng chứng trong context cuối cùng gửi cho model
        context_chunks = [chunks_by_id[cid] for cid in context_ids if cid in chunks_by_id]
        context_hit = any(hit(c, q["gold"]) for c in context_chunks)
        complete_hit = context_hit and check_complete_context(context, q)

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "type": q.get("type", "unknown"),
                "question": q["question"],
                "config_name": config_name,
                "strategy": getattr(retriever, "__class__", type(retriever)).__name__,
                "top_k": top_k,
                "budget_chars": budget,
                "neighbor_expansion": enable_neighbor,
                "user_filter": user_filter,
                "effective_filter": effective_filter,
                "filter_reason": filter_reason,
                "gold": q["gold"],
                "required_evidence_parts": q.get("required_evidence_parts", []),
                "results": marked_all,
                "context": context,
                "context_ids": context_ids,
                "context_chars": len(context),
                "context_tokens_est": max(1, round(len(context) / 4)) if context else 0,
                "elapsed_ms": round(elapsed_ms, 3),
                "hit1": hit1,
                "hit3": hit3,
                "rr3": round(rr3, 4),
                "context_hit": context_hit,
                "complete_context_hit": complete_hit,
            }
        )

    n = len(traces)
    summary = {
        "config_name": config_name,
        "description": description,
        "split": questions[0]["split"] if questions else "unknown",
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "neighbor_expansion": enable_neighbor,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4) if n else 0.0,
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4) if n else 0.0,
        "mrr3": round(statistics.mean(t["rr3"] for t in traces), 4) if n else 0.0,
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4) if n else 0.0,
        "complete_context_hit": sum(t["complete_context_hit"] for t in traces),
        "complete_context_hit_rate": round(
            sum(t["complete_context_hit"] for t in traces) / n, 4
        )
        if n
        else 0.0,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1)
        if n
        else 0.0,
        "avg_context_tokens_est": round(
            statistics.mean(t["context_tokens_est"] for t in traces), 1
        )
        if n
        else 0.0,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 3) if n else 0.0,
        "mean_ms": round(statistics.mean(t["elapsed_ms"] for t in traces), 3) if n else 0.0,
        "index_seconds": round(index_seconds, 4),
        "service_cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy so sánh 4 phương pháp Retrieval & Context Engineering"
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark" / "chunks.json",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark" / "questions.json",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "benchmark_results",
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected_questions = [q for q in all_questions if q.get("split") == args.split]
    validate(chunks, selected_questions)
    chunks_by_id = {c["id"]: c for c in chunks}

    # Khởi tạo 4 phương pháp Retrieval
    t0 = time.perf_counter()
    tfidf_unfiltered = TfidfRetriever(chunks, use_filters=False)
    t_tfidf = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25_unfiltered = BM25Retriever(chunks, k1=1.5, b=0.75, use_filters=False)
    bm25_filtered = BM25Retriever(chunks, k1=1.5, b=0.75, use_filters=True)
    t_bm25 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_unfiltered = DenseQdrantRetriever(use_filters=False, auto_infer_filters=False)
    dense_filtered = DenseQdrantRetriever(use_filters=True, auto_infer_filters=True)
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    tfidf_filtered = TfidfRetriever(chunks, use_filters=True)
    hybrid_rrf = HybridRRFRetriever(
        chunks=chunks,
        dense_retriever=dense_filtered,
        bm25_retriever=bm25_filtered,
        tfidf_retriever=tfidf_filtered,
        use_filters=True,
        auto_infer_filters=True,
    )
    t_hybrid = time.perf_counter() - t0

    # Định nghĩa các cấu hình thực nghiệm (4 phương pháp ở cùng budget=800 + 2 cấu hình cải thiện Context)
    configs = [
        {
            "name": "M1_Baseline_TFIDF_b800",
            "desc": "Method 1 (Baseline): Lexical TF-IDF, top_k=3, no filter, budget=800 chars",
            "retriever": tfidf_unfiltered,
            "top_k": 3,
            "budget": 800,
            "use_user_filter": False,
            "auto_infer_filter": False,
            "enable_neighbor": False,
            "idx_sec": t_tfidf,
        },
        {
            "name": "M2_Lexical_BM25_b800",
            "desc": "Method 2: Lexical Okapi BM25 (k1=1.5, b=0.75, technical tokenizer), top_k=3, no filter, budget=800 chars",
            "retriever": bm25_unfiltered,
            "top_k": 3,
            "budget": 800,
            "use_user_filter": False,
            "auto_infer_filter": False,
            "enable_neighbor": False,
            "idx_sec": t_bm25,
        },
        {
            "name": "M3_Dense_Gemma2_Qdrant_b800",
            "desc": "Method 3: Dense Qdrant + EmbeddingGemma-2 (768d), top_k=3, user filter, budget=800 chars",
            "retriever": DenseQdrantRetriever(use_filters=True, auto_infer_filters=False),
            "top_k": 3,
            "budget": 800,
            "use_user_filter": True,
            "auto_infer_filter": False,
            "enable_neighbor": False,
            "idx_sec": t_dense,
        },
        {
            "name": "M4_Hybrid_RRF_SafeFilter_b800",
            "desc": "Method 4: Hybrid RRF (Dense + BM25 + TF-IDF) + Safe Query Filter Inference, top_k=3, budget=800 chars",
            "retriever": hybrid_rrf,
            "top_k": 3,
            "budget": 800,
            "use_user_filter": True,
            "auto_infer_filter": True,
            "enable_neighbor": False,
            "idx_sec": t_hybrid,
        },
        {
            "name": "Exp1_Hybrid_RRF_Budget1800",
            "desc": "Improvement Config 1: Hybrid RRF + Safe Filter + Expanded Context Budget (budget=1800 chars, top_k=3)",
            "retriever": hybrid_rrf,
            "top_k": 3,
            "budget": 1800,
            "use_user_filter": True,
            "auto_infer_filter": True,
            "enable_neighbor": False,
            "idx_sec": t_hybrid,
        },
        {
            "name": "Exp2_Hybrid_RRF_Neighbor_Budget2400",
            "desc": "Improvement Config 2: Hybrid RRF + Safe Filter + Neighboring Chunk Expansion + Budget 2400 chars",
            "retriever": hybrid_rrf,
            "top_k": 3,
            "budget": 2400,
            "use_user_filter": True,
            "auto_infer_filter": True,
            "enable_neighbor": True,
            "idx_sec": t_hybrid,
        },
    ]

    all_summaries: list[dict[str, Any]] = []
    split_dir = args.output / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        summary, traces = evaluate_configuration(
            config_name=cfg["name"],
            description=cfg["desc"],
            retriever=cfg["retriever"],
            chunks_by_id=chunks_by_id,
            questions=selected_questions,
            top_k=cfg["top_k"],
            budget=cfg["budget"],
            use_user_filter=cfg["use_user_filter"],
            auto_infer_filter=cfg["auto_infer_filter"],
            enable_neighbor=cfg["enable_neighbor"],
            index_seconds=cfg["idx_sec"],
        )
        all_summaries.append(summary)
        cfg_dir = split_dir / cfg["name"]
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (cfg_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    comparison_path = split_dir / "comparison_summary.json"
    comparison_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"=== RETRIEVAL & CONTEXT BENCHMARK RESULTS (split={args.split}, N={len(selected_questions)}) ===")
    for s in all_summaries:
        print(
            f"{s['config_name']:36s} | Hit@1: {s['hit1']:2d}/{s['n']} ({s['hit1_rate']*100:5.1f}%) "
            f"| Hit@3: {s['hit3']:2d}/{s['n']} ({s['hit3_rate']*100:5.1f}%) "
            f"| CtxHit: {s['context_hit']:2d}/{s['n']} ({s['context_hit_rate']*100:5.1f}%) "
            f"| FullCtx: {s['complete_context_hit']:2d}/{s['n']} ({s['complete_context_hit_rate']*100:5.1f}%) "
            f"| Chars: {s['avg_context_chars']:6.1f} | MedMs: {s['median_ms']:6.2f}"
        )


if __name__ == "__main__":
    main()
