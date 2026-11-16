"""Thực thi đánh giá Agentic RAG thật (`gemini-2.5-flash`) trên tập Dev và Test cho Day 07.

Lưu đầy đủ `traces.json` theo đúng hợp đồng của `starter/agent_metrics.py`, kèm thống kê
số lượt gọi tool, độ trễ, token usage và chi phí ước tính.
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

from agentic import run_agent  # noqa: E402
from src.agent_runner import (  # noqa: E402
    GeminiAgentController,
    make_interleaved_context_packer,
)
from src.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseQdrantRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)


def run_agent_config(
    config_name: str,
    description: str,
    cases: list[dict[str, Any]],
    controller: GeminiAgentController,
    retriever: Any,
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    max_searches: int = 2,
    use_user_filter: bool = False,
    context_packer: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Chạy Agent thật trên danh sách cases và trả về (telemetry_summary, traces)."""
    traces: list[dict[str, Any]] = []
    controller.max_searches = max_searches

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        question = case["question"]
        user_filter = case.get("filter") if use_user_filter else None
        controller.reset_telemetry()
        t0 = time.perf_counter()

        try:
            res = run_agent(
                question,
                controller,
                retriever,
                filters=user_filter,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
                context_packer=context_packer,
                all_chunks_by_id=chunks_by_id,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            search_events = [e for e in res["events"] if e.get("event") == "search"]
            final_ctx = search_events[-1]["context"] if search_events else ""
            final_ctx_ids = search_events[-1]["context_ids"] if search_events else []

            trace_item = {
                "query_id": cid,
                "split": case["split"],
                "answerable": case["answerable"],
                "type": case.get("type", "unknown"),
                "question": question,
                "config_name": config_name,
                "model": controller.model,
                "provider": controller.provider,
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "search_calls": len(search_events),
                "llm_calls": controller.llm_calls,
                "prompt_tokens": controller.prompt_tokens,
                "completion_tokens": controller.completion_tokens,
                "total_tokens": controller.total_tokens,
                "estimated_cost_usd": round(controller.estimated_cost_usd, 6),
                "elapsed_ms": round(elapsed_ms, 2),
                "final_context": final_ctx,
                "final_context_ids": final_ctx_ids,
                "final_context_chars": len(final_ctx),
                "events": res["events"],
                "step_logs": controller.step_logs,
            }
        except Exception as exc:
            if "11001" in str(exc) or "getaddrinfo failed" in str(exc):
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            trace_item = {
                "query_id": cid,
                "split": case["split"],
                "answerable": case["answerable"],
                "type": case.get("type", "unknown"),
                "question": question,
                "config_name": config_name,
                "model": controller.model,
                "provider": controller.provider,
                "status": "error",
                "error_message": str(exc),
                "answer": "",
                "citations": [],
                "search_calls": 0,
                "llm_calls": controller.llm_calls,
                "prompt_tokens": controller.prompt_tokens,
                "completion_tokens": controller.completion_tokens,
                "total_tokens": controller.total_tokens,
                "estimated_cost_usd": round(controller.estimated_cost_usd, 6),
                "elapsed_ms": round(elapsed_ms, 2),
                "final_context": "",
                "final_context_ids": [],
                "final_context_chars": 0,
                "events": [],
                "step_logs": controller.step_logs,
            }

        traces.append(trace_item)
        print(
            f"  [{config_name}] ({idx:02d}/{len(cases):02d}) {cid} (ans={case['answerable']}): "
            f"status={trace_item['status']} | searches={trace_item['search_calls']} "
            f"| cits={trace_item['citations']} | tokens={trace_item['total_tokens']} "
            f"| {trace_item['elapsed_ms']:.0f}ms",
            flush=True,
        )

    valid_traces = [t for t in traces if t["status"] != "error"]
    search_dist: dict[str, int] = {}
    for t in valid_traces:
        k = str(t["search_calls"])
        search_dist[k] = search_dist.get(k, 0) + 1

    telemetry = {
        "config_name": config_name,
        "description": description,
        "split": cases[0]["split"] if cases else "unknown",
        "model": controller.model,
        "provider": controller.provider,
        "total_cases": len(cases),
        "provider_errors": sum(t["status"] == "error" for t in traces),
        "answered_count": sum(t["status"] == "answered" for t in traces),
        "abstain_count": sum(t["status"] == "abstain" for t in traces),
        "search_calls_distribution": search_dist,
        "avg_search_calls": round(
            statistics.mean(t["search_calls"] for t in valid_traces), 2
        )
        if valid_traces
        else 0.0,
        "avg_llm_calls": round(statistics.mean(t["llm_calls"] for t in valid_traces), 2)
        if valid_traces
        else 0.0,
        "avg_context_chars": round(
            statistics.mean(t["final_context_chars"] for t in valid_traces), 1
        )
        if valid_traces
        else 0.0,
        "total_tokens": sum(t["total_tokens"] for t in traces),
        "avg_tokens_per_case": round(
            statistics.mean(t["total_tokens"] for t in valid_traces), 1
        )
        if valid_traces
        else 0.0,
        "total_cost_usd": round(sum(t["estimated_cost_usd"] for t in traces), 6),
        "median_latency_ms": round(
            statistics.median(t["elapsed_ms"] for t in valid_traces), 1
        )
        if valid_traces
        else 0.0,
        "mean_latency_ms": round(statistics.mean(t["elapsed_ms"] for t in valid_traces), 1)
        if valid_traces
        else 0.0,
    }
    return telemetry, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy đánh giá Agentic RAG trên Dev/Test")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark" / "chunks.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "data" / "benchmark" / "cases.json",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "exp1", "exp2"],
        choices=("baseline", "exp1", "exp2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "benchmark_results",
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    selected_cases = [c for c in all_cases if c.get("split") == args.split]
    chunks_by_id = {c["id"]: c for c in chunks}

    # Tiền mã hóa (pre-warm) toàn bộ câu hỏi gốc trong split bằng EmbeddingGemma-2 để không tốn thời gian chờ ở Turn 1
    dense_ret = DenseQdrantRetriever(use_filters=True, auto_infer_filters=True)
    for c in selected_cases:
        dense_ret.embed_query(c["question"])

    tfidf_unfiltered = TfidfRetriever(chunks, use_filters=False)
    bm25_filtered = BM25Retriever(chunks, k1=1.5, b=0.75, use_filters=True)
    tfidf_filtered = TfidfRetriever(chunks, use_filters=True)
    hybrid_rrf = HybridRRFRetriever(
        chunks=chunks,
        dense_retriever=dense_ret,
        bm25_retriever=bm25_filtered,
        tfidf_retriever=tfidf_filtered,
        use_filters=True,
        auto_infer_filters=True,
    )

    controller = GeminiAgentController()

    config_defs = {
        "baseline": {
            "name": "Agent_Baseline_TFIDF_b800",
            "desc": "Baseline Agent: TF-IDF retriever, top_k=3, FIFO pack_context, budget=800 chars, max_searches=2",
            "retriever": tfidf_unfiltered,
            "top_k": 3,
            "budget": 800,
            "max_searches": 2,
            "use_user_filter": False,
            "packer": None,
        },
        "exp1": {
            "name": "Agent_Exp1_Hybrid_Budget1800",
            "desc": "Experiment 1 Agent: Hybrid RRF + Safe Filter + Interleaved Context Packer, budget=1800 chars, max_searches=2",
            "retriever": hybrid_rrf,
            "top_k": 3,
            "budget": 1800,
            "max_searches": 2,
            "use_user_filter": True,
            "packer": make_interleaved_context_packer(
                chunks_by_id, enable_neighbor_expansion=False
            ),
        },
        "exp2": {
            "name": "Agent_Exp2_Hybrid_Neighbor_Budget2400",
            "desc": "Experiment 2 Agent (Selected): Hybrid RRF + Safe Filter + Two-Tier Neighbor Expansion + Interleaved Packer, budget=2400 chars, max_searches=2",
            "retriever": hybrid_rrf,
            "top_k": 3,
            "budget": 2400,
            "max_searches": 2,
            "use_user_filter": True,
            "packer": make_interleaved_context_packer(
                chunks_by_id, enable_neighbor_expansion=True, neighbor_anchors=2
            ),
        },
    }

    split_dir = args.output / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for key in args.configs:
        cfg = config_defs[key]
        print(f"\n>>> Running {cfg['name']} on split={args.split} ({len(selected_cases)} cases) <<<")
        telemetry, traces = run_agent_config(
            config_name=cfg["name"],
            description=cfg["desc"],
            cases=selected_cases,
            controller=controller,
            retriever=cfg["retriever"],
            chunks_by_id=chunks_by_id,
            top_k=cfg["top_k"],
            budget=cfg["budget"],
            max_searches=cfg["max_searches"],
            use_user_filter=cfg["use_user_filter"],
            context_packer=cfg["packer"],
        )
        summaries.append(telemetry)
        out_dir = split_dir / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "telemetry.json").write_text(
            json.dumps(telemetry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (split_dir / "agent_telemetry_comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== AGENT TELEMETRY SUMMARY ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
