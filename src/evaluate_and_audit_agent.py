"""Công cụ hỗ trợ đối chiếu (audit) từng trace của Agent và tổng hợp kết quả bằng starter/agent_metrics.py.

Đối với mỗi trace:
1. Kiểm tra `final_context` và `final_context_ids` so với `gold` và `required_evidence_parts` để chấm
   `context_completeness_grade` (0, 1, 2).
2. Đối chiếu từng `citation` trong `trace["citations"]` với nội dung chunk thực tế và `gold` để gán
   nhãn boolean `citation_support`.
3. Đối chiếu nội dung `trace["answer"]` với `expected_answer`, `required_evidence_parts` và `final_context`
   để gán `answer_grade` (0, 1, 2) cho câu có đáp án, hoặc `abstention_correct` (True/False) cho câu
   thiếu bằng chứng (`answerable=False`), kèm `audit_notes` chi tiết cho từng câu.
4. Gọi `starter.agent_metrics.aggregate` để xuất `agent_summary.json` đúng chuẩn repo.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
sys.path.insert(0, str(REPO_ROOT))

from agent_metrics import aggregate  # noqa: E402
from baseline import hit  # noqa: E402


def chunk_matches_gold(chunk: dict[str, Any], gold_item: dict[str, Any]) -> bool:
    return bool(hit(chunk, [gold_item]))


def _normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản về chữ thường, chuẩn hóa dấu gạch nối/khoảng trắng Unicode và đơn vị số học."""
    s = (text or "").lower()
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = re.sub(r"[\u00a0\u202f\u2009]", " ", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*million\b", r"\1m \1 million", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*billion\b", r"\1b \1 billion", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_key_tokens(phrase: str) -> list[str]:
    """Tách các từ khóa/số liệu quan trọng từ một mệnh đề bằng chứng bắt buộc."""
    norm = _normalize_text(phrase)
    # Giữ các token chữ/số có độ dài >= 2 hoặc các con số
    tokens = re.findall(r"[0-9]+(?:\.[0-9]+)?|[a-z0-9_+-]{2,}", norm)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "from",
        "into",
        "used",
        "using",
        "are",
        "was",
        "were",
        "has",
        "have",
        "its",
        "per",
        "all",
        "both",
        "each",
    }
    return [t for t in tokens if t not in stop]


def _coverage_ratio(reference_text: str, candidate_text: str) -> float:
    """Tính tỷ lệ từ khóa/số liệu quan trọng của `reference_text` xuất hiện trong `candidate_text`."""
    norm_cand = _normalize_text(candidate_text)
    key_tokens = _extract_key_tokens(reference_text)
    if not key_tokens:
        return 1.0
    matched = sum(1 for tok in key_tokens if tok in norm_cand)
    return matched / len(key_tokens)


def audit_single_trace(
    case: dict[str, Any],
    trace: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Đối chiếu một trace cụ thể và sinh bản ghi review kèm giải thích chi tiết."""
    cid = case["id"]
    answerable = bool(case["answerable"])
    gold_items = list(case.get("gold") or [])
    expected_answer = str(case.get("expected_answer") or "")
    gold_ids = {
        chunk_id
        for chunk_id, chunk_obj in chunks_by_id.items()
        if any(chunk_matches_gold(chunk_obj, g) for g in gold_items)
    }
    status = trace.get("status", "abstain")
    answer = trace.get("answer", "") or ""
    citations = list(trace.get("citations") or [])
    final_ctx_ids = set(trace.get("final_context_ids") or [])
    final_ctx_chunks = [chunks_by_id[x] for x in final_ctx_ids if x in chunks_by_id]

    # 1. Chấm Context Completeness (0, 1, 2) cho câu answerable
    if answerable and gold_items:
        matched_gold_count = sum(
            1
            for g in gold_items
            if any(chunk_matches_gold(c, g) for c in final_ctx_chunks)
        )
        if matched_gold_count == len(gold_items):
            ctx_grade = 2
        elif matched_gold_count > 0:
            ctx_grade = 1
        else:
            ctx_grade = 0
    else:
        matched_gold_count = 0
        ctx_grade = 0

    # 2. Kiểm tra từng citation trong trace["citations"]
    citation_support: list[bool] = []
    for cited_id in citations:
        if not answerable:
            citation_support.append(False)
            continue
        if cited_id in gold_ids:
            citation_support.append(True)
            continue
        chunk_obj = chunks_by_id.get(cited_id)
        if chunk_obj and cited_id in final_ctx_ids:
            ctext = chunk_obj.get("text", "")
            # Nếu chunk cùng source & page với một gold item và bao phủ >= 45% từ khóa của expected_answer
            same_page_gold = any(
                chunk_obj.get("source_id") == g.get("source_id")
                and int(chunk_obj.get("page", -1)) == int(g.get("page", -2))
                for g in gold_items
            )
            if same_page_gold and _coverage_ratio(expected_answer, ctext) >= 0.45:
                citation_support.append(True)
            else:
                citation_support.append(False)
        else:
            citation_support.append(False)

    # 3. Chấm answer_grade (cho câu answerable) hoặc abstention_correct (cho câu unanswerable)
    if answerable:
        if status != "answered":
            answer_grade = 0
            note = (
                f"False abstention: Câu hỏi có đáp án ({sorted(gold_ids)}), nhưng Agent trả về "
                f"status={status}. Context khớp {matched_gold_count}/{len(gold_items)} gold items."
            )
        elif not citations or not any(citation_support):
            answer_grade = 0
            note = (
                f"Answered nhưng không có citation hợp lệ hỗ trợ bằng chứng gốc "
                f"(citations={citations}, support={citation_support}, gold={sorted(gold_ids)})."
            )
        else:
            ans_cov = _coverage_ratio(expected_answer, answer)
            cited_chunks = [chunks_by_id[x] for x in citations if x in chunks_by_id]
            cited_gold_count = sum(
                1
                for g in gold_items
                if any(chunk_matches_gold(c, g) for c in cited_chunks)
            )
            if ans_cov >= 0.50 and cited_gold_count == len(gold_items) and ctx_grade == 2:
                answer_grade = 2
                note = (
                    f"Đầy đủ và có căn cứ (Grade 2): Bao phủ {ans_cov:.0%} từ khóa đáp án chuẩn, "
                    f"khớp đủ {cited_gold_count}/{len(gold_items)} bằng chứng đích, trích dẫn {citations}."
                )
            elif ans_cov >= 0.30 and any(citation_support):
                answer_grade = 1
                note = (
                    f"Đúng một phần (Grade 1): Có trích dẫn hợp lệ ({citations}), độ phủ từ khóa {ans_cov:.0%}, "
                    f"khớp {cited_gold_count}/{len(gold_items)} đoạn bằng chứng đích."
                )
            else:
                answer_grade = 0
                note = f"Câu trả lời thiếu thông tin cốt lõi ({ans_cov:.0%}) dù có trích dẫn {citations}."

        return {
            "query_id": cid,
            "answer_grade": answer_grade,
            "citation_support": citation_support,
            "context_completeness_grade": ctx_grade,
            "audit_notes": note,
        }
    else:
        abstention_correct = status == "abstain" and len(citations) == 0
        if abstention_correct:
            note = (
                f"True abstention: Từ chối trả lời chính xác trước câu hỏi thiếu bằng chứng/bẫy tiền đề "
                f"({case.get('type')})."
            )
        else:
            note = (
                f"Hallucination / False answer trên câu unanswerable: status={status}, "
                f"citations={citations}."
            )
        return {
            "query_id": cid,
            "abstention_correct": abstention_correct,
            "citation_support": citation_support,
            "context_completeness_grade": 0,
            "audit_notes": note,
        }


def evaluate_config_dir(
    config_dir: Path,
    cases: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Đọc traces.json trong config_dir, tạo reviews.json và chạy aggregate()."""
    traces_path = config_dir / "traces.json"
    if not traces_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {traces_path}")

    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    trace_by_id = {t["query_id"]: t for t in traces}

    reviews: list[dict[str, Any]] = []
    for case in cases:
        t = trace_by_id[case["id"]]
        reviews.append(audit_single_trace(case, t, chunks_by_id))

    reviews_path = config_dir / "reviews.json"
    reviews_path.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Gọi đúng hàm aggregate() của starter/agent_metrics.py
    official_metrics = aggregate(cases, traces, reviews)

    # Tính thêm Context Completeness (0..2) trên các câu answerable
    ans_reviews = [r for r in reviews if "answer_grade" in r]
    ctx_counts = {
        "0": sum(1 for r in ans_reviews if r["context_completeness_grade"] == 0),
        "1": sum(1 for r in ans_reviews if r["context_completeness_grade"] == 1),
        "2": sum(1 for r in ans_reviews if r["context_completeness_grade"] == 2),
        "N": len(ans_reviews),
    }
    ctx_counts["mean"] = (
        round((ctx_counts["1"] + 2 * ctx_counts["2"]) / ctx_counts["N"], 4)
        if ctx_counts["N"]
        else None
    )

    telemetry_path = config_dir / "telemetry.json"
    telemetry = (
        json.loads(telemetry_path.read_text(encoding="utf-8"))
        if telemetry_path.exists()
        else {}
    )

    full_report = {
        "split": split,
        "config_name": config_dir.name,
        **official_metrics,
        "context_completeness": ctx_counts,
        "telemetry": telemetry,
    }

    summary_path = config_dir / "agent_summary.json"
    summary_path.write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit traces của Agent và tổng hợp bảng điểm chuẩn"
    )
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
        "--results-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "benchmark_results",
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    split_cases = [c for c in all_cases if c.get("split") == args.split]

    split_dir = args.results_root / args.split
    reports: list[dict[str, Any]] = []
    for cfg_dir in sorted(split_dir.glob("Agent_*")):
        if (cfg_dir / "traces.json").exists():
            rep = evaluate_config_dir(cfg_dir, split_cases, chunks_by_id, args.split)
            reports.append(rep)

    out_cmp = split_dir / "agent_evaluation_comparison.json"
    out_cmp.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
