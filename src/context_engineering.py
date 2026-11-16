"""Context Engineering Module cho Day 07:
- Khử trùng lặp (Deduplication) theo chunk ID.
- Sắp xếp lại ứng viên qua nhiều lượt tìm kiếm (Relevance & Recency Re-ordering).
- Mở rộng đoạn văn lân cận trên cùng trang (Neighboring Chunk Expansion) để cứu các câu bị cắt bằng chứng.
- Đóng gói ngữ cảnh trong ngân sách giới hạn (Context Budget Management) với đầy đủ nhãn nguồn `[id | source | page]`.
"""
import re
from typing import Any


def format_chunk_block(chunk: dict[str, Any]) -> str:
    """Định dạng chuẩn một khối ngữ cảnh có gắn mã trích dẫn, nguồn và số trang."""
    return f"[{chunk['id']} | {chunk['source']} | page {chunk['page']}]\n{chunk['text']}"


def parse_chunk_index(chunk_id: str) -> tuple[str, int] | None:
    """Tách tiền tố trang và chỉ số đoạn từ chunk_id dạng `{doc}_p{page}_para_{idx}`."""
    m = re.match(r"^(.+_p\d+_(?:para|par|fix)_?)(\d+)$", chunk_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def expand_neighboring_chunks(
    results: list[dict[str, Any]],
    all_chunks_by_id: dict[str, dict[str, Any]],
    *,
    max_anchors_to_expand: int = 2,
    window: int = 1,
) -> list[dict[str, Any]]:
    """Giữ toàn bộ các ứng viên chính (Primary Top-K) ở đầu danh sách ưu tiên, sau đó nối thêm
    các chunk liền kề cùng trang (`idx + 1`, `idx - 1`) của các chunk hạng cao nhất.

    Thiết kế ưu tiên 2 tầng (Two-Tier Priority) này đảm bảo:
    1. Các ứng viên chính ở vị trí Rank 2, Rank 3 không bao giờ bị các chunk lân cận của Rank 1
       chiếm mất chỗ trong Context Budget.
    2. Các câu hỏi bị cắt bằng chứng qua 2 đoạn văn liên tiếp (`q12`, `q18`, `q22`, `q26`) vẫn
       được bổ sung đầy đủ đoạn liền kề ngay khi còn ngân sách.
    """
    primary: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in results:
        cid = chunk["id"]
        if cid not in seen:
            primary.append(chunk)
            seen.add(cid)

    neighbors: list[dict[str, Any]] = []
    for rank_idx, chunk in enumerate(primary[:max_anchors_to_expand]):
        cid = chunk["id"]
        parsed = parse_chunk_index(cid)
        if not parsed:
            continue
        prefix, base_idx = parsed
        # Ưu tiên lấy đoạn liền sau (base_idx + 1) trước vì phần tiếp theo của mục A/B hoặc chú thích nằm ngay dưới
        for delta in range(1, window + 1):
            for neighbor_idx in (base_idx + delta, base_idx - delta):
                if neighbor_idx < 0:
                    continue
                for fmt in (f"{neighbor_idx:02d}", f"{neighbor_idx:03d}", str(neighbor_idx)):
                    cand_id = f"{prefix}{fmt}"
                    if cand_id in all_chunks_by_id and cand_id not in seen:
                        neighbor_chunk = dict(all_chunks_by_id[cand_id])
                        neighbor_chunk["score"] = round(float(chunk.get("score", 0.0)) * 0.92, 6)
                        neighbor_chunk["strategy"] = f"{chunk.get('strategy', 'unknown')}+neighbor"
                        neighbors.append(neighbor_chunk)
                        seen.add(cand_id)
                        break

    return primary + neighbors


def pack_context_smart(
    candidates: list[dict[str, Any]],
    budget: int = 800,
    *,
    all_chunks_by_id: dict[str, dict[str, Any]] | None = None,
    enable_neighbor_expansion: bool = False,
    neighbor_anchors: int = 1,
    reorder_by_score: bool = False,
) -> tuple[str, list[str]]:
    """Đóng gói ngữ cảnh có khử trùng lặp, mở rộng đoạn lân cận và kiểm soát ngân sách ký tự."""
    if budget < 1 or not candidates:
        return "", []

    # Bước 1: Khử trùng lặp, giữ bản ghi có điểm cao nhất hoặc xuất hiện sớm nhất
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    working = list(candidates)
    if reorder_by_score:
        working.sort(key=lambda c: (-float(c.get("score", 0.0)), c["id"]))

    for c in working:
        cid = c["id"]
        if cid not in seen_ids:
            deduped.append(c)
            seen_ids.add(cid)

    # Bước 2: Mở rộng sang chunk liền kề cùng trang nếu được bật
    if enable_neighbor_expansion and all_chunks_by_id:
        deduped = expand_neighboring_chunks(
            deduped,
            all_chunks_by_id,
            max_anchors_to_expand=neighbor_anchors,
            window=1,
        )

    # Bước 3: Đóng gói tuần tự vào ngân sách ký tự
    blocks: list[str] = []
    ids: list[str] = []
    for c in deduped:
        block = format_chunk_block(c)
        candidate_str = "\n\n".join(blocks + [block])
        if len(candidate_str) <= budget:
            blocks.append(block)
            ids.append(c["id"])

    return "\n\n".join(blocks), ids
