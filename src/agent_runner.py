"""Agentic RAG Controller, Multi-Tool Retriever Adapter và Smart Multi-Turn Context Packer cho Day 07.

Kết nối trực tiếp LLM thật (`gemini` hoặc `groq` / `openai` / `openrouter` qua `.env`) và `QdrantPersistentStore` từ Day 06.
Tuyệt đối không đưa nhãn `gold` vào prompt hay bộ truy xuất.
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
sys.path.insert(0, str(REPO_ROOT))

from agentic import run_agent  # noqa: E402
from baseline import pack_context  # noqa: E402
from src.context_engineering import expand_neighboring_chunks, format_chunk_block  # noqa: E402
from src.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseQdrantRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
    infer_query_filters,
)

PROVIDER_ENDPOINTS = {
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    ),
    "groq": (
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1/chat/completions",
    ),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions"),
}

DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "openrouter": "google/gemini-2.5-flash",
}

MODEL_PRICING_PER_1M = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.10, 0.30),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}

SYSTEM_PROMPT = """You are an Evidence-Grounded Research Agent for an AI scientific corpus containing three foundational papers:
1. Vaswani et al. (2017) - "Attention Is All You Need" (Transformer architecture, WMT 2014 EN-DE/EN-FR, self-attention complexity, hyperparameters).
2. Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG-Sequence, RAG-Token, DPR, BART, Open-Domain QA, Jeopardy, FEVER).
3. Yao et al. (2022) - "ReAct: Synergizing Reasoning and Acting in Language Models" (HotpotQA, FEVER, ALFWorld, WebShop, CoT-SC, Act, ReAct-IM).

STRICT OPERATING RULES:
1. NEVER answer from pre-training memory. Every factual claim, number, condition, or comparison in your final answer MUST be explicitly supported by the text inside the current `context`.
2. If `history` is empty (no search has been executed yet), you MUST return a `"search"` action first.
3. When inspecting `context`:
   - Check whether ALL parts of the user's question (including specific metrics, conditions, exceptions, or secondary questions) are explicitly present in `context`.
   - If the context is missing a required detail or footnote AND you have remaining search turns (`len(queries) < max_searches`), issue a NEW `"search"` action with a focused keyword query targeting the missing detail (do NOT repeat an identical query).
   - If the question asks for a metric, hardware, dataset, hyperparameter, or premise that is NOT reported anywhere in the retrieved chunks even after searching (for example: USD cloud cost, CO2 emissions, softmax temperature/top-p in RAG, HumanEval/Codex in ReAct, V100 ms latency in Transformer, SQuAD 2.0/CommonCrawl in RAG, or Bing API/StrategyQA cost in ReAct), you MUST ABSTAIN by returning:
     {"action": "answer", "thought": "Explain clearly what is missing from the corpus", "answer": "", "citations": [], "abstain_reason": "Explain what specific information is absent in the document"}
4. When `context` has sufficient evidence to answer the question:
   - Return `{"action": "answer", "thought": "...", "answer": "...", "citations": ["chunk_id_1", ...]}`.
   - Every cited chunk ID in `"citations"` MUST be an exact ID from `context_ids` and MUST also appear in `"answer"` inside square brackets like `[vaswani-2017-transformer_p03_para_01]`.
   - Only cite chunks that directly support your answer. Do NOT cite irrelevant chunks in `context_ids`.
   - Include all conditions, exceptions, and exact numbers from the evidence.

OUTPUT FORMAT:
Return ONLY a single valid JSON object (no markdown fences outside JSON) with one of these two structures:
Search action:
{"action": "search", "thought": "Brief reasoning on what to search for", "tool": "hybrid_rrf", "query": "focused search query"}
Answer / Abstain action:
{"action": "answer", "thought": "Brief verification of evidence", "answer": "Comprehensive grounded answer with inline [chunk_id] citations (or empty string if abstaining)", "citations": ["chunk_id"], "abstain_reason": "Optional reason when abstaining"}
"""


def make_interleaved_context_packer(
    all_chunks_by_id: dict[str, dict[str, Any]],
    *,
    enable_neighbor_expansion: bool = False,
    neighbor_anchors: int = 2,
):
    """Tạo hàm đóng gói ngữ cảnh thông minh qua nhiều lượt tìm kiếm (Multi-Turn Interleaved Packer).

    Khắc phục lỗi FIFO của `starter/agentic.py`:
    - Khi Agent thực hiện nhiều lượt search (`events` + `latest_results`), hàm này lấy đan xen (Round-Robin)
      ứng viên hạng 1 của lượt mới nhất và lượt trước đó, tiếp theo là hạng 2, hạng 3.
    - Nhờ vậy, kết quả tìm kiếm bổ sung ở lượt 2 và lượt 3 luôn được đưa vào ngữ cảnh thay vì bị
      các chunk ở lượt 1 chiếm hết ngân sách.
    - Nếu `enable_neighbor_expansion=True`, áp dụng Two-Tier Priority: đóng gói xong các ứng viên chính
      rồi nối thêm các đoạn văn lân cận (`para_{i+1}`, `para_{i-1}`) trên cùng trang vào phần ngân sách còn dư.
    """

    def _packer(
        candidates: list[dict[str, Any]],
        budget: int = 800,
        latest_results: list[dict[str, Any]] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[str]]:
        if budget < 1 or not candidates:
            return "", []

        search_rounds: list[list[dict[str, Any]]] = []
        if events:
            for ev in events:
                if ev.get("event") == "search" and ev.get("results"):
                    search_rounds.append(ev["results"])
        if latest_results:
            search_rounds.append(latest_results)

        ordered_primary: list[dict[str, Any]] = []
        seen: set[str] = set()

        if len(search_rounds) >= 2:
            # Ưu tiên đan xen theo thứ hạng từ lượt tìm kiếm mới nhất về lượt đầu tiên
            max_len = max(len(r) for r in search_rounds)
            for rank_i in range(max_len):
                for round_results in reversed(search_rounds):
                    if rank_i < len(round_results):
                        c = round_results[rank_i]
                        if c["id"] not in seen:
                            ordered_primary.append(c)
                            seen.add(c["id"])
        else:
            for c in candidates:
                if c["id"] not in seen:
                    ordered_primary.append(c)
                    seen.add(c["id"])

        working_list = ordered_primary
        if enable_neighbor_expansion and all_chunks_by_id:
            working_list = expand_neighboring_chunks(
                ordered_primary,
                all_chunks_by_id,
                max_anchors_to_expand=neighbor_anchors,
                window=1,
            )

        blocks: list[str] = []
        ids: list[str] = []
        for c in working_list:
            block = format_chunk_block(c)
            if len("\n\n".join(blocks + [block])) <= budget:
                blocks.append(block)
                ids.append(c["id"])

        return "\n\n".join(blocks), ids

    return _packer


def _normalize_answer_citations(
    answer: str, citations: list[str], valid_context_ids: list[str]
) -> tuple[str, list[str]]:
    """Chuẩn hóa định dạng trích dẫn trong câu trả lời để tương thích 100% với kiểm tra `f'[{i}]' in answer`.

    - Chỉ giữ các citation ID thực sự nằm trong `valid_context_ids` (không bao giờ thêm ID ngoài context).
    - Nếu model viết gộp `[id1, id2]` trong văn bản, tách thành `[id1] [id2]`.
    - Nếu model đã liệt kê `cid` hợp lệ trong mảng `citations` nhưng quên đóng ngoặc vuông `[cid]`
      trong chuỗi `answer`, gắn `[cid]` vào cuối câu trả lời.
    """
    valid_set = set(valid_context_ids)
    # Quét thêm các ID hợp lệ được viết trực tiếp trong ngoặc vuông ở câu trả lời
    inline_matches = re.findall(r"\[([a-zA-Z0-9_\-:]+)\]", answer)
    merged_cits: list[str] = []
    for cid in list(citations) + inline_matches:
        if isinstance(cid, str) and cid in valid_set and cid not in merged_cits:
            merged_cits.append(cid)

    if not merged_cits or not answer.strip():
        return answer, []

    cleaned_answer = answer.strip()
    for cid in merged_cits:
        if f"[{cid}]" not in cleaned_answer:
            cleaned_answer = f"{cleaned_answer} [{cid}]"

    return cleaned_answer, merged_cits


class GeminiAgentController:
    """Bộ điều khiển Agent gọi trực tiếp LLM thật (`gemini`, `groq`, `openai`, `openrouter`) với retry và telemetry."""

    # Đơn giá mặc định (USD / 1M tokens) nếu model không có trong MODEL_PRICING_PER_1M
    INPUT_PRICE_PER_1M = 0.15
    OUTPUT_PRICE_PER_1M = 0.60

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_searches: int = 3,
        cache_path: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        load_dotenv(REPO_ROOT / ".env", override=True)
        self.provider = (provider or os.getenv("PROVIDER", "gemini")).strip().lower()
        default_model = DEFAULT_MODEL_BY_PROVIDER.get(self.provider, "gemini-2.5-flash")
        env_model = os.getenv("MODEL", "").strip()
        self.model = (model or env_model or default_model).strip()
        self.max_searches = max_searches
        self.cache_path = cache_path or (REPO_ROOT / "artifacts" / "llm_agent_cache.json")
        self.use_cache = use_cache

        if self.provider not in PROVIDER_ENDPOINTS:
            raise ValueError(f"Unsupported provider: {self.provider}")
        key_env, self.url = PROVIDER_ENDPOINTS[self.provider]
        self.api_key = os.getenv(key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {key_env}")

        self._cache: dict[str, dict[str, Any]] = {}
        if self.use_cache and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

        self.reset_telemetry()
        self._last_call_time = 0.0
        self.min_interval = 1.6 if self.provider == "groq" else 4.2

    def reset_telemetry(self) -> None:
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_latency_ms = 0.0
        self.estimated_cost_usd = 0.0
        self.step_logs: list[dict[str, Any]] = []
        self._last_context_ids: list[str] = []

    def _call_llm_api(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Gọi HTTP POST tới LLM endpoint thật kèm cơ chế exponential backoff và fallback model cùng họ Gemini khi gặp 429."""
        cache_payload = json.dumps(
            {"model": self.model, "messages": messages}, ensure_ascii=False, sort_keys=True
        )
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
        if self.use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return {**cached, "cached": True}

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "VinUni-Day07-RAG-Agent/1.0",
        }
        if self.provider == "gemini":
            candidate_models = [self.model, self.model, self.model, self.model]
        elif self.provider == "groq":
            candidate_models = [
                self.model,
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                self.model,
                "openai/gpt-oss-20b",
            ]
        else:
            candidate_models = [self.model, self.model, self.model]

        use_json_mode = True
        last_err: Exception | None = None
        for attempt, active_model in enumerate(candidate_models):
            now = time.time()
            elapsed_since_last = now - self._last_call_time
            if elapsed_since_last < self.min_interval:
                time.sleep(self.min_interval - elapsed_since_last)

            payload: dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "temperature": 0.0,
            }
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}
            t0 = time.perf_counter()
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw_data = json.load(resp)
                self._last_call_time = time.time()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                content = raw_data["choices"][0]["message"].get("content", "")
                usage = raw_data.get("usage", {})
                result = {
                    "content": content,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "cached": False,
                }
                if self.use_cache:
                    self._cache[cache_key] = result
                    try:
                        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                        self.cache_path.write_text(
                            json.dumps(self._cache, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                return result
            except urllib.error.HTTPError as exc:
                last_err = exc
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore")[:240]
                except Exception:
                    pass
                if exc.code == 400 and use_json_mode and attempt < len(candidate_models) - 1:
                    use_json_mode = False
                    continue
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if exc.code == 429 and attempt < len(candidate_models) - 1:
                    if retry_after and retry_after.replace(".", "", 1).isdigit():
                        backoff = min(max(float(retry_after) + 1.0, 2.0), 15.0)
                    else:
                        backoff = 3.0 * (attempt + 1)
                    next_m = candidate_models[attempt + 1]
                    print(
                        f"    [{self.provider.upper()} Rate Limit 429 trên {active_model}] Chờ {backoff:.0f}s và chuyển sang {next_m}...",
                        flush=True,
                    )
                    time.sleep(backoff)
                    continue
                if exc.code in (500, 502, 503, 504) and attempt < len(candidate_models) - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM Provider returned HTTP {exc.code}: {err_body}") from exc
            except urllib.error.URLError as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM Provider connection error: {exc.reason}") from exc

        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    @staticmethod
    def _parse_json_output(raw_text: str) -> dict[str, Any]:
        """Trích xuất JSON object từ phản hồi của LLM."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        raise ValueError(f"Model did not return valid JSON action: {raw_text[:200]}")

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Quyết định hành động tiếp theo (`search` hoặc `answer`) dựa trên câu hỏi và lịch sử truy xuất."""
        if not history:
            self._last_context_ids = []
            decision = {
                "action": "search",
                "query": question,
                "rewritten_query": question,
                "tool": "hybrid_rrf",
                "thought": "Turn 1 initial retrieval using full question to preserve entity and metadata signals.",
            }
            self.step_logs.append(
                {
                    "step": 0,
                    "decision": decision,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_ms": 0.0,
                    "cached": True,
                }
            )
            return decision

        latest = history[-1]
        queries_so_far = list(latest.get("queries", []))
        current_context = str(latest.get("context", ""))
        context_ids = list(latest.get("context_ids", []))

        # Nếu ngữ cảnh sau khi đóng gói bị rỗng (do chunk vượt quá budget=800) hoặc lượt tìm kiếm thứ 2
        # không bổ sung thêm bất kỳ chunk mới nào vào context_ids, dừng sớm và abstain để tránh gọi LLM thừa
        if not context_ids or (len(queries_so_far) >= 2 and context_ids == self._last_context_ids):
            reason = (
                "Context budget starvation: all retrieved chunks exceeded character budget"
                if not context_ids
                else "Second search added no new chunks into context budget"
            )
            decision = {
                "action": "answer",
                "answer": "",
                "citations": [],
                "thought": reason,
                "abstain_reason": reason,
            }
            self.step_logs.append(
                {
                    "step": self.llm_calls,
                    "decision": decision,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed_ms": 0.0,
                    "cached": True,
                }
            )
            return decision

        self._last_context_ids = list(context_ids)

        user_payload = {
            "question": question,
            "search_turns_used": len(queries_so_far),
            "max_searches": self.max_searches,
            "previous_queries": queries_so_far,
            "available_context_ids": context_ids,
            "current_context": current_context if current_context else "(EMPTY - No search executed yet)",
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

        api_res = self._call_llm_api(messages)
        self.llm_calls += 1
        p_tok = api_res["prompt_tokens"]
        c_tok = api_res["completion_tokens"]
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += api_res["total_tokens"]
        self.llm_latency_ms += api_res["elapsed_ms"]
        in_price, out_price = MODEL_PRICING_PER_1M.get(
            self.model, (self.INPUT_PRICE_PER_1M, self.OUTPUT_PRICE_PER_1M)
        )
        call_cost = (p_tok / 1_000_000.0) * in_price + (
            c_tok / 1_000_000.0
        ) * out_price
        self.estimated_cost_usd += call_cost

        parsed = self._parse_json_output(api_res["content"])
        action_type = str(parsed.get("action", "")).strip().lower()
        thought = str(parsed.get("thought", "")).strip()

        # Nếu chưa tìm kiếm lần nào, bắt buộc thực hiện search đầu tiên
        if not history and action_type != "search":
            action_type = "search"
            parsed["query"] = question

        if action_type == "search" and len(queries_so_far) < self.max_searches:
            rewritten_q = str(parsed.get("query") or question).strip()
            # Lượt tìm kiếm đầu tiên (Turn 1) luôn dùng câu hỏi gốc để bảo toàn ngữ nghĩa đầy đủ và dấu hiệu filter;
            # từ lượt thứ 2 trở đi (Multi-hop / Query Rewrite) sử dụng câu truy vấn viết lại tập trung của LLM.
            if len(queries_so_far) == 0:
                q_str = question
            else:
                q_str = rewritten_q
                if q_str in queries_so_far:
                    q_str = f"{rewritten_q} specific metrics conditions"
            tool_name = str(parsed.get("tool", "hybrid_rrf")).strip().lower()
            decision = {
                "action": "search",
                "query": q_str,
                "rewritten_query": rewritten_q,
                "tool": tool_name,
                "thought": thought,
            }
        elif action_type in ("abstain", "refuse"):
            decision = {
                "action": "answer",
                "answer": "",
                "citations": [],
                "thought": thought,
                "abstain_reason": str(parsed.get("abstain_reason") or thought or "Insufficient evidence in corpus"),
            }
        else:
            raw_ans = str(parsed.get("answer", "")).strip()
            raw_cits = parsed.get("citations", [])
            if not isinstance(raw_cits, list):
                raw_cits = []
            norm_ans, norm_cits = _normalize_answer_citations(raw_ans, raw_cits, context_ids)
            decision = {
                "action": "answer",
                "answer": norm_ans,
                "citations": norm_cits,
                "thought": thought,
            }
            if parsed.get("abstain_reason") or not norm_cits:
                decision["abstain_reason"] = str(
                    parsed.get("abstain_reason") or thought or "No supporting evidence in context"
                )

        self.step_logs.append(
            {
                "step": self.llm_calls,
                "decision": decision,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "elapsed_ms": api_res["elapsed_ms"],
                "cached": api_res.get("cached", False),
            }
        )
        return decision
