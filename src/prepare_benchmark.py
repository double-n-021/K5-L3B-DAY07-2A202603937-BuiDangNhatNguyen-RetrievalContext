"""Chuẩn bị dữ liệu bàn giao từ Day 06 và xây dựng bộ Benchmark mở rộng 32 câu cho Day 07.

- Xuất 232 chunks (Strategy B/C: by_paragraph, max_size=800) từ QdrantPersistentStore của Day 06.
- Chuẩn hóa 26 câu hỏi có đáp án (19 câu dev + 7 câu test) trong `questions.json`.
- Xây dựng 32 cases đánh giá Agent (22 câu dev + 10 câu test, gồm 26 câu có đáp án và 6 câu thiếu bằng chứng) trong `cases.json`.
"""
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY06_ROOT = REPO_ROOT.parent / "K5-L3-Day06-DataIngestionEngineering"
sys.path.insert(0, str(REPO_ROOT / "starter"))
from baseline import hit, validate  # noqa: E402
from convert_day6 import convert_chunk, convert_question  # noqa: E402


def load_day06_chunks() -> list[dict[str, Any]]:
    """Đọc 232 chunks của Strategy B từ QdrantPersistentStore của Day 06 và chuyển sang schema Day 07."""
    export_path = REPO_ROOT / "starter" / "data" / "day6_export" / "day6_chunks.json"
    if export_path.exists():
        raw_chunks = json.loads(export_path.read_text(encoding="utf-8"))
    else:
        sys.path.insert(0, str(DAY06_ROOT))
        from starter.src.store import QdrantPersistentStore  # type: ignore

        store = QdrantPersistentStore(DAY06_ROOT / "results" / "qdrant_db", 768)
        pts, _ = store._client.scroll(
            "day06_document_chunks",
            scroll_filter=store._qmodels.Filter(
                must=[
                    store._qmodels.FieldCondition(
                        key="strategy", match=store._qmodels.MatchValue(value="B")
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
        )
        store.close()
        raw_chunks = sorted([p.payload for p in pts if p.payload], key=lambda x: x["chunk_id"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(raw_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    return [convert_chunk(c) for c in raw_chunks]


def build_benchmark_questions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tạo 26 câu hỏi có đáp án (19 dev, 7 test) và 6 câu hỏi thiếu bằng chứng (3 dev, 3 test)."""
    day06_q_path = DAY06_ROOT / "benchmark" / "questions.json"
    day06_qs = json.loads(day06_q_path.read_text(encoding="utf-8"))["questions"]

    answerable_questions: list[dict[str, Any]] = []
    for q in day06_qs:
        qid = q["id"]
        # Với q12 và q18, bằng chứng nằm vắt qua 2 đoạn văn liên tiếp trên cùng 1 trang PDF
        if qid == "q12":
            gold = [
                {
                    "source_id": "yao-2022-react",
                    "page": 5,
                    "evidence": "A) ReAct→ CoT-SC: when ReAct fails to return an answer within given steps, back off to CoT-SC. We set 7 and 5 steps for HotpotQA and FEVER respectively",
                },
                {
                    "source_id": "yao-2022-react",
                    "page": 5,
                    "evidence": "B) CoT-SC→ ReAct: when the majority answer among n CoT-SC samples occurs less thann/2 times (i.e. internal knowledge might not support the task confidently), back off toReAct.",
                },
            ]
            req_parts = [g["evidence"] for g in gold]
            q_type = "split_evidence"
        elif qid == "q18":
            gold = [
                {
                    "source_id": "yao-2022-react",
                    "page": 6,
                    "evidence": "A) Hallucination is a serious problem for CoT, resulting in much higher false positive rate than ReAct (14% vs. 6%) in success mode, and make up its major failure mode (56%).",
                },
                {
                    "source_id": "yao-2022-react",
                    "page": 6,
                    "evidence": "B) While interleaving reasoning, action and observation steps improves ReAct’s groundedness and trustworthiness, such a structural constraint also reduces its flexibility in formulating reasoning steps",
                },
            ]
            req_parts = [g["evidence"] for g in gold]
            q_type = "split_evidence"
        else:
            gold = [
                {
                    "source_id": q["gold_source_id"],
                    "page": int(q["gold_page"]),
                    "evidence": q["evidence_quote"],
                }
            ]
            req_parts = [q["evidence_quote"]]
            q_type = q.get("type", "truc_tiep")

        item: dict[str, Any] = {
            "id": qid,
            "split": q["split"],
            "type": q_type,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": gold,
            "required_evidence_parts": req_parts,
            "evidence_anchors": q.get("evidence_anchors", []),
            "conditions_or_exceptions": (
                "Cần nêu đủ cả hai chiều kết hợp A) ReAct -> CoT-SC (7 và 5 bước) và B) CoT-SC -> ReAct (ngưỡng < n/2)."
                if qid == "q12"
                else (
                    "Cần nêu đủ hai mặt đối lập: A) CoT bị hallucination cao (14% vs 6%, chiếm 56% lỗi) và B) ReAct kém linh hoạt hơn, dễ lặp lại thought/action trước đó."
                    if qid == "q18"
                    else "Bám sát thông số, điều kiện và ngoại lệ trong đoạn bằng chứng gốc."
                )
            ),
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        answerable_questions.append(item)

    # Bổ sung 6 câu hỏi có đáp án mới (4 câu dev: q21..q24 và 2 câu test: q25..q26)
    additional_answerable: list[dict[str, Any]] = [
        {
            "id": "q21",
            "split": "dev",
            "type": "exact_lexical",
            "question": "During Transformer training, what label smoothing value εls is applied, and what trade-off does it create between perplexity and BLEU score?",
            "expected_answer": "During training, the authors applied label smoothing of value εls = 0.1. It hurts perplexity because the model learns to be more unsure, but it improves accuracy and BLEU score.",
            "gold": [
                {
                    "source_id": "vaswani-2017-transformer",
                    "page": 8,
                    "evidence": "During training, we employed label smoothing of value εls = 0 .1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.",
                }
            ],
            "required_evidence_parts": [
                "During training, we employed label smoothing of value εls = 0 .1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."
            ],
            "evidence_anchors": ["εls = 0 .1", "hurts perplexity", "improves accuracy and BLEU score"],
            "conditions_or_exceptions": "Phải nêu đúng giá trị εls = 0.1 và giải thích rõ đánh đổi: làm giảm (hurt) perplexity nhưng cải thiện accuracy và BLEU.",
        },
        {
            "id": "q22",
            "split": "dev",
            "type": "split_evidence",
            "question": "How do the authors estimate the number of floating-point operations (FLOPs) used to train a translation model, and what exact sustained single-precision TFLOPS values did they assume for K80, K40, M40, and P100 GPUs?",
            "expected_answer": "They estimate training FLOPs by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU, assuming 2.8, 3.7, 6.0, and 9.5 TFLOPS for K80, K40, M40, and P100 respectively.",
            "gold": [
                {
                    "source_id": "vaswani-2017-transformer",
                    "page": 8,
                    "evidence": "We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU",
                },
                {
                    "source_id": "vaswani-2017-transformer",
                    "page": 8,
                    "evidence": "We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.",
                },
            ],
            "required_evidence_parts": [
                "We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU",
                "We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.",
            ],
            "evidence_anchors": ["multiplying the training time", "2.8, 3.7, 6.0 and 9.5 TFLOPS"],
            "conditions_or_exceptions": "Công thức nằm ở đoạn văn chính (para_05) còn 4 con số TFLOPS của K80/K40/M40/P100 nằm ở chú thích cuối trang (para_06); cần ghép đủ cả hai đoạn để trả lời trọn vẹn.",
        },
        {
            "id": "q23",
            "split": "dev",
            "type": "dien_dat_lai",
            "question": "In the Natural Questions (NQ) benchmark, how well does RAG perform when none of the retrieved passages actually contain the ground-truth answer, compared to an extractive reader?",
            "expected_answer": "RAG can still generate correct answers even when the correct answer is not present in any retrieved document, achieving 11.8% accuracy in such cases on Natural Questions (NQ), whereas an extractive model scores 0%.",
            "gold": [
                {
                    "source_id": "lewis-2020-rag",
                    "page": 6,
                    "evidence": "Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%.",
                }
            ],
            "required_evidence_parts": [
                "Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%."
            ],
            "evidence_anchors": ["11.8% accuracy", "extractive model would score 0%"],
            "conditions_or_exceptions": "Phải nêu rõ điều kiện 'khi đáp án đúng không nằm trong bất kỳ tài liệu truy xuất nào' và đối chiếu con số 11.8% của RAG với 0% của mô hình trích xuất (extractive model).",
        },
        {
            "id": "q24",
            "split": "dev",
            "type": "dieu_kien_ngoai_le",
            "question": "In the 2022 agent paper, how does ReAct compare against Inner Monologue-style prompting (ReAct-IM) on ALFWorld in overall success rate, and what two qualitative shortcomings caused ReAct-IM to fail?",
            "expected_answer": "ReAct substantially outperforms Inner Monologue-style prompting (ReAct-IM) on ALFWorld with a 71% vs. 53% overall success rate. Qualitatively, ReAct-IM failed because: (1) it made mistakes identifying when subgoals were finished or what the next subgoal should be due to a lack of high-level goal decomposition, and (2) it struggled to determine likely locations of items within ALFWorld due to a lack of commonsense reasoning.",
            "gold": [
                {
                    "source_id": "yao-2022-react",
                    "page": 8,
                    "evidence": "As can be seen in Table 3,ReAct substantially outperforms IM-style prompting ( ReAct-IM) (71 vs. 53 overall success rate), with consistent advantages on five out of six tasks. Qualitatively, we observed thatReAct-IM often made mistakes in identifying when subgoals were finished, or what the next subgoal should be, due to a lack of highlevel goal decomposition. Additionally, many ReAct-IM trajectories struggled to determine where an item would likely be within the ALFWorld environment, due to a lack of commonsense reasoning.",
                }
            ],
            "required_evidence_parts": [
                "As can be seen in Table 3,ReAct substantially outperforms IM-style prompting ( ReAct-IM) (71 vs. 53 overall success rate)"
            ],
            "evidence_anchors": ["71 vs. 53 overall success rate", "highlevel goal decomposition", "commonsense reasoning"],
            "conditions_or_exceptions": "Cần nêu đủ tỷ lệ thành công 71% vs 53% và 2 nguyên nhân định tính: thiếu phân rã mục tiêu cấp cao (high-level goal decomposition) và thiếu suy luận thường thức (commonsense reasoning).",
            "filter": {"category": "agent", "year": 2022},
        },
        {
            "id": "q25",
            "split": "test",
            "type": "exact_lexical",
            "question": "For the English-to-French WMT 2014 translation task, what BLEU score did the Transformer (big) model achieve, what dropout rate Pdrop was used compared to English-to-German, and what beam size and length penalty α were used during inference?",
            "expected_answer": "On the WMT 2014 English-to-French task, the Transformer (big) model achieved a BLEU score of 41.0 (or 41.8 in Table 2), using a dropout rate Pdrop = 0.1 instead of 0.3. During inference, the authors used beam search with a beam size of 4 and length penalty α = 0.6.",
            "gold": [
                {
                    "source_id": "vaswani-2017-transformer",
                    "page": 8,
                    "evidence": "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.",
                }
            ],
            "required_evidence_parts": [
                "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0",
                "We used beam search with a beam size of 4 and length penalty α = 0 .6",
            ],
            "evidence_anchors": ["BLEU score of 41.0", "Pdrop = 0.1, instead of 0.3", "beam size of 4", "α = 0 .6"],
            "conditions_or_exceptions": "Cần nêu đủ BLEU (41.0), điều kiện thay đổi dropout Pdrop = 0.1 thay vì 0.3 cho EN-FR, cùng tham số suy luận beam size = 4 và α = 0.6.",
        },
        {
            "id": "q26",
            "split": "test",
            "type": "split_evidence",
            "question": "In the Jeopardy question generation experiment on Hemingway's novels ('The Sun Also Rises' and 'A Farewell to Arms'), how does the document posterior behave after the first token of each book title is generated, and how did the authors use the BART-only baseline with partial decoding to show that parametric and non-parametric memories work together?",
            "expected_answer": "When generating 'Sun', document 2 ('The Sun Also Rises') has a high posterior, and document 1 dominates when 'A Farewell to Arms' is generated, but after the first token of each book is generated, the document posterior flattens. Feeding the BART-only baseline with the partial decoding 'The Sun' or 'A' allows BART to complete the full titles from its own parameters, showing that non-parametric memory guides generation by drawing out specific knowledge stored in parametric memory.",
            "gold": [
                {
                    "source_id": "lewis-2020-rag",
                    "page": 6,
                    "evidence": "When generating “Sun”, the posterior is high for document 2 which mentions “The Sun Also Rises”. Similarly, document 1 dominates the posterior when “A Farewell to Arms” is generated. Intriguingly, after the first token of each book is generated, the document posterior flattens.",
                },
                {
                    "source_id": "lewis-2020-rag",
                    "page": 6,
                    "evidence": "BART completes the generation \"The Sun Also Rises\" is a novel by this author of \"The Sun Also Rises\" indicating the title \"The Sun Also Rises\" is stored in BART’s parameters.",
                },
            ],
            "required_evidence_parts": [
                "after the first token of each book is generated, the document posterior flattens.",
                "This example shows how parametric and non-parametric memories work together—the non-parametric component helps to guide the generation, drawing out specific knowledge stored in the parametric memory.",
            ],
            "evidence_anchors": ["document posterior flattens", "parametric and non-parametric memories work together"],
            "conditions_or_exceptions": "Câu chuyện về posterior phẳng lại nằm ở cuối đoạn para_04, còn kết quả hoàn thành chuỗi của BART và kết luận phối hợp hai loại bộ nhớ nằm ở đoạn para_05 kế tiếp trên trang 6.",
        },
    ]
    answerable_questions.extend(additional_answerable)

    # Sắp xếp để dev đứng trước test hoặc theo ID rõ ràng
    unanswerable_cases: list[dict[str, Any]] = [
        {
            "id": "u01",
            "split": "dev",
            "type": "unanswerable_missing_metric",
            "question": "What was the exact cloud computing bill in US dollars and the total carbon footprint in kg CO2e reported by Vaswani et al. (2017) for training the Transformer (big) model on 8 P100 GPUs?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "expected_answer": "Không đủ bằng chứng trong tài liệu: Bài báo Vaswani et al. (2017) chỉ báo cáo thời gian huấn luyện (3.5 ngày trên 8 GPU P100) và số phép tính dấu phẩy động ước tính (2.3 * 10^19 FLOPs), hoàn toàn không đề cập chi phí tính bằng USD hay lượng phát thải carbon (kg CO2e).",
            "absence_reason": "Đã kiểm tra toàn bộ 10 trang hợp lệ của vaswani-2017-transformer (và toàn bộ 232 chunks): chỉ có thông tin '3.5 days on 8 P100 GPUs' và '2.3 * 10^19 FLOPs' ở trang 8; các từ khóa 'dollar', 'USD', 'CO2', 'carbon', 'emission' xuất hiện 0 lần.",
        },
        {
            "id": "u02",
            "split": "dev",
            "type": "unanswerable_missing_hyperparam",
            "question": "When generating open-domain QA responses in RAG-Sequence and RAG-Token (Lewis et al., 2020), what exact softmax temperature τ and nucleus sampling threshold top-p did the authors configure?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "expected_answer": "Không đủ bằng chứng trong tài liệu: Bài báo Lewis et al. (2020) chỉ mô tả cơ chế giải mã bằng beam search (Thorough Decoding và Fast Decoding), không sử dụng hay báo cáo tham số softmax temperature τ hoặc nucleus sampling top-p cho bài toán QA.",
            "absence_reason": "Đã kiểm tra toàn bộ 10 trang hợp lệ của lewis-2020-rag (Mục 2.3 Decoding và Mục 3/4 Experiments): tác giả chỉ dùng beam search; các thuật ngữ 'temperature', 'nucleus', 'top-p' xuất hiện 0 lần.",
        },
        {
            "id": "u03",
            "split": "dev",
            "type": "unanswerable_premise_trap",
            "question": "Confirm that in Yao et al. (2022), the ReAct framework achieved an 84.5% pass@1 accuracy on the HumanEval Python code generation benchmark using Codex-175B, and cite the corresponding table.",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "expected_answer": "Không đủ bằng chứng trong tài liệu: Bài báo Yao et al. (2022) chỉ đánh giá ReAct trên 4 bộ dữ liệu (HotpotQA, FEVER, ALFWorld, WebShop) với mô hình PaLM và GPT-3, hoàn toàn không có thực nghiệm nào trên bộ dữ liệu lập trình HumanEval hay mô hình Codex-175B.",
            "absence_reason": "Câu hỏi gài tiền đề sai ngoài phạm vi tài liệu: đã quét toàn bộ 17 trang hợp lệ của yao-2022-react; các từ khóa 'HumanEval', 'pass@1', 'Codex' đều xuất hiện 0 lần.",
            "filter": {"category": "agent", "year": 2022},
        },
        {
            "id": "u04",
            "split": "test",
            "type": "unanswerable_missing_metric",
            "question": "What was the exact end-to-end wall-clock inference latency in milliseconds per token reported for the base Transformer model on an NVIDIA V100 GPU with batch size 1?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "expected_answer": "Không đủ bằng chứng trong tài liệu: Vaswani et al. (2017) chỉ phân tích độ phức tạp tính toán lý thuyết theo tầng (Table 1) và thời gian huấn luyện trên GPU P100, không báo cáo độ trễ suy luận thực tế bằng mili-giây mỗi token hay thử nghiệm trên GPU NVIDIA V100.",
            "absence_reason": "Đã kiểm tra toàn bộ các đoạn văn bản của vaswani-2017-transformer: chỉ có K80, K40, M40, P100 ở trang 8; từ khóa 'V100', 'millisecond', 'latency' xuất hiện 0 lần.",
        },
        {
            "id": "u05",
            "split": "test",
            "type": "unanswerable_missing_benchmark",
            "question": "What Exact Match (EM) and F1 scores did RAG-Sequence achieve on the SQuAD 2.0 unanswerable question benchmark when the FAISS index was switched from Wikipedia to CommonCrawl?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "expected_answer": "Không đủ bằng chứng trong tài liệu: Lewis et al. (2020) chỉ dùng Wikipedia (tháng 12/2018) làm kho tri thức non-parametric và không công bố bảng điểm EM/F1 nào của RAG trên SQuAD 2.0 hay chỉ mục CommonCrawl.",
            "absence_reason": "Đã kiểm tra toàn bộ bài báo lewis-2020-rag: tác giả chỉ nhắc tới SQuAD 1.1 ở phần giới thiệu/thảo luận như một ví dụ về bài toán extractive QA; từ khóa 'SQuAD 2.0' và 'CommonCrawl' xuất hiện 0 lần.",
        },
        {
            "id": "u06",
            "split": "test",
            "type": "unanswerable_premise_trap",
            "question": "In Yao et al. (2022), what was the exact API cost in US dollars per trajectory and the average HTTP round-trip latency in milliseconds when ReAct queried the commercial Bing Web Search API on StrategyQA?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "expected_answer": "Không đủ bằng chứng trong tài liệu: Yao et al. (2022) sử dụng Wikipedia API đơn giản (search, lookup, finish) thay vì Bing Web Search API, không đánh giá trên StrategyQA trong nội dung bài báo và không báo cáo chi phí USD mỗi trajectory hay độ trễ HTTP bằng mili-giây.",
            "absence_reason": "Câu hỏi gài tiền đề không có trong tài liệu: đã kiểm tra toàn bộ 17 trang hợp lệ của yao-2022-react; các từ khóa 'Bing', 'StrategyQA', 'round-trip', 'millisecond' xuất hiện 0 lần.",
            "filter": {"category": "agent", "year": 2022},
        },
    ]

    cases: list[dict[str, Any]] = []
    for q in answerable_questions:
        case_item: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "type": q["type"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
            "required_evidence_parts": q["required_evidence_parts"],
            "conditions_or_exceptions": q["conditions_or_exceptions"],
        }
        if "filter" in q:
            case_item["filter"] = q["filter"]
        cases.append(case_item)

    for u in unanswerable_cases:
        cases.append(u)

    return answerable_questions, cases


def main() -> None:
    chunks = load_day06_chunks()
    questions, cases = build_benchmark_questions()

    # Kiểm tra hợp đồng dữ liệu bằng chính hàm validate() của starter/baseline.py
    dev_qs = [q for q in questions if q["split"] == "dev"]
    test_qs = [q for q in questions if q["split"] == "test"]
    validate(chunks, dev_qs)
    validate(chunks, test_qs)

    # Kiểm tra chắc chắn mọi phần tử trong required_evidence_parts đều nằm trong ít nhất 1 chunk đúng nguồn/trang
    for q in questions:
        src_id = q["gold"][0]["source_id"]
        page = q["gold"][0]["page"]
        for part in q["required_evidence_parts"]:
            matched = any(
                c["source_id"] == src_id and c["page"] == page and part in c["text"]
                for c in chunks
            )
            if not matched:
                raise ValueError(f"Question {q['id']} required part not found in {src_id} p.{page}: {part[:80]}")

    # Lưu vào cả data/benchmark/ (nộp chính thức) và starter/data/converted/ (tương thích CLI starter)
    for out_dir in (REPO_ROOT / "data" / "benchmark", REPO_ROOT / "starter" / "data" / "converted"):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "questions.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    dev_cases = [c for c in cases if c["split"] == "dev"]
    test_cases = [c for c in cases if c["split"] == "test"]
    print(f"Total chunks exported & validated: {len(chunks)}")
    print(
        f"Answerable questions (questions.json): {len(questions)} "
        f"(dev={len(dev_qs)}, test={len(test_qs)})"
    )
    print(
        f"Total Agent cases (cases.json): {len(cases)} "
        f"(dev={len(dev_cases)} [{sum(c['answerable'] for c in dev_cases)} ans + {sum(not c['answerable'] for c in dev_cases)} unans], "
        f"test={len(test_cases)} [{sum(c['answerable'] for c in test_cases)} ans + {sum(not c['answerable'] for c in test_cases)} unans])"
    )


if __name__ == "__main__":
    main()
