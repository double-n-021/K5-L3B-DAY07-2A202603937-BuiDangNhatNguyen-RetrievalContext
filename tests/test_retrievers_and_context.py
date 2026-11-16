"""Bộ kiểm thử tự động mở rộng cho 4 phương pháp Retrieval, Safe Filter Inference, Context Engineering và Benchmark 32 câu."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT))

from baseline import hit, validate  # noqa: E402
from src.context_engineering import (  # noqa: E402
    expand_neighboring_chunks,
    pack_context_smart,
    parse_chunk_index,
)
from src.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseQdrantRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
    infer_query_filters,
    matches_filter,
    tokenize_technical,
)


class RetrievalAndContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks_path = ROOT / "data" / "benchmark" / "chunks.json"
        cls.questions_path = ROOT / "data" / "benchmark" / "questions.json"
        cls.cases_path = ROOT / "data" / "benchmark" / "cases.json"
        cls.chunks = json.loads(cls.chunks_path.read_text(encoding="utf-8"))
        cls.questions = json.loads(cls.questions_path.read_text(encoding="utf-8"))
        cls.cases = json.loads(cls.cases_path.read_text(encoding="utf-8"))
        cls.chunks_by_id = {c["id"]: c for c in cls.chunks}

    def test_01_benchmark_32_cases_contract(self) -> None:
        """Kiểm tra bộ Benchmark mở rộng có đúng 232 chunks, 26 câu có đáp án và 32 cases (22 dev + 10 test)."""
        self.assertEqual(len(self.chunks), 232)
        self.assertEqual(len(self.questions), 26)
        self.assertEqual(len(self.cases), 32)

        dev_q = [q for q in self.questions if q["split"] == "dev"]
        test_q = [q for q in self.questions if q["split"] == "test"]
        self.assertEqual(len(dev_q), 19)
        self.assertEqual(len(test_q), 7)
        validate(self.chunks, dev_q)
        validate(self.chunks, test_q)

        dev_cases = [c for c in self.cases if c["split"] == "dev"]
        test_cases = [c for c in self.cases if c["split"] == "test"]
        self.assertEqual(len(dev_cases), 22)
        self.assertEqual(len(test_cases), 10)
        self.assertEqual(sum(not c["answerable"] for c in dev_cases), 3)
        self.assertEqual(sum(not c["answerable"] for c in test_cases), 3)

    def test_02_unanswerable_questions_truly_absent(self) -> None:
        """Đảm bảo 6 câu thiếu bằng chứng có đầy đủ lý do kiểm chứng và không chứa từ khóa giả định trong corpus."""
        unans = [c for c in self.cases if not c["answerable"]]
        self.assertEqual(len(unans), 6)
        full_corpus_lower = "\n".join(c["text"].lower() for c in self.chunks)
        for u in unans:
            self.assertEqual(u["expected_behavior"], "search_and_abstain")
            self.assertTrue(isinstance(u.get("absence_reason"), str) and len(u["absence_reason"]) > 30)
        # Kiểm tra các thuật ngữ đặc thù trong 6 câu unanswerable thực sự không xuất hiện trong tài liệu đích
        for absent_term in ("co2e", "top-p", "humaneval", "commoncrawl", "strategyqa"):
            self.assertNotIn(absent_term, full_corpus_lower)
        vaswani_lower = "\n".join(
            c["text"].lower() for c in self.chunks if c["source_id"] == "vaswani-2017-transformer"
        )
        self.assertNotIn("v100", vaswani_lower)

    def test_03_technical_tokenizer_preserves_hyphenated_terms(self) -> None:
        """Kiểm tra bộ tách từ kỹ thuật giữ được các cụm gạch nối như rag-sequence, cot-sc, react-im."""
        tokens = tokenize_technical("The RAG-Sequence and CoT-SC models outperform ReAct-IM on EN-DE.")
        self.assertIn("rag-sequence", tokens)
        self.assertIn("cot-sc", tokens)
        self.assertIn("react-im", tokens)
        self.assertIn("en-de", tokens)
        self.assertNotIn("the", tokens)

    def test_04_safe_query_filter_inference_single_domain(self) -> None:
        """Kiểm tra suy luận filter tự động từ câu hỏi khi chỉ có 1 năm hoặc 1 miền duy nhất."""
        f1, r1 = infer_query_filters("In the 2017 Transformer paper, how many heads are used?")
        self.assertEqual(f1, {"year": 2017, "category": "architecture"})
        self.assertIn("year_in_query=2017", r1)

        f2, _ = infer_query_filters("How does RAG-Sequence marginalize latent documents in 2020?")
        self.assertEqual(f2, {"year": 2020, "category": "rag"})

        f3, _ = infer_query_filters("How does ReAct compare with CoT-SC on ALFWorld?")
        self.assertEqual(f3, {"category": "agent"})

    def test_05_safe_query_filter_avoids_multi_domain_locking(self) -> None:
        """Nếu câu hỏi nhắc tới nhiều bài báo hoặc nhiều năm, không được khóa cứng filter gây mất dữ liệu."""
        f_multi, reason = infer_query_filters(
            "Compare the 2017 Transformer architecture with the 2022 ReAct agent framework."
        )
        self.assertIsNone(f_multi)
        self.assertEqual(reason, "no_filter")

    def test_06_matches_filter_validation(self) -> None:
        """Kiểm tra hàm lọc metadata hỗ trợ cả source_id, document_id, year, category và từ chối khóa lạ."""
        sample = self.chunks[0]
        self.assertTrue(matches_filter(sample, {"source_id": sample["source_id"]}))
        self.assertTrue(matches_filter(sample, {"year": sample["metadata"]["year"]}))
        self.assertFalse(matches_filter(sample, {"year": 1999}))
        with self.assertRaises(ValueError):
            matches_filter(sample, {"malicious_sql_key": "1; DROP TABLE"})

    def test_07_tfidf_retriever_contract_and_filtering(self) -> None:
        """Kiểm tra Method 1 (TfidfRetriever) trả đúng định dạng và áp dụng filter khi bật."""
        r = TfidfRetriever(self.chunks, use_filters=True)
        res = r.search("label smoothing εls perplexity BLEU", filters={"year": 2017}, top_k=3)
        self.assertGreaterEqual(len(res), 1)
        self.assertLessEqual(len(res), 3)
        self.assertEqual(res[0]["id"], "vaswani-2017-transformer_p08_para_02")
        self.assertTrue(all(c["metadata"]["year"] == 2017 for c in res))
        with self.assertRaises(ValueError):
            r.search("   ")

    def test_08_bm25_retriever_saturation_and_ranking(self) -> None:
        """Kiểm tra Method 2 (BM25Retriever) tìm chính xác các câu có ký hiệu kỹ thuật và thuật ngữ gạch nối."""
        bm25 = BM25Retriever(self.chunks, k1=1.5, b=0.75, use_filters=True)
        res = bm25.search(
            "How does ReAct compare against Inner Monologue ReAct-IM on ALFWorld 71 vs 53?",
            top_k=3,
        )
        self.assertEqual(res[0]["id"], "yao-2022-react_p08_para_05")
        self.assertEqual(res[0]["strategy"], "bm25")

    def test_09_dense_qdrant_retriever_real_store(self) -> None:
        """Kiểm tra Method 3 (DenseQdrantRetriever) truy vấn trực tiếp Qdrant store thật từ Day 06."""
        dense = DenseQdrantRetriever(use_filters=True, auto_infer_filters=False)
        q07_text = next(q["question"] for q in self.questions if q["id"] == "q07")
        res = dense.search(q07_text, filters={"category": "rag"}, top_k=3)
        self.assertEqual(len(res), 3)
        self.assertEqual(len({c["id"] for c in res}), 3)
        self.assertEqual(res[0]["id"], "lewis-2020-rag_p04_para_00")
        self.assertTrue(all(c["metadata"]["category"] == "rag" for c in res))

    def test_10_hybrid_rrf_retriever_fusion_and_fallback(self) -> None:
        """Kiểm tra Method 4 (HybridRRFRetriever) kết hợp điểm RRF + MinMax và tự động fallback nếu filter rỗng."""
        dense = DenseQdrantRetriever(use_filters=True, auto_infer_filters=True)
        bm25 = BM25Retriever(self.chunks, use_filters=True)
        tfidf = TfidfRetriever(self.chunks, use_filters=True)
        hybrid = HybridRRFRetriever(self.chunks, dense, bm25, tfidf)

        q14_text = next(q["question"] for q in self.questions if q["id"] == "q14")
        res = hybrid.search(q14_text, top_k=3)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["id"], "lewis-2020-rag_p17_para_00")
        self.assertIn("dense_rank", res[0]["metadata"])
        self.assertIn("bm25_rank", res[0]["metadata"])

        # Kiểm tra fallback khi truyền filter không khớp bất kỳ tài liệu nào (ví dụ page=999)
        res_fallback = hybrid.search(q14_text, filters={"page": 999}, top_k=2)
        self.assertEqual(len(res_fallback), 2)
        self.assertTrue(hybrid.last_filter_info.get("fallback_unfiltered"))

    def test_11_neighbor_chunk_expansion_rescues_split_evidence(self) -> None:
        """Kiểm tra Neighboring Chunk Expansion tự động lấy thêm đoạn liền kề để cứu câu q12 và q22."""
        parsed = parse_chunk_index("yao-2022-react_p05_para_03")
        self.assertEqual(parsed, ("yao-2022-react_p05_para_", 3))

        anchor = self.chunks_by_id["yao-2022-react_p05_para_03"]
        expanded = expand_neighboring_chunks([anchor], self.chunks_by_id, max_anchors_to_expand=1, window=1)
        expanded_ids = [c["id"] for c in expanded]
        # Phải chứa cả đoạn gốc para_03 và đoạn kế tiếp para_04 (chứa ý B của câu q12)
        self.assertIn("yao-2022-react_p05_para_03", expanded_ids)
        self.assertIn("yao-2022-react_p05_para_04", expanded_ids)

    def test_12_pack_context_smart_budget_and_dedup(self) -> None:
        """Kiểm tra đóng gói ngữ cảnh khử trùng lặp và mở rộng ngân sách cho split-evidence."""
        c3 = self.chunks_by_id["yao-2022-react_p05_para_03"]
        ctx_800, ids_800 = pack_context_smart(
            [c3, c3],
            budget=800,
            all_chunks_by_id=self.chunks_by_id,
            enable_neighbor_expansion=True,
        )
        # Với budget=800, chỉ chứa được 1 chunk (para_03), chưa đủ chỗ cho para_04
        self.assertEqual(ids_800, ["yao-2022-react_p05_para_03"])
        self.assertLessEqual(len(ctx_800), 800)

        ctx_2400, ids_2400 = pack_context_smart(
            [c3],
            budget=2400,
            all_chunks_by_id=self.chunks_by_id,
            enable_neighbor_expansion=True,
        )
        self.assertIn("yao-2022-react_p05_para_03", ids_2400)
        self.assertIn("yao-2022-react_p05_para_04", ids_2400)
        self.assertLessEqual(len(ctx_2400), 2400)


if __name__ == "__main__":
    unittest.main()
