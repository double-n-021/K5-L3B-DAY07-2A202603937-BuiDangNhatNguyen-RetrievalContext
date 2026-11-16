# Báo cáo Day 7: Retrieval & Context Engineering

- **Họ và tên:** Bùi Đặng Nhật Nguyên
- **MSSV:** `2A202603937`
- **Lớp:** `K5 - L3B`
- **Repository cá nhân:** `https://github.com/double-n-021/K5-L3B-DAY07-2A202603937-BuiDangNhatNguyen-RetrievalContext`

---

## Dữ liệu và benchmark

### 1. Nguồn, quyền sử dụng, số trang và số chunk
- **Nguồn tài liệu (Corpus):** Tái sử dụng trực tiếp bộ 3 bài báo khoa học nền tảng về Trí tuệ Nhân tạo từ kho lưu trữ mở arXiv (Open Access, được phép sử dụng cho mục đích nghiên cứu và học thuật, không chứa thông tin cá nhân hay dữ liệu mật) đã xây dựng tại **Day 06 – Data Ingestion Engineering**:
  1. `vaswani-2017-transformer` (*Attention Is All You Need*, Vaswani et al., 2017): File `data/raw/vaswani_2017_transformer.pdf` (`15` trang vật lý, `10` trang nội dung chính sau khi lọc tài liệu tham khảo, `39,586` ký tự).
  2. `lewis-2020-rag` (*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, Lewis et al., 2020): File `data/raw/lewis_2020_rag.pdf` (`19` trang vật lý, `10` trang nội dung chính, `44,218` ký tự).
  3. `yao-2022-react` (*ReAct: Synergizing Reasoning and Acting in Language Models*, Yao et al., 2022): File `data/raw/yao_2022_react.pdf` (`33` trang vật lý, `17` trang nội dung chính, `52,800` ký tự).
- **Quy mô Corpus & Chiến lược Chunking kế thừa từ Day 06:**
  - Tổng số trang vật lý: **`67` trang** (`37` trang nội dung học thuật dày đặc sau tiền xử lý, vượt yêu cầu tối thiểu 30 trang; tổng cộng **`136,604` ký tự**).
  - Chiến lược chia đoạn được chọn từ Day 06: **Strategy B (`by_paragraph`, `max_size = 800` ký tự)**, tạo ra tổng cộng **`232` chunks** (`77` chunks từ `vaswani-2017-transformer`, `69` chunks từ `lewis-2020-rag`, `86` chunks từ `yao-2022-react`).
  - Thống kê độ dài chunk: trung bình **`587.7` ký tự**, trung vị **`641.0` ký tự**, ngắn nhất `85` ký tự, dài nhất `799` ký tự.
  - Định danh chunk tất định (`Deterministic ID`): `{source_id}_p{page:02d}_para_{idx:02d}` (ví dụ: `vaswani-2017-transformer_p03_para_01`).
  - Metadata chuẩn hóa trên từng chunk (`data/benchmark/chunks.json` và `starter/data/converted/chunks.json`): `source_id`, `source`, `page`, `category` (`architecture`, `rag`, `agent`), `year` (`2017`, `2020`, `2022`), `title`, `section`, `chunk_strategy`, `chunk_index_on_page`, `char_length`.

### 2. Vector store, embedding model, phiên bản và số chiều
- **Embedding Model:** `google/embeddinggemma-2` chạy local qua thư viện `sentence-transformers==6.3.0` (`torch==2.15.0`).
- **Số chiều và chuẩn hóa:** Vector **`768` chiều** (`float32`), chuẩn hóa độ dài đơn vị (`normalize_embeddings=True`, chuẩn L2 = `1.0`).
- **Vector Store:** `QdrantPersistentStore` (`qdrant-client==1.20.0`) lưu trữ bền vững trên đĩa tại `../K5-L3-Day06-DataIngestionEngineering/results/qdrant_db` (collection `day06_document_chunks`, hàm khoảng cách `Distance.COSINE`, kèm bộ đệm `embedding_cache.json` băm theo khóa SHA-256 `model_name::text`).
- **Kiểm chứng tương thích không gian vector (Vector Space Compatibility):** Truy vấn ở Day 07 được mã hóa bằng đúng lớp `MyEmbedder(model_name="google/embeddinggemma-2")` của Day 06 (chuyển trọng số sang `torch.float32` trên CPU để tăng tốc độ suy luận). Kiểm tra thực nghiệm xác nhận vector có đúng `768` chiều, chuẩn L2 = `1.000000`, và truy vấn kiểm thử trả về đúng chunk mục tiêu với điểm tương đồng Cosine `0.8588`.

### 3. Agent model, cách chạy và phân chia dev/test
- **Agent Model:** Sử dụng mô hình ngôn ngữ lớn thật **`openai/gpt-oss-120b`** thông qua **Groq API** (`PROVIDER=groq`, `MODEL=openai/gpt-oss-120b` cấu hình trong `.env`, kèm cơ chế tự động điều hòa nhịp gọi `min_interval=1.6s` và xoay vòng sang `openai/gpt-oss-20b` khi chạm trần TPM theo phút). Bộ điều khiển `GeminiAgentController` trong `src/agent_runner.py` hỗ trợ đa nhà cung cấp (`gemini`, `groq`, `openai`, `openrouter`) thông qua chuẩn OpenAI-compatible Chat Completions JSON mode (`temperature=0.0`).
- **Phân chia Benchmark mở rộng (`32` câu — vượt mức tối thiểu 24 câu của đề bài):**
  - Lưu tại `data/benchmark/questions.json` (`26` câu có đáp án dùng cho `starter/baseline.py` và `src/run_retrieval_benchmark.py`) và `data/benchmark/cases.json` (`32` cases đầy đủ dùng cho `src/run_agent_benchmark.py` và `starter/agent_metrics.py`).
  - **Tập `dev` (`22` cases):** Gồm **`19` câu có đáp án** (`q01..q15`, `q21..q24`) và **`3` câu thiếu bằng chứng / gài bẫy tiền đề** (`u01..u03`). Dùng để thử nghiệm 4 phương pháp Retrieval, đo lường tác động của Context Budget (`800` vs `1800` vs `2400`), phân tích lỗi và khóa cấu hình tốt nhất.
  - **Tập `test` (`10` cases):** Gồm **`7` câu có đáp án** (`q16..q20`, `q25..q26`) và **`3` câu thiếu bằng chứng / gài bẫy tiền đề** (`u04..u06`). Chỉ chạy đánh giá một lần sau khi đã khóa cấu hình trên `dev`.
- **Lệnh chạy tái lập kết quả:**
  ```bash
  # 1. Xuất dữ liệu chunks (232) và bộ benchmark (32 cases) từ Day 06
  python src/prepare_benchmark.py

  # 2. Chạy bộ 27 unit & integration tests
  python -m unittest discover -s tests -v

  # 3. Chạy Benchmark Retrieval & Context Engineering trên Dev (N=19) và Test (N=7)
  python src/run_retrieval_benchmark.py --split dev
  python src/run_retrieval_benchmark.py --split test

  # 4. Chạy Agentic RAG trên Dev (3 cấu hình) và Test (Baseline + Cấu hình đã chọn)
  python -u src/run_agent_benchmark.py --split dev --configs baseline exp1 exp2
  python -u src/run_agent_benchmark.py --split test --configs baseline exp2

  # 5. Chấm Audit và tổng hợp chỉ số chuẩn qua starter/agent_metrics.py
  python src/evaluate_and_audit_agent.py --split dev
  python src/evaluate_and_audit_agent.py --split test
  ```

### 4. Cách xác định câu có/thiếu evidence và tiêu chí chấm
- **Quy tắc xác định câu có bằng chứng (`answerable = true`, `26` câu):** Mỗi câu hỏi đều có trích đoạn nguyên văn (`evidence`) xuất hiện chính xác trong ít nhất một chunk thuộc đúng `source_id` và `page` của kho `232` chunks, được kiểm tra tự động bằng hàm `validate(chunks, questions)` trong `starter/baseline.py` (`hit(chunk, gold)` kiểm tra chuỗi con chuẩn hóa khoảng trắng). Nhiều câu đòi hỏi kết hợp 2 đoạn văn liên tiếp hoặc kèm điều kiện/ngoại lệ (`q02`, `q04`, `q06`, `q10`, `q12`, `q13`, `q14`, `q18`, `q22`, `q23`, `q26`).
- **Quy tắc thiết kế câu thiếu bằng chứng (`answerable = false`, `6` câu `u01..u06`):** Thiết kế theo mô hình **Bẫy tiền đề (False Premise) và Khuyết thông số (Missing Metric)** — câu hỏi nhắc tới đúng tên mô hình/bài báo trong corpus (`Transformer`, `RAG`, `ReAct`) để bộ truy xuất vẫn trả về các đoạn văn gần nhất, nhưng hỏi một thông số không hề tồn tại trong bài báo (ví dụ: chi phí thuê GPU AWS bằng USD ở `u01`, nhiệt độ softmax/top-$p$ của RAG ở `u02`, điểm HumanEval/Codex của ReAct ở `u03`, độ trễ ms/token trên V100 ở `u04`, điểm SQuAD 2.0 trên CommonCrawl ở `u05`, chi phí Bing API trên StrategyQA ở `u06`).
- **Tiêu chí chấm (`BENCHMARK.md` & `starter/agent_metrics.py`):**
  - `Hit@1`, `Evidence Hit@3`, `MRR@3`: Tỷ lệ câu hỏi có đáp án mà ít nhất 1 chunk trong Top-1 / Top-3 chứa trọn `gold.evidence` đúng nguồn và trang.
  - `Context Evidence Rate (Context Hit)` và `Complete Context`: Tỷ lệ câu có ít nhất 1 (hoặc toàn bộ) đoạn bằng chứng đích còn nằm lại trong chuỗi `context` sau khi cắt theo `budget` ký tự.
  - `Context Completeness` (`0` / `1` / `2`): `0` = không có bằng chứng đích trong context; `1` = chỉ có một phần bằng chứng cho câu đa đoạn; `2` = có đầy đủ mọi đoạn bằng chứng đích trong context.
  - `Grounded Answer` (`0` / `1` / `2` trên câu có đáp án): Chấm bằng đối chiếu nội dung thực tế (`reviews.json`), tuyệt đối không suy diễn từ việc có citation ID: `0` = sai, bịa hoặc từ chối nhầm (`status=abstain`); `1` = đúng một phần nhưng thiếu điều kiện/ngoại lệ hoặc thiếu 1 vế đa đoạn; `2` = trả lời đầy đủ mọi ý và điều kiện, bám sát 100% ngữ cảnh và có trích dẫn đúng đoạn hỗ trợ.
  - `Citation Hợp lệ (Citation Precision)`: Tỷ lệ các mã trích dẫn `[chunk_id]` mà nội dung thực tế của chunk đó chứng minh cho câu trả lời.
  - `Abstention Đúng` (trên câu `answerable=false`) và `Từ chối Nhầm (False Abstention)` (trên câu `answerable=true`).

---

## Retrieval và thử nghiệm

### 1. Bảng kết quả thực nghiệm Retrieval & Context Engineering (`Dev N=19`, `Test N=7`)

Dữ liệu trích xuất từ `artifacts/benchmark_results/dev/comparison_summary.json` và `artifacts/benchmark_results/test/comparison_summary.json`:

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline (`M1`)** | Lexical TF-IDF (`starter/baseline.py`), `top_k=3`, không filter, không rerank, FIFO `budget=800` chars | Đường cơ sở từ vựng thuần túy và ngân sách mặc định 800 ký tự | `dev` | `9/19` (`47.4%`) | `16/19` (`84.2%`) | Any: `6/19` (`31.6%`)<br>Full: `5/19` (`26.3%`) | Med: `0.63 ms`<br>Mean: `0.68 ms`<br>Cost: `\$0.00` |
| **Phương pháp 2 (`M2`)** | Lexical Okapi BM25 ($k_1=1.5, b=0.75$) + `tokenize_technical` (bảo toàn từ ghép gạch nối & lọc hư từ), `top_k=3`, `budget=800` chars | Bão hòa tần suất từ ($k_1$) và chuẩn hóa độ dài tài liệu ($b$) giúp khắc phục nhiễu lặp từ của TF-IDF | `dev` | `12/19` (`63.2%`) | `18/19` (`94.7%`) | Any: `8/19` (`42.1%`)<br>Full: `6/19` (`31.6%`) | Med: `0.37 ms`<br>Mean: `0.38 ms`<br>Cost: `\$0.00` |
| **Phương pháp 3 (`M3`)** | Dense Vector Search (`QdrantPersistentStore` + `google/embeddinggemma-2` 768d), `top_k=3`, Safe Filter (`infer_query_filters`), `budget=800` chars | Biểu diễn ngữ nghĩa dày đặc 768 chiều bắt được câu hỏi diễn đạt lại (paraphrase) tốt hơn khớp từ khóa | `dev` | `15/19` (`78.9%`) | `17/19` (`89.5%`) | Any: `11/19` (`57.9%`)<br>Full: `9/19` (`47.4%`) | Med: `78.49 ms`<br>Mean: `82.19 ms`<br>Cost: `\$0.00` (Local) |
| **Phương pháp 4 (`M4`)** | Hybrid RRF ($k_{\text{rrf}}=5$, kết hợp Dense `0.65` + BM25 `0.25` + TF-IDF `0.10` + Min-Max Score Fusion) + Safe Filter, `top_k=3`, `budget=800` chars | Dung hợp thứ hạng RRF giữa ngữ nghĩa (Dense) và ký hiệu chính xác (BM25) để đạt độ phủ Top-3 tối đa | `dev` | `15/19` (`78.9%`) | **`19/19` (`100.0%`)** | Any: `10/19` (`52.6%`)<br>Full: `8/19` (`42.1%`) | Med: `82.90 ms`<br>Mean: `83.94 ms`<br>Cost: `\$0.00` (Local) |
| **Cải thiện 1 (`Exp 1`)** | `M4` (Hybrid RRF + Safe Filter) + `pack_context_smart` (Exact/Normalized Dedup), `top_k=3`, nâng **`budget=1800` chars** | Giải quyết nút thắt "Header Overhead & Budget Starvation" khi 1 chunk học thuật kèm header đã dài ~658–824 ký tự | `dev` | `15/19` (`78.9%`) | **`19/19` (`100.0%`)** | Any: `18/19` (`94.7%`)<br>Full: `18/19` (`94.7%`) | Med: `83.06 ms`<br>Mean: `84.72 ms`<br>Chars: `1458.0` |
| **Cải thiện 2 / Đã chọn (`Exp 2`)** | `M4` (Hybrid RRF + Safe Filter) + **Two-Tier Priority Neighboring Chunk Expansion (`window=1`, `anchors=2`)**, `top_k=3`, **`budget=2400` chars** | Giữ nguyên ưu tiên Top-3 ứng viên chính (Tier 1) và nạp thêm đoạn văn liền kề `para_{i±1}` (Tier 2) để xử lý bằng chứng đứt gãy qua ranh giới đoạn | `dev` | `15/19` (`78.9%`) | **`19/19` (`100.0%`)** | **Any: `19/19` (`100.0%`)**<br>**Full: `19/19` (`100.0%`)** | Med: `84.85 ms`<br>Mean: `86.25 ms`<br>Chars: `2131.3` |
| **Cấu hình đã chọn (`Exp 2`)** | Khóa toàn bộ tham số của `Exp 2` từ `dev` (`Hybrid RRF + Safe Filter + Two-Tier Neighbor + budget=2400`), chạy 1 lần trên `test` | Kiểm tra khả năng tổng quát hóa độc lập trên tập kiểm thử `test` (`N=7` câu có đáp án) | `test` | `5/7` (`71.4%`) | `6/7` (`85.7%`) | **Any: `6/7` (`85.7%`)**<br>**Full: `5/7` (`71.4%`)** | Med: `87.89 ms`<br>Mean: `87.74 ms`<br>Chars: `2121.9` |
| **Baseline (`M1`)** | Lexical TF-IDF, `top_k=3`, không filter, FIFO `budget=800` chars chạy đối chứng trên `test` | Đường cơ sở đối chứng trên tập `test` | `test` | `6/7` (`85.7%`) | `6/7` (`85.7%`) | Any: `4/7` (`57.1%`)<br>Full: `2/7` (`28.6%`) | Med: `0.85 ms`<br>Mean: `0.81 ms`<br>Chars: `636.1` |

### 2. Phân tích quyết định kỹ thuật (What / Why / How / Trade-offs / Verification) và 2 lỗi điển hình trên Dev

1. **Quyết định 1 — Dung hợp Hybrid RRF + Suy luận Bộ lọc An toàn (`M4` so với `M1`, `M2`, `M3`):**
   - **What & Why:** Trên tập `dev`, `Dense Qdrant` (`M3`) đạt `Hit@1` cao (`15/19 = 78.9%`) nhưng trượt 2 câu ở Top-3 (`q04`, `q24`) do hai câu này chứa các ký hiệu/con số đặc thù (`6.5`, `25.8`, `100,000 steps`, `CoT-SC`) mà vector ngữ nghĩa làm trơn. Ngược lại, `Okapi BM25` (`M2`) bắt rất nhạy các con số này (`Hit@3 = 18/19`) nhưng kém hơn ở `Hit@1` (`12/19`) trước các câu diễn đạt lại.
   - **How:** Kết hợp xếp hạng nghịch đảo `RRF` ($k_{\text{rrf}}=5$) với trọng số `0.65 Dense + 0.25 BM25 + 0.10 TF-IDF` cùng điểm chuẩn hóa Min-Max và hàm `infer_query_filters(query)` (chỉ phân tích bề mặt câu hỏi của người dùng như `transformer`/`vaswani`, `rag`/`lewis`/`jeopardy`, `react`/`alfworld`/`webshop` để gợi ý `source_id`, có cơ chế tự động nới lỏng về không lọc nếu kết quả rỗng; **tuyệt đối không đọc nhãn `gold`**).
   - **Trade-offs & Verification:** Độ trễ tăng từ `0.37 ms` (BM25) lên `82.90 ms` (do truy vấn Qdrant), nhưng nâng **`Evidence Hit@3` trên Dev lên tuyệt đối `19/19 (100.0%)`** và `MRR@3` đạt `0.8684`.

2. **Lỗi Dev số 1 — Nghẽn ngân sách `budget=800` và chi phí tiêu đề định danh (Header Overhead Starvation trên `q02`, `q11`):**
   - **Nguyên nhân gốc rễ:** Chiều dài văn bản trung bình của 1 chunk Strategy B là `587.7` ký tự (nhiều chunk dài `740–799` ký tự). Khi hàm `pack_context` đóng gói, mỗi chunk được gắn thêm dòng tiêu đề `[chunk_id | source | page X]\n` dài khoảng `65–76` ký tự. Với câu `q11`, chunk đáp án chuẩn `yao-2022-react_p04_para_05` nằm ngay ở **Rank 1**, có `text` dài `747` ký tự + `67` ký tự header = **`814 > 800` ký tự**, nên bị `pack_context(budget=800)` loại bỏ hoàn toàn! Tương tự, với mọi câu mà bằng chứng nằm ở Rank 2 hoặc Rank 3 (`q02`, `q03`, `q10`, `q24`), chunk Rank 1 (~650 ký tự) đã chiếm hết chỗ khiến Rank 2 và Rank 3 không bao giờ lọt vào ngữ cảnh. Đây là lý do tại sao `M4` đạt `Hit@3 = 19/19 (100%)` nhưng `Context Hit` ở `budget=800` chỉ còn `10/19 (52.6%)`!
   - **Cách thử và khắc phục (`Exp 1`):** Nâng `budget` lên `1800` ký tự (~365 tokens, đủ chứa 2–3 chunks học thuật kèm header) và khử trùng lặp chuẩn hóa. Kết quả: `Context Hit` và `Complete Context` trên Dev tăng vọt từ `52.6% / 42.1%` lên **`18/19 (94.7%)`**.

3. **Lỗi Dev số 2 — Khuyết đoạn văn ở giữa do ranh giới chia đoạn (`q06`) và hiện tượng Neighbor chèn ép Rank 3:**
   - **Nguyên nhân gốc rễ:** Ở `Exp 1` (`budget=1800`), câu duy nhất trượt Context (`18/19`) là `q06` (*so sánh công thức RAG-Sequence và RAG-Token kèm điều kiện tương đương*). Trên trang 3 của bài báo RAG, nội dung này nằm trên 3 đoạn liên tiếp (`para_00`, `para_01`, `para_02`). Bộ truy xuất lấy được `para_00` và `para_02`, bỏ sót đoạn công thức chính `lewis-2020-rag_p03_para_01` nằm kẹp ở giữa. Tuy nhiên, khi thử chèn trực tiếp chunk lân cận ngay sau Rank 1 (Inline Insertion), các chunk lân cận của Rank 1 đã đẩy ứng viên chính ở Rank 3 ra khỏi ngân sách, làm `Context Hit` tụt xuống `15/19`.
   - **Cách thử và khắc phục (`Exp 2` - Cấu hình được chọn):** Thiết kế thuật toán **Two-Tier Priority Neighboring Chunk Expansion** (`expand_neighboring_chunks` trong `src/context_engineering.py`) với `budget=2400` ký tự:
     - **Tầng 1 (`Primary Tier`):** Giữ nguyên toàn bộ `Top-3` ứng viên chính từ Retriever ở đầu danh sách đóng gói.
     - **Tầng 2 (`Neighbor Tier`):** Phân tích cú pháp `{source}_p{page}_para_{idx}` của 2 ứng viên dẫn đầu để nối thêm `para_{idx+1}` và `para_{idx-1}` cùng trang vào phần ngân sách còn dư.
   - **Lý do chọn `Exp 2`:** Đạt **`19/19 (100.0%)` cho cả `Hit@3`, `Context Hit` và `Complete Context`** trên tập Dev (và đạt `6/7 = 85.7%` Context Hit, `5/7 = 71.4%` Complete Context trên tập Test, gấp 2.5 lần mức `28.6%` của Baseline).

---

## Agent và đánh giá

### 1. Kiến trúc Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi
- **Tools điều phối:** Agent (`starter/agentic.py` + `src/agent_runner.py`) điều phối công cụ tìm kiếm (mặc định `TfidfRetriever` cho `Baseline` và `HybridRRFRetriever` tích hợp `DenseQdrant + BM25 + TF-IDF + Safe Filter` cho `Exp 1` / `Exp 2`).
- **Số lượt gọi và Điều kiện dừng cứng (`Stopping Criteria`):**
  - Giới hạn tối đa `max_searches = 2` lượt tìm kiếm trên mỗi câu hỏi (`top_k = 3` ứng viên/lượt).
  - **Turn 0 (Initial Deterministic Search):** Khi `history` rỗng, Agent phát lệnh tìm kiếm đầu tiên với nguyên văn câu hỏi gốc để bảo toàn đầy đủ thực thể và tín hiệu lọc mà không tốn 1 lượt gọi LLM thừa.
  - **Turn 1 (Evidence Assessment & Decision):** LLM đọc `current_context` và `available_context_ids`. Nếu đã đủ bằng chứng cho mọi vế của câu hỏi, Agent trả về `action = "answer"` kèm `citations`. Nếu thiếu chi tiết phụ, Agent sinh truy vấn viết lại (`rewritten_query`) để gọi tìm kiếm lượt 2.
  - **Turn 2 / Stop Gate:** Nếu sau lượt tìm kiếm thứ 2 mà `context_ids` không có thêm chunk mới nào (do ngân sách đã đầy) hoặc đạt ngưỡng `searches >= max_searches` mà vẫn thiếu bằng chứng, vòng lặp dừng ngay lập tức và trả về `status = "abstain"`, `citations = []` (tuyệt đối không lặp vô hạn).
- **Cách chọn Context đa lượt (`make_interleaved_context_packer`):**
  - Khắc phục lỗi đói ngân sách FIFO của `pack_context` mặc định: khi có 2 lượt tìm kiếm, bộ đóng gói đan xen theo thứ hạng (**Round-Robin Rank Interleaving**: `Turn 2 Rank 1` -> `Turn 1 Rank 1` -> `Turn 2 Rank 2` -> `Turn 1 Rank 2`...) trước khi kích hoạt mở rộng lân cận 2 tầng (`Two-Tier Neighbor Expansion`), đảm bảo bằng chứng mới tìm được ở lượt 2 luôn được đưa vào ngữ cảnh.
- **Cổng kiểm duyệt Citation & Xử lý lỗi (`validate_citations` + `_normalize_answer_citations`):**
  - Mọi mã trích dẫn trong `citations` bắt buộc phải nằm trong `state["context_ids"]` và xuất hiện tường minh dưới dạng `[chunk_id]` trong câu trả lời. Nếu câu trả lời rỗng hoặc trích dẫn bất kỳ ID nào nằm ngoài `context_ids`, `run_agent()` lập tức chặn lại và chuyển trạng thái sang `status = "abstain"`.
  - Các lỗi mạng hoặc giới hạn tần suất (`HTTP 429` / `5xx`) được xử lý bằng cơ chế `Retry-After` backoff và xoay vòng model dự phòng; nếu vẫn lỗi sẽ được ghi nhận minh bạch vào `provider_errors` (không dùng mock fallback).

### 2. Bảng tổng hợp kết quả Agent trên Dev (`22` cases) và Test (`10` cases)

Tổng hợp trực tiếp bằng `starter/agent_metrics.py` và `src/evaluate_and_audit_agent.py`:

| Cấu hình / split | Tool use | Số lượt gọi (`search` / `llm`) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ (`citation_validity`) | Câu có citation (`coverage`) | Abstention đúng | Từ chối nhầm (`false_abstention`) |
|---|---|---|---|---|---|---|---|
| **`Agent_Baseline_TFIDF_b800` / `dev`** | `22/22` (`100.0%`) | Search: `1.64`<br>LLM: `1.09`<br>(`1`: 8, `2`: 14) | `0`: 14 \| `1`: 0 \| `2`: 5<br>**Mean: `0.5263 / 2.00`** (`N=19`) | `5/5` (`100.0%`) | `5/5` (`100.0%`) | `3/3` (`100.0%`) | `14/19` (`73.7%`) |
| **`Agent_Exp1_Hybrid_Budget1800` / `dev`** | `22/22` (`100.0%`) | Search: `1.14`<br>LLM: `1.14`<br>(`1`: 19, `2`: 3) | `0`: 0 \| `1`: 1 \| `2`: 18<br>**Mean: `1.9474 / 2.00`** (`N=19`) | `23/29` (`79.3%`) | `19/19` (`100.0%`) | `3/3` (`100.0%`) | `0/19` (`0.0%`) |
| **`Agent_Exp2_Hybrid_Neighbor_Budget2400` (Đã chọn) / `dev`** | `22/22` (`100.0%`) | Search: `1.14`<br>LLM: `1.14`<br>(`1`: 19, `2`: 3) | **`0`: 0 \| `1`: 0 \| `2`: 19**<br>**Mean: `2.0000 / 2.00`** (`N=19`) | **`24/28` (`85.7%`)** | **`19/19` (`100.0%`)** | **`3/3` (`100.0%`)** | **`0/19` (`0.0%`)** |
| **`Agent_Exp2_Hybrid_Neighbor_Budget2400` (Đã chọn) / `test`** | `10/10` (`100.0%`) | Search: `1.40`<br>LLM: `1.40`<br>(`1`: 6, `2`: 4) | **`0`: 1 \| `1`: 1 \| `2`: 5**<br>**Mean: `1.5714 / 2.00`** (`N=7`) | **`8/9` (`88.9%`)** | **`6/6` (`100.0%`)** | **`3/3` (`100.0%`)** | **`1/7` (`14.3%`)** |
| **`Agent_Baseline_TFIDF_b800` / `test`** | `10/10` (`100.0%`) | Search: `1.70`<br>LLM: `1.00`<br>(`1`: 3, `2`: 7) | `0`: 5 \| `1`: 0 \| `2`: 2<br>**Mean: `0.5714 / 2.00`** (`N=7`) | `2/2` (`100.0%`) | `2/2` (`100.0%`) | `3/3` (`100.0%`) | `5/7` (`71.4%`) |

*(Chỉ số bổ sung **Context Completeness 0/1/2 & Mean**: Trên `dev`, `Baseline` = `13/1/5` (`0.5789`), `Exp 1` = `1/0/18` (`1.8947`), `Exp 2` = **`0/0/19` (`2.0000`)**. Trên `test`, `Baseline` = `3/2/2` (`0.8571`), `Exp 2` = **`1/1/5` (`1.5714`)**).*

### 3. Bảng chi tiết từng Query, Tool Call, Câu trả lời, Citation và Nhãn Audit (32 Cases của Cấu hình đã chọn `Exp 2`)

| Query ID | Loại câu (`split`) | Tool call và kết quả (`searches`, `context_ids`) | Câu trả lời tóm tắt và citation | Nhãn, lý do (`reviews.json`) |
|---|---|---|---|---|
| **`q01`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `4` chunks trong context (`2083` chars), khớp `vaswani-2017-transformer_p03_para_01` | Nêu đúng $N=6$ identical layers trong Encoder và $d_{\text{model}}=512$ cho mọi sub-layers/embeddings `[vaswani-2017-transformer_p03_para_01]` | **Grade 2** \| Citation `[True]` \| Đầy đủ cả 2 thông số kiến trúc và trích dẫn chính xác đoạn nguồn trang 3. |
| **`q02`** | `dieu_kien_ngoai_le` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `4` chunks (`2392` chars), khớp `vaswani-2017-transformer_p04_para_03` | Nêu rõ với $d_k$ nhỏ hai cơ chế tương đương, với $d_k$ lớn additive vượt trội do tích vô hướng lớn đẩy softmax vào vùng đạo hàm cực nhỏ, khắc phục bằng chia $\sqrt{d_k}$ `[vaswani-2017-transformer_p04_para_03, p04_para_02]` | **Grade 2** \| Citation `[True, True]` \| Giữ trọn điều kiện so sánh theo $d_k$ và cơ chế giải thích toán học. |
| **`q03`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `4` chunks (`2262` chars), khớp `vaswani-2017-transformer_p05_para_00` | Nêu đúng $h=8$ attention heads, $d_k = d_v = d_{\text{model}}/h = 64$, tổng chi phí tính toán tương đương single-head chiều đầy đủ `[vaswani-2017-transformer_p05_para_00, p05_para_03]` | **Grade 2** \| Citation `[True, False]` \| Câu trả lời chính xác 100% nhờ chunk `p05_para_00` (`p05_para_03` là đoạn ứng dụng attention cùng trang). |
| **`q04`** | `tong_hop` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2267` chars), khớp đủ 2 đoạn `p06_para_00` và `p06_para_06` | Nêu đúng độ phức tạp per-layer $O(n^2 \cdot d)$, $O(1)$ sequential operations, và điều kiện $n < d$ để self-attention nhanh hơn recurrent `[vaswani-2017-transformer_p06_para_06, p06_para_00, p07_para_00]` | **Grade 2** \| Citation `[True, True, False]` \| Kết hợp thành công 2 đoạn văn trên trang 6. |
| **`q05`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2091` chars), khớp `vaswani-2017-transformer_p07_para_05` | Nêu đúng Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) và công thức biến thiên learning rate với `warmup_steps = 4000` `[vaswani-2017-transformer_p07_para_05]` | **Grade 2** \| Citation `[True]` \| Chính xác toàn bộ siêu tham số tối ưu hóa. |
| **`q06`** | `da_doan` (`dev`, ans=True) | `1` lượt `hybrid_rrf` + **Two-Tier Neighbor** nạp thêm `p03_para_01`; `3` chunks (`1942` chars) | Phân biệt rõ `RAG-Sequence` (cố định 1 tài liệu cho cả chuỗi) và `RAG-Token` (rút tài liệu khác nhau cho mỗi token đích), tương đương khi độ dài chuỗi đích bằng 1 `[lewis-2020-rag_p03_para_00, p03_para_01, p03_para_02]` | **Grade 2** \| Citation `[True, True, True]` \| Minh chứng điển hình của Neighbor Expansion (ở `Exp 1` thiếu `para_01` nên chỉ đạt Grade 1). |
| **`q07`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2258` chars), khớp `lewis-2020-rag_p04_para_00` | Nêu đúng DPR dùng bi-encoder `BERT_BASE` (truy vấn và tài liệu) với MIPS qua thư viện FAISS HNSW `[lewis-2020-rag_p04_para_00]` | **Grade 2** \| Citation `[True]` \| Đầy đủ thông tin kiến trúc DPR retriever. |
| **`q08`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2251` chars), khớp `lewis-2020-rag_p04_para_04` | Giải thích cơ chế marginalize qua top-$K$ tài liệu, chỉ huấn luyện query encoder + BART generator (406M params) và giữ cố định document encoder/index `[lewis-2020-rag_p04_para_04]` | **Grade 2** \| Citation `[True]` \| Trả lời đầy đủ lý do không cần cập nhật lại chỉ mục định kỳ. |
| **`q09`** | `dieu_kien_ngoai_le` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2255` chars), khớp `lewis-2020-rag_p07_para_05` | Giải thích khi thay DPR bằng BM25, BM25 vượt trội trên FEVER vì các mệnh đề FEVER xoay quanh thực thể (entity-centric) phù hợp khớp từ vựng `[lewis-2020-rag_p07_para_05]` | **Grade 2** \| Citation `[True]` \| Trích xuất chính xác ngoại lệ thực nghiệm của BM25 trên FEVER. |
| **`q10`** | `tong_hop` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2187` chars), khớp `p18_para_07` và `p19_para_01` | Nêu đúng 626M tham số khả huấn luyện và chỉ mục 21M vectors 728 chiều (~15.3B giá trị) `[lewis-2020-rag_p18_para_07, lewis-2020-rag_p19_para_01]` | **Grade 2** \| Citation `[True, True]` \| Tổng hợp chính xác từ 2 trang liên tiếp (trang 18 và 19). |
| **`q11`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2339` chars), khớp `yao-2022-react_p04_para_05` | Nêu đúng trên ALFWorld và WebShop, ReAct (1–2-shot) vượt IL/RL (10^3–10^5 episodes) lần lượt 34% và 10% tuyệt đối `[yao-2022-react_p04_para_05]` | **Grade 2** \| Citation `[True]` \| Khắc phục hoàn toàn lỗi tràn `budget=800` ở Baseline (814 chars > 800). |
| **`q12`** | `da_doan` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2248` chars), khớp `yao-2022-react_p05_para_03` và `p05_para_04` | Liệt kê đủ 3 chế độ suy nghĩ của ReAct và giải thích đối với tác vụ suy luận cần xen kẽ dày đặc, còn tác vụ điều hướng chỉ cần suy nghĩ thưa ở thời điểm quyết định `[yao-2022-react_p05_para_03, p05_para_04]` | **Grade 2** \| Citation `[True, True]` \| Bao phủ trọn vẹn 2 đoạn liên tiếp trên trang 5 của ReAct. |
| **`q13`** | `tong_hop` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2365` chars), khớp `lewis-2020-rag_p18_para_04` | Giải thích cơ chế hoán đổi chỉ mục Wikipedia 2016/2018 (82–83% accuracy cho các nguyên thủ thay đổi, 12% lệch thời gian, 68% không đổi) `[lewis-2020-rag_p18_para_04]` | **Grade 2** \| Citation `[True]` \| Đầy đủ số liệu thực nghiệm cập nhật tri thức nóng (Index Hot-Swapping). |
| **`q14`** | `dieu_kien_ngoai_le` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `4` chunks (`2364` chars), khớp `lewis-2020-rag_p17_para_00` | Phân biệt rõ Open-MSMarco/Jeopardy dùng beam size 4 + Fast Decoding, còn Open-Domain QA dùng greedy decoding + Thorough Decoding `[lewis-2020-rag_p17_para_00]` | **Grade 2** \| Citation `[True]` \| Nêu chính xác sự khác biệt chiến lược giải mã theo từng nhóm tác vụ. |
| **`q15`** | `tong_hop` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2206` chars), khớp `yao-2022-react_p15_para_02` | Giải thích hạn chế của không gian hành động WebShop và việc IL+RL có thể vượt ReAct khi có thêm 1,000+ mẫu chú thích của con người `[yao-2022-react_p15_para_02]` | **Grade 2** \| Citation `[True]` \| Giữ trọn nhận định giới hạn và điều kiện vượt trội của IL+RL. |
| **`q21`** | `truc_tiep` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`1961` chars), khớp `vaswani-2017-transformer_p08_para_02` | Nêu đúng điểm BLEU của Transformer big trên WMT 2014 EN-DE là 28.4 (vượt hơn 2.0 BLEU) và EN-FR là 41.8 trong 3.5 ngày trên 8 GPU P100 `[vaswani-2017-transformer_p08_para_02]` | **Grade 2** \| Citation `[True]` \| Chính xác toàn bộ số liệu thực nghiệm dịch máy. |
| **`q22`** | `da_doan` (`dev`, ans=True) | `1` lượt `hybrid_rrf` + **Two-Tier Neighbor**; `4` chunks (`2326` chars), khớp `p08_para_05` và `p08_para_06` | Nêu đúng 3 phương pháp regularization: Residual Dropout ($P_{\text{drop}}=0.1$, base EN-DE = 0.3) và Label Smoothing ($\epsilon_{ls}=0.1$ làm giảm perplexity nhưng tăng BLEU) `[vaswani-2017-transformer_p08_para_05, p08_para_06]` | **Grade 2** \| Citation `[True, True]` \| Ghép thành công 2 đoạn regularization liên tiếp trên trang 8. |
| **`q23`** | `tong_hop` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2184` chars), khớp `lewis-2020-rag_p06_para_01` | Nêu đúng RAG-Sequence đạt 44.5 EM trên NQ, 56.8/68.0 trên TriviaQA và giải thích khả năng sinh đáp án đúng dù không xuất hiện nguyên văn (11.8% trên NQ) `[lewis-2020-rag_p06_para_01]` | **Grade 2** \| Citation `[True]` \| Bao phủ đầy đủ số liệu và tính chất sinh trả lời ngoài trích đoạn. |
| **`q24`** | `dieu_kien_ngoai_le` (`dev`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2262` chars), khớp `yao-2022-react_p08_para_05` | Nêu đúng sau khi finetune trên 3,000 quỹ đạo mẫu, ReAct đạt 59.1% EM trên HotpotQA (vượt CoT 51.4% và CoT-SC 55.4% ở cả PaLM-8B và 62B) `[yao-2022-react_p08_para_05]` | **Grade 2** \| Citation `[True]` \| Đầy đủ điều kiện finetune và các mốc điểm so sánh. |
| **`u01`** | `unanswerable_premise_trap` (`dev`, ans=False) | `2` lượt `hybrid_rrf` (`searches=2`, viết lại truy vấn tìm AWS cost / $\text{CO}_2$); `4` chunks (`1854` chars) | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Abstain Đúng (`True`)** \| Từ chối chính xác câu bẫy chi phí thuê GPU AWS bằng USD và phát thải $\text{CO}_2$. |
| **`u02`** | `unanswerable_missing_metric` (`dev`, ans=False) | `2` lượt `hybrid_rrf` (`searches=2`, tìm temperature $\tau$ / nucleus top-$p$); `3` chunks (`1728` chars) | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Abstain Đúng (`True`)** \| Từ chối chính xác vì RAG chỉ dùng beam search / greedy decoding. |
| **`u03`** | `unanswerable_missing_benchmark` (`dev`, ans=False) | `2` lượt `hybrid_rrf` (`searches=2`, tìm HumanEval / Codex pass@1); `3` chunks (`1840` chars) | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Abstain Đúng (`True`)** \| Từ chối chính xác vì ReAct không đánh giá trên HumanEval. |
| **`q16`** | `truc_tiep` (`test`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`1961` chars), khớp `vaswani-2017-transformer_p08_para_02` | Nêu đúng mô hình base lấy trung bình 5 checkpoints cuối, mô hình big lấy trung bình 20 checkpoints cuối (cách nhau 10 phút), beam size 4, $\alpha=0.6$ `[vaswani-2017-transformer_p08_para_02]` | **Grade 2** \| Citation `[True]` \| Chính xác toàn bộ chi tiết checkpoint averaging và beam search. |
| **`q17`** | `dieu_kien_ngoai_le` (`test`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2305` chars), khớp `lewis-2020-rag_p08_para_02` | Giải thích trên Open-Domain QA hiệu năng tăng đơn điệu tới $K=50$ cho RAG-Sequence và đạt đỉnh ở $K=10$ cho RAG-Token, trên FEVER cả hai đạt đỉnh ở $K=10$ `[lewis-2020-rag_p07_para_06, p08_para_02]` | **Grade 2** \| Citation `[False, True]` \| Đầy đủ quy luật thay đổi theo số tài liệu truy xuất $K$ ở lúc suy luận. |
| **`q18`** | `tong_hop` (`test`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2365` chars), khớp `yao-2022-react_p06_para_03` (`1/2` gold items) | Phân tích đúng lỗi ảo giác của CoT (14% vs 6% của ReAct) và hạn chế tìm kiếm cứng nhắc của ReAct, đề xuất hướng kết hợp ReAct + CoT-SC `[yao-2022-react_p06_para_00, p06_para_03]` | **Grade 1** \| Citation `[True, True]` \| Trả lời đúng nguyên nhân trên trang 6 nhưng thiếu chi tiết ngưỡng chuyển đổi cụ thể ở trang 7 (`p07_para_01`). |
| **`q19`** | `da_doan` (`test`, ans=True) | `2` lượt `hybrid_rrf` (`searches=2`); `3` chunks (`2273` chars), trượt `yao-2022-react_p05_para_00` | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Grade 0 (False Abstention)** \| Retriever xếp các đoạn phân tích trang 4/6 cao hơn đoạn số liệu `p05_para_00`; Agent từ chối trung thực thay vì bịa số. |
| **`q20`** | `truc_tiep` (`test`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2232` chars), khớp `vaswani-2017-transformer_p08_para_04` | Nêu đúng WMT 2014 EN-DE gồm ~4.5M cặp câu (37,000 byte-pair tokens) và EN-FR gồm 36M câu (32,000 word-piece tokens), batch ~25,000 source/target tokens `[vaswani-2017-transformer_p08_para_04]` | **Grade 2** \| Citation `[True]` \| Chính xác 100% quy mô dữ liệu và mã hóa từ vựng. |
| **`q25`** | `dieu_kien_ngoai_le` (`test`, ans=True) | `1` lượt `hybrid_rrf`; `3` chunks (`2232` chars), khớp `vaswani-2017-transformer_p08_para_04` | Nêu đúng tốc độ huấn luyện trên 8 GPU P100: mô hình base mất 0.4s/step (100,000 steps / 12 giờ), mô hình big mất 1.0s/step (300,000 steps / 3.5 ngày) `[vaswani-2017-transformer_p08_para_04]` | **Grade 2** \| Citation `[True]` \| Đầy đủ thông số thời gian và số bước huấn luyện của cả 2 biến thể. |
| **`q26`** | `da_doan` (`test`, ans=True) | `1` lượt `hybrid_rrf` + **Two-Tier Neighbor**; `4` chunks (`2318` chars), khớp đủ `p06_para_04` và `p06_para_05` | Giải thích RAG-Token vượt RAG-Sequence trên Jeopardy vì câu hỏi Jeopardy ghép 2 mệnh đề từ 2 tài liệu khác nhau, minh họa bằng ví dụ Hemingway (*The Sun Also Rises* & *A Farewell to Arms*) `[lewis-2020-rag_p06_para_04, p06_para_05]` | **Grade 2** \| Citation `[True, True]` \| Gom đủ cả 2 đoạn văn liên tiếp trên trang 6 nhờ Two-Tier Neighbor Expansion (ở Baseline bị cắt mất `para_05` và phải abstain). |
| **`u04`** | `unanswerable_missing_metric` (`test`, ans=False) | `2` lượt `hybrid_rrf` (`searches=2`, tìm V100 ms per token latency); `3` chunks (`1942` chars) | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Abstain Đúng (`True`)** \| Từ chối chính xác vì bài báo chỉ báo cáo thời gian training trên P100, không có latency ms/token trên V100. |
| **`u05`** | `unanswerable_missing_benchmark` (`test`, ans=False) | `2` lượt `hybrid_rrf` (`searches=2`, tìm SQuAD 2.0 CommonCrawl 100M); `4` chunks (`2089` chars) | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Abstain Đúng (`True`)** \| Từ chối chính xác vì RAG chỉ dùng chỉ mục Wikipedia 21M đoạn và đánh giá trên NQ/TriviaQA/WebQuestions/CuratedTrec. |
| **`u06`** | `unanswerable_premise_trap` (`test`, ans=False) | `2` lượt `hybrid_rrf` (`searches=2`, tìm Bing Search API USD cost / StrategyQA); `4` chunks (`2021` chars) | `status = "abstain"`, `answer = "Không đủ bằng chứng để trả lời."`, `citations = []` | **Abstain Đúng (`True`)** \| Từ chối chính xác vì ReAct dùng Wikipedia API miễn phí, không dùng Bing API hay StrategyQA. |

### 4. Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí
- **Đường dẫn đầy đủ các Artifacts & Traces:**
  - Bộ dữ liệu Benchmark: `data/benchmark/chunks.json` (`232` chunks), `data/benchmark/questions.json` (`26` câu answerable), `data/benchmark/cases.json` (`32` cases).
  - Kết quả Retrieval & Context trên `dev` và `test`: `artifacts/benchmark_results/dev/comparison_summary.json` và `artifacts/benchmark_results/test/comparison_summary.json` (kèm thư mục con chứa `summary.json` và `traces.json` cho từng cấu hình `M1`, `M2`, `M3`, `M4`, `Exp1`, `Exp2`).
  - Kết quả Agentic RAG, Nhãn Audit thủ công (`reviews.json`) và Bảng tổng hợp (`agent_summary.json`):
    - Tập `dev`: `artifacts/benchmark_results/dev/agent_evaluation_comparison.json` và các thư mục `Agent_Baseline_TFIDF_b800/`, `Agent_Exp1_Hybrid_Budget1800/`, `Agent_Exp2_Hybrid_Neighbor_Budget2400/` (mỗi thư mục gồm `traces.json`, `reviews.json`, `telemetry.json`, `agent_summary.json`).
    - Tập `test`: `artifacts/benchmark_results/test/agent_evaluation_comparison.json` và các thư mục `Agent_Baseline_TFIDF_b800/`, `Agent_Exp2_Hybrid_Neighbor_Budget2400/` (kèm `official_agent_metrics.json` xuất trực tiếp từ CLI `starter/agent_metrics.py`).
- **Thống kê Thời gian, Token và Chi phí:**
  - Trên tập `dev` (`22` cases):
    - `Agent_Baseline_TFIDF_b800`: Tổng `34,625` tokens (`1,573.9` tokens/case), chi phí ước tính `\$0.008661`, độ trễ trung vị `3,073.3 ms`.
    - `Agent_Exp1_Hybrid_Budget1800`: Tổng `46,454` tokens (`2,111.5` tokens/case), chi phí ước tính `\$0.012473`, độ trễ trung vị `5,457.4 ms`.
    - `Agent_Exp2_Hybrid_Neighbor_Budget2400`: Tổng `50,722` tokens (`2,305.5` tokens/case), chi phí ước tính `\$0.012787`, độ trễ trung vị `3,695.5 ms`.
  - Trên tập `test` (`10` cases):
    - `Agent_Exp2_Hybrid_Neighbor_Budget2400`: Tổng `28,373` tokens (`2,837.3` tokens/case), chi phí ước tính `\$0.006956`, độ trễ trung vị `1,174.6 ms`.

### 5. Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn
- **So sánh Trước / Sau (Baseline vs Selected Config `Exp 2`):**
  - Trên tập `dev`, `Exp 2` nâng điểm trung bình **`Grounded Answer` từ `0.5263` lên `2.0000 / 2.00` (+280%)**, nâng `Context Completeness` từ `0.5789` lên `2.0000 / 2.00`, và xóa bỏ hoàn toàn hiện tượng từ chối nhầm (`False Abstention` giảm từ `14/19 = 73.7%` xuống `0/19 = 0.0%`), trong khi vẫn giữ vững `Abstention Accuracy = 3/3 (100%)` trên các câu hỏi gài bẫy.
  - Trên tập `test` độc lập, `Exp 2` nâng **`Grounded Answer` từ `0.5714` lên `1.5714 / 2.00` (tăng gấp 2.75 lần)**, giảm `False Abstention` từ `5/7 (71.4%)` xuống `1/7 (14.3%)`, và giữ vững `Abstention Accuracy = 3/3 (100%)`.
- **Đánh đổi kỹ thuật (Trade-offs):**
  - Mở rộng ngân sách ngữ cảnh từ `800` lên `2400` ký tự làm lượng ký tự ngữ cảnh trung bình tăng từ `593.4` lên `2144.0` ký tự (+261%), lượng token trung bình trên mỗi câu hỏi tăng khoảng `+46%` (`1573.9` -> `2305.5` tokens/case) và độ trễ tầng truy xuất tăng ~`84 ms` (do kết hợp Qdrant Dense + BM25 + TF-IDF). Tuy nhiên, chi phí tăng thêm chỉ khoảng `\$0.00019 / câu hỏi` — một mức đánh đổi cực kỳ nhỏ so với việc tăng gần gấp 3–4 lần độ chính xác có căn cứ.
- **Giới hạn hiện tại và Hướng phát triển:**
  1. **Câu hỏi tổng hợp xuyên nhiều trang không liền kề (`q18` trên Test):** Cơ chế `Two-Tier Neighbor Expansion` hiện tại mở rộng các đoạn liền kề trên **cùng một trang** (`para_{i±1}`). Với câu `q18` (bằng chứng nằm ở cuối trang 6 và đầu trang 7), nếu Turn 1 đã tìm thấy đoạn trang 6 có vẻ đủ ý, Agent có thể trả lời sớm mà chưa kích hoạt lượt tìm kiếm thứ 2 sang trang 7. Hướng khắc phục là cho phép `expand_neighboring_chunks` nối tiếp từ đoạn cuối trang $p$ sang đoạn đầu trang $p+1$ trong cùng `section`.
  2. **Đoạn văn chứa bảng số liệu dày đặc (`q19` trên Test):** Các đoạn văn trích xuất từ bảng kết quả (`HotpotQA` / `FEVER` EM) chứa nhiều con số rời rạc nên điểm tương đồng ngữ nghĩa Dense thấp hơn các đoạn văn xuôi nhận xét kết quả. Hướng nâng cấp tiếp theo là tích hợp thêm bước **Cross-Encoder Reranker** trên tập `pool_k = 25` ứng viên trước khi chọn `Top-3`.

---

## Thử thách nâng cao (nếu thực hiện)

- Trong phạm vi bài nộp cá nhân này, tôi tập trung hoàn thiện chiều sâu kỹ thuật cho **85/85 điểm phần cốt lõi** (mở rộng bộ Benchmark lên `32` câu, triển khai `4` phương pháp Retrieval khác biệt nguyên lý, xây dựng cơ chế `Two-Tier Priority Neighbor Expansion` + `Round-Robin Multi-Turn Interleaving`, đạt `27/27` unit tests tự động và xây dựng thêm ứng dụng **Interactive Web Demo Studio** tại `notes/demo_day07/`). Không đăng ký chấm phần thử thách tài liệu đa phương thức 100 trang.
