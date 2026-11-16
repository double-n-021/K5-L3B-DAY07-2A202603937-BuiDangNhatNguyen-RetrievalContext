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

### 1. Tài liệu >= 100 trang, nguồn, phiên bản, quyền sử dụng, cấu hình index và phần không xử lý được
- **Thông tin tài liệu gốc:**
  - **Tên tài liệu:** *GPT-4 Technical Report* (bao gồm báo cáo kỹ thuật chính, *GPT-4 System Card* và các Phụ lục A–G).
  - **Tác giả & Năm công bố:** OpenAI (2023).
  - **Phiên bản & Định danh:** `arXiv:2303.08774v6` — đúng **`100` trang vật lý** (`5,122 KB`).
  - **Nguồn truy cập hợp lệ & Quyền sử dụng:** Bản gốc mở hợp pháp trên arXiv (`https://arxiv.org/pdf/2303.08774.pdf`), giấy phép Open Access cho mục đích nghiên cứu học thuật và kiểm chứng tái lập. Để tuân thủ quy định không đẩy file nhị phân nặng trái phép lên Git, script `src/advanced_challenge.py` tự động tải lại đúng bản gốc vào `starter/data/advanced/gpt4_2023_technical_report.pdf` nếu chưa tồn tại trên máy người chấm.
- **Cấu hình lập chỉ mục đa phương thức (`data/advanced_benchmark/multimodal_chunks.json` & `index_manifest.json`):**
  - Toàn bộ **`81 / 100` trang nội dung thực nghiệm hữu ích** được lập chỉ mục thành **`303` đơn vị đa phương thức (`multimodal units`)**, giữ nguyên liên kết tới tài liệu gốc (`source_id = "gpt4-2023-techreport"`), số trang (`page`), loại phương thức (`modality`), định danh bảng/hình (`figure_or_table_id`) và tọa độ vùng hiển thị (`bbox = [x0, y0, x1, y1]`):
    1. **Text Chunks (`276` đơn vị):** Trích xuất toàn bộ các khối văn bản theo thứ tự bố cục trang (`by_paragraph`, gom nhóm tối đa `850` ký tự/chunk) kèm tọa độ khung bao `bbox` hợp nhất trên từng trang.
    2. **Structured Tables (`8` cụm bảng lớn bao phủ `Table 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13`):** Trích xuất cấu trúc hàng, cột, đơn vị đo (`score / 400`, `percentile`, `F1`, `% contamination`, `% degradation`, `k=1 pass count`), kèm vùng `bbox` và lưu ảnh cắt gốc độ phân giải cao (`1.5x DPI`) tại `artifacts/advanced/crops/gpt4-2023-techreport_p*_table_*.png`.
    3. **Visual Images & Diagrams (`9` đơn vị hình/sơ đồ tại các trang `9, 34, 35, 36, 37, 38, 39, 68, 99`):** Bao phủ các bài toán đầu vào thị giác (`Table 3` cáp VGA-to-Lightning 3 panel, `Table 14` biểu đồ tiêu thụ thịt Georgia & Western Asia, `Table 15` sơ đồ thanh dẫn nhiệt 1D đề thi École Polytechnique, `Table 16` ảnh là quần áo trên nóc taxi, `Table 17` sơ đồ 3 bước SFT/RM/PPO của InstructGPT, `Table 18` meme bản đồ thế giới bằng gà rán, `Table 19` truyện tranh Statistical Learning vs Neural Networks, `System Card Figure 10` sơ đồ 2 kỹ thuật Jailbreak Opposite Mode & System Message Attack, và sơ đồ luồng tác tử ReAct công cụ hóa học `Pages 99–100`), lưu kèm ảnh cắt gốc PNG tại `artifacts/advanced/crops/`.
    4. **Quantitative Charts & Plots (`10` đơn vị biểu đồ tại các trang `3, 4, 6, 8, 10, 11, 12, 14, 63, 98`):** Bao phủ `Figure 1` & `Figure 2` (Predictable Scaling), `Figure 3` (Inverse Scaling Hindsight Neglect), `Figure 4` (Exam Percentiles), `Figure 5` (Multilingual MMLU 26 ngôn ngữ), `Figure 6` (Internal Factuality 9 danh mục), `Figure 7` & `Table 4` (TruthfulQA mc1), `Figure 8` (Calibration pre-train ECE=0.007 vs PPO ECE=0.074), `Figure 9` (Incorrect behavior rate trên Sensitive & Disallowed prompts), `System Card Figure 6 & 7` (RBRM Rubric & RealToxicityPrompts), và `System Card Figure 11` (Instruction Following), lưu kèm ảnh cắt gốc PNG.
- **Ghi nhận minh bạch các trang bỏ qua và phần xử lý hạn chế (`index_manifest.json`):**
  - **`19 / 100` trang không lập chỉ mục nội dung:** Gồm các trang `15–18` (danh sách tác giả & phân công đóng góp), `19–23` (tài liệu tham khảo `[1]–[92]` của phần báo cáo chính), `70` (lời cảm ơn của System Card), `71–78` (tài liệu tham khảo `[1]–[108]` của System Card) và trang `79` (trang bìa ngăn cách phụ lục chỉ có 12 ký tự `"Appendix 79"`). Lý do: các trang này chỉ chứa tên tác giả hoặc danh mục thư mục, nếu đưa vào chỉ mục sẽ gây nhiễu từ khóa tên riêng (ví dụ `USMLE` xuất hiện trong tiêu đề tài liệu tham khảo `[26]` dù GPT-4 không công bố điểm USMLE trong báo cáo).
  - **`1` trang xử lý một phần (`Trang 98 — System Card Figure 11`):** Biểu đồ `Figure 11` được nhúng dưới dạng ảnh bitmap phẳng (`imgs=1`) chỉ có `82` ký tự vector ở dòng chú thích và không in số thập phân trên từng cột biểu đồ. Đơn vị `gpt4-2023-techreport_p098_chart_01` vẫn được lập chỉ mục kèm ảnh crop và xu hướng tổng quát, nhưng không thể đọc giá trị thập phân chính xác của từng cột nếu không có bảng số liệu gốc đi kèm.

### 2. Benchmark đa phương thức, cách chia `dev/test` và kiểm chứng với bản gốc
- **Quy mô và cấu trúc bộ Benchmark (`data/advanced_benchmark/cases.json`):**
  - Tổng cộng **`48` ca đánh giá độc lập**, chia cố định trước khi chạy thực nghiệm thành **`18` ca `dev`** (`adv_dev_01..18`) và **`30` ca `test`** (`adv_test_01..30`). Giữa `dev` và `test` hoàn toàn tách biệt câu hỏi, bảng/hình/trang trọng tâm hoặc khía cạnh kiểm tra, tuyệt đối không trùng lặp.
  - **Phân bố theo nhóm chính (mỗi ca chỉ thuộc đúng 1 nhóm chính `primary_group`):**

| Nhóm chính (`primary_group`) | Yêu cầu tối thiểu trên `test` (`ADVANCED.md`) | Số câu trên `dev` (`N = 18`) | Số câu trên `test` (`N = 30`) | Tổng số câu (`N = 48`) | Nội dung kiểm chứng tiêu biểu trên bản gốc PDF |
| --- | --- | --- | --- | --- | --- |
| **Bảng (`table`)** | `>= 6` | `3` (`adv_dev_01..03`) | **`6`** (`adv_test_01..06`) | `9` | Tra cứu và tính toán từ `Table 1` (Bar Exam, GRE, Sommelier, LeetCode), `Table 2` (MMLU, HellaSwag, ARC, WinoGrande, DROP, GSM-8K), `Table 6 & 7` (Safety refusals), `Table 8` (Base vs RLHF), `Table 9 & 10` (Contamination & Degradation). |
| **Ảnh, sơ đồ (`image_diagram`)** | `>= 5` | `3` (`adv_dev_04..06`) | **`5`** (`adv_test_07..11`) | `8` | Đọc giá trị và giải thích từ ảnh/sơ đồ: `Table 3` (VGA adapter), `Table 14` (79.84g + 69.62g = 149.46g), `Table 15` (thanh dẫn nhiệt 1D $T(x) = (T_b - T_0)(x/L) + T_0$), `Table 16` (taxi ironing), `Table 17` (sơ đồ 3 bước SFT/RM/PPO), `Table 18` (meme gà rán), `Table 19` (comic Moar Layers), `System Card Fig 10` (Opposite Mode & System Message Attack). |
| **Biểu đồ (`chart`)** | `>= 5` | `3` (`adv_dev_07..09`) | **`5`** (`adv_test_12..16`) | `8` | Đọc trục, đơn vị, chú giải và xu hướng: `Figure 1 & 2` (Bits per word, -Mean Log Pass Rate), `Figure 3` (Inverse scaling Hindsight Neglect), `Figure 5` (26 ngôn ngữ MMLU, Italian/Afrikaans 84.1% tới Marathi 66.7%/Telugu 62.0%), `Figure 6` (9 danh mục factuality +19%), `Figure 7` (TruthfulQA ~30% -> ~41% -> ~60%), `Figure 8` (ECE 0.007 vs 0.074), `Figure 9` (-82% disallowed). |
| **Multi-hop (`multi_hop`)** | `>= 4` | `3` (`adv_dev_10..12`) | **`4`** (`adv_test_17..20`) | `7` | Kết hợp bằng chứng xuyên nhiều trang/loại dữ liệu: `Fig 5` (P.8) + `Table 13` (P.33); `Table 1` (P.5) + `Fig 4` (P.6); `Table 2` (P.7) + `Table 11` (P.32); Sơ đồ ReAct Hóa học (P.99) + Trace thực thi (P.100); `Table 1` (P.5) + `Table 8` (P.28); `System Card Fig 6 & 7` (P.63) + Text (P.64); `Fig 2` (P.3) + Text (P.4). |
| **Text (`text`)** | `>= 4` | `3` (`adv_dev_13..15`) | **`4`** (`adv_test_21..24`) | `7` | Tổng hợp thông tin có điều kiện từ văn bản: Abstract & Section 2 (P.1–2), Appendix A.2–A.7 (P.24–25: lỗi giới hạn độ dài AMC 10/12, temperature 0.6, Codeforces ELO, ngày snapshot mô hình), Appendix C (P.29: thuật toán lọc 3 chuỗi con 50 ký tự), System Card (P.64 & P.66: quy trình 4 bước giảm ảo giác và Moderation API). |
| **Thiếu bằng chứng (`unanswerable`)** | `>= 6` | `3` (`adv_dev_16..18`) | **`6`** (`adv_test_25..30`) | `9` | Bẫy thông tin không công bố trong tài liệu: số tham số/GPU/chi phí điện MWh của GPT-4, kiến trúc ViT patch size, điểm USMLE Step 1/2/3 & ImageNet-1K, điểm BLEU WMT 2014 trong Table 2, lương theo giờ của chuyên gia chấm bài/red-teaming, giá tiền USD & mã SKU mua chất hóa học trên ZINC, độ chính xác MMLU tiếng Việt/Tagalog/Amharic. |
| **Tổng cộng** | **`>= 30`** (`24+6`) | **`18`** (`15+3`) | **`30`** (`24+6`) | **`48`** | Đối chiếu trực tiếp 100% với trang, bảng, hình và vùng `bbox` trên PDF gốc. |

### 3. Pipeline truy xuất đa phương thức, thực nghiệm trên `dev` và quyết định khóa cấu hình
- **Thiết kế Pipeline Đa phương thức (`src/advanced_challenge.py`):**
  - Để chứng minh vai trò quyết định của việc lập chỉ mục đa phương thức và kỹ thuật đóng gói ngữ cảnh, tôi đối sánh **3 cấu hình** trên tập `dev`:
    1. **`Adv_Baseline_TextOnly_TFIDF_b1200` (Baseline):** Chỉ lập chỉ mục trên các đoạn văn bản thuần (`276` text chunks, bỏ qua các đơn vị bảng/biểu đồ/hình có cấu trúc), dùng `TfidfRetriever` với ngân sách `budget = 1200` ký tự và không mở rộng lân cận.
    2. **`Adv_Exp1_Multimodal_BM25_b2000` (Thử nghiệm 1):** Lập chỉ mục trên toàn bộ `303` đơn vị đa phương thức (text + table + image_diagram + chart), dùng `BM25Retriever` với `budget = 2000` ký tự nhưng chưa có hợp nhất RRF và chưa mở rộng đoạn liền kề.
    3. **`Adv_Selected_Multimodal_Hybrid_b3200` (Cấu hình Đề xuất — Khóa để chạy `test`):**
       - **Multimodal Hybrid RRF:** Kết hợp điểm xếp hạng Reciprocal Rank Fusion ($k_{rrf}=30$) giữa `BM25` ($w=1.25$) và `TF-IDF` ($w=1.0$) trên toàn bộ `303` đơn vị đa phương thức, cộng thêm **Modality & Reference Tag Boosting** (tự động nhận diện các tham chiếu `Table X`, `Figure Y`, `Page Z` hoặc từ khóa trực quan `chart/plot/diagram/panel/comic/meme/axis` trong câu hỏi để ưu tiên đúng đơn vị bảng/biểu đồ/hình tương ứng).
       - **Cross-Modal & Same-Page Neighbor Expansion (`budget = 3200` ký tự):** Đóng gói ưu tiên 3 hạt nhân truy xuất đứng đầu (đảm bảo câu hỏi multi-hop giữ đủ bằng chứng từ cả 2 trang/bảng/biểu đồ), sau đó tự động mở rộng các đoạn văn liền kề cùng trang (`text_{k±1}`) hoặc liên trang (ví dụ nối sơ đồ ReAct trang 99 với trace trang 100, biểu đồ Figure 2 trang 3 với đoạn giải thích 6 difficulty buckets ở trang 4, biểu đồ Figure 6–7 trang 63 với số liệu human preference ở trang 64). Mỗi khối ngữ cảnh được đóng nhãn nguồn đầy đủ: `[chunk_id | page=P | modality=M | ref=Table/Figure | bbox=[x0, y0, x1, y1]]`.
- **Bảng kết quả so sánh trên tập `dev` (`18` ca = `15` có đáp án + `3` thiếu bằng chứng) và `test` (`30` ca = `24` có đáp án + `6` thiếu bằng chứng):**

| Cấu hình Pipeline | Tập | `Hit@1` (`n/N`) | `Hit@5` (`n/N`) | Truy hồi đủ mọi mảnh Gold (`Full Evidence`) | Ngữ cảnh cuối chứa đủ mọi mảnh Gold (`Complete Context`) | Ký tự ngữ cảnh TB (`Avg Chars`) | Tỷ lệ Agent Đạt toàn diện (`Pass Rate n/N`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Adv_Baseline_TextOnly_TFIDF_b1200` | `dev` | `1/15` (`6.7%`) | `3/15` (`20.0%`) | `2/15` (`13.3%`) | `0/15` (`0.0%`) | `1050.9` | `3/18` (`16.7%`) |
| `Adv_Exp1_Multimodal_BM25_b2000` | `dev` | `10/15` (`66.7%`) | `14/15` (`93.3%`) | `13/15` (`86.7%`) | `9/15` (`60.0%`) | `1782.7` | — *(Loại ở tầng Context)* |
| **`Adv_Selected_Multimodal_Hybrid_b3200` (Khóa)** | **`dev`** | **`14/15` (`93.3%`)** | **`15/15` (`100.0%`)** | **`15/15` (`100.0%`)** | **`15/15` (`100.0%`)** | **`2972.1`** | **`17/18` (`94.4%`)** |
| `Adv_Baseline_TextOnly_TFIDF_b1200` | `test` | `5/24` (`20.8%`) | `7/24` (`29.2%`) | `2/24` (`8.3%`) | `0/24` (`0.0%`) | `1044.3` | — *(Baseline đối chiếu)* |
| `Adv_Exp1_Multimodal_BM25_b2000` | `test` | `22/24` (`91.7%`) | `24/24` (`100.0%`) | `22/24` (`91.7%`) | `17/24` (`70.8%`) | `1805.3` | — *(Đối chiếu tầng Context)* |
| **`Adv_Selected_Multimodal_Hybrid_b3200` (Khóa)** | **`test`** | **`23/24` (`95.8%`)** | **`24/24` (`100.0%`)** | **`24/24` (`100.0%`)** | **`24/24` (`100.0%`)** | **`3018.2`** | **`30/30` (`100.0%`)** |

- **Phân tích quyết định khóa cấu hình từ `dev`:**
  - Ở `Adv_Baseline_TextOnly_TFIDF_b1200`, việc thiếu các đơn vị bảng/biểu đồ/hình có cấu trúc khiến `Complete Context` trên `dev` bằng `0/15 (0.0%)`; Agent chỉ đạt `3/18 (16.7%)` nhờ từ chối trả lời trên 3 câu `unanswerable`.
  - Khi bổ sung chỉ mục đa phương thức ở `Adv_Exp1_Multimodal_BM25_b2000`, `Full Evidence Retrieval` tăng vọt lên `13/15 (86.7%)`, nhưng ngân sách `2000` ký tự và việc thiếu mở rộng đoạn văn liền kề khiến `Complete Context` chỉ đạt `9/15 (60.0%)`.
  - Ở **`Adv_Selected_Multimodal_Hybrid_b3200`**, cả `Full Evidence Retrieval` và `Complete Context` trên `dev` đều đạt tuyệt đối **`15/15 (100.0%)`**, giúp tỷ lệ Agent Đạt nhị phân trên `dev` tăng từ `3/18 (16.7%)` lên **`17/18 (94.4%)`**. Ca không đạt duy nhất trên `dev` là `adv_dev_04` (trang 9): ngữ cảnh đã chứa cả `gpt4-2023-techreport_p009_image_diagram_01` và `gpt4-2023-techreport_p009_text_01`, mô hình mô tả chính xác cả 3 panel của cáp VGA-to-Lightning nhưng chỉ trích dẫn mã `[gpt4-2023-techreport_p009_text_01]` nên bị bộ chấm nghiêm ngặt đánh trượt lỗi thiếu trích dẫn mã `image_diagram`. Tôi giữ nguyên quy tắc chấm nghiêm ngặt này và khóa `Adv_Selected_Multimodal_Hybrid_b3200` để chạy một lần trên toàn bộ `30` ca `test`.

### 4. Kết quả trên toàn bộ `test` theo từng nhóm (`30` Ca) và Kiểm toán Từng Ca
- **Bảng tổng hợp kết quả chính thức trên tập `test` (`N = 30` ca) theo đúng cấu trúc của `ADVANCED.md`:**

| Nhóm đánh giá | Số câu đạt (`n/N`) | Tỷ lệ đạt (`%`) | Số lỗi hệ thống / Timeout | Ghi chú đối chiếu với bản gốc PDF |
| --- | --- | --- | --- | --- |
| **Bảng (`table`)** | **`6 / 6`** | **`100.0%`** | `0` | Trả lời đúng toàn bộ số liệu, đơn vị, phân vị và tỷ lệ nhiễm dữ liệu từ `Table 1, 2, 6, 7, 9, 10`, trích dẫn đúng mã bảng. |
| **Ảnh, sơ đồ (`image_diagram`)** | **`5 / 5`** | **`100.0%`** | `0` | Đọc đúng số liệu biểu đồ trong ảnh (`149.46g`), công thức vật lý thanh dẫn nhiệt 1D, sơ đồ 3 bước InstructGPT, comic Moar Layers và sơ đồ 2 kỹ thuật Jailbreak. |
| **Biểu đồ (`chart`)** | **`5 / 5`** | **`100.0%`** | `0` | Đọc đúng trục, đơn vị và số liệu trên `Figure 2, 5, 6, 7, 9` (`24/26` ngôn ngữ, `+19` điểm %, `30% -> 41% -> 60%`, giảm `82%` disallowed). |
| **Multi-hop (`multi_hop`)** | **`4 / 4`** | **`100.0%`** | `0` | Kết hợp và trích dẫn đầy đủ cả 2 nguồn bằng chứng trên mỗi câu (P.99+P.100, P.5+P.28, P.63+P.64, P.3+P.4). |
| **Text (`text`)** | **`4 / 4`** | **`100.0%`** | `0` | Tổng hợp đầy đủ cả 2 đoạn văn liên tiếp trên các trang `24, 25, 64, 66`, nêu đúng mọi điều kiện và mốc thời gian. |
| **Thiếu bằng chứng (`unanswerable`)** | **`6 / 6`** | **`100.0%`** | `0` | Từ chối trả lời (`status = "abstain"`) chính xác 100%, không suy đoán hay bịa đặt bất kỳ thông số nào ngoài tài liệu. |
| **Toàn bộ (`overall`)** | **`30 / 30`** | **`100.0%`** | **`0`** | **Vượt ngưỡng `>= 90%` (`27/30`) của Thử thách Nâng cao.** |

- **Bảng kiểm toán thủ công từng câu trên tập `test` (`30/30` ca `adv_test_01..30` — bản đầy đủ lưu tại `artifacts/advanced/test_manual_audit_30_cases.md`):**

| ID | Nhóm chính | Nguồn & Vùng `bbox` trên PDF gốc | Citations của Agent | Tóm tắt câu trả lời / Hành vi Abstain & Đối chiếu bản gốc | Nhãn |
| --- | --- | --- | --- | --- | --- |
| `adv_test_01` | `table` | P.5 `Table 1` `[50,60,560,720]` | `[p005_table_01, p005_text_01]` | Nêu đúng GRE Quantitative (`163/170 ~80th` vs `147/170 ~25th`), GRE Verbal (`169/170 ~99th` vs `154/170 ~63rd`), GRE Writing (`4/6 ~54th` cho cả hai). | **Đạt** |
| `adv_test_02` | `table` | P.5 `Table 1` `[50,60,560,720]` | `[p005_table_01]` | Nêu đúng 3 cấp Sommelier (`92% vs 80%`, `86% vs 58%`, `77% vs 46%`) và LeetCode `k=1` (`31/41, 21/39, 3/20` vs `12/41, 8/39, 0/20`). | **Đạt** |
| `adv_test_03` | `table` | P.7 `Table 2` `[50,55,560,480]` | `[p007_table_01]` | Nêu đúng điểm và số shot: MMLU 5-shot (`86.4% vs 70.0%`), HellaSwag 10-shot (`95.3% vs 85.5%`), ARC 25-shot (`96.3% vs 85.2%`), WinoGrande 5-shot (`87.5% vs 81.6%`), GSM-8K 5-shot (`92.0% vs 57.1%`). | **Đạt** |
| `adv_test_04` | `table` | P.13 `Table 6 & 7` `[55,55,555,520]` | `[p013_table_01, p013_text_01]` | Đối chiếu đúng Early GPT-4 (hướng dẫn chế tạo bom ở Table 6 và từ chối quá đà câu hỏi mua thuốc lá giá rẻ ở Table 7) với Latest GPT-4 (từ chối câu hỏi bom và liệt kê 4 lựa chọn hợp pháp kèm cảnh báo sức khỏe ở Table 7). | **Đạt** |
| `adv_test_05` | `table` | P.30 `Table 9` `[50,55,560,730]` | `[p030_table_01]` | Nêu đúng tỷ lệ contamination: Uniform Bar Exam `0%` (`298/400`), LSAT `39%` (`163` toàn bộ vs `169` non-contaminated), GRE Quantitative `35%` (`163/170` vs `165/170` non-contaminated). | **Đạt** |
| `adv_test_06` | `table` | P.31 `Table 10` `[50,55,560,730]` | `[p031_table_01]` | Nêu đúng AP English Literature (`55` câu, `81.82%` contam, `60.00%` non-contam, `75.56%` contam-only, `-17.50%` degradation) và MKSAP (`1080` câu, `18.52%` contam, `75.11%` non-contam, `73.00%` contam-only, `+0.52%` degradation). | **Đạt** |
| `adv_test_07` | `image_diagram` | P.34 `Table 14` `[55,55,555,690]` | `[p034_image_diagram_01, p034_text_01, p034_text_02]` | Đọc đúng từ ảnh biểu đồ ngang: Georgia = `79.84` grams per person per day, Western Asia = `69.62` grams per person per day, tổng = `149.46` grams per person per day. | **Đạt** |
| `adv_test_08` | `image_diagram` | P.35 `Table 15` `[55,55,555,700]` | `[p035_image_diagram_01, p035_text_01]` | Mô tả đúng thanh dẫn nhiệt 1D chiều dài $L$ dọc trục $x$ với điều kiện biên $T(0)=T_0, T(L)=T_b$ và nghiệm phân bố nhiệt độ tuyến tính $T(x) = (T_b - T_0)(x/L) + T_0$. | **Đạt** |
| `adv_test_09` | `image_diagram` | P.37 `Table 17` `[55,55,555,720]` | `[p037_image_diagram_01, p037_text_04, ...]` | Giải thích đúng 3 bước trong sơ đồ Figure 2 bài báo InstructGPT: (1) Supervised fine-tuning (SFT), (2) Reward model (RM) training, (3) Reinforcement learning via PPO. | **Đạt** |
| `adv_test_10` | `image_diagram` | P.39 `Table 19` `[55,55,555,650]` | `[p039_image_diagram_01, p039_text_02, p039_text_01]` | Đối조 đúng khung trên Statistical Learning (minimizing structural risk, reworking loss function, soft margin) với khung dưới Neural Networks ("stack more layers"). | **Đạt** |
| `adv_test_11` | `image_diagram` | P.68 `System Card Fig 10` `[55,55,555,710]` | `[p068_image_diagram_01, p068_text_03]` | Nêu đúng 2 kỹ thuật: "Prompting with Opposite Mode" (`[ChatGPT]` vs `[AntiGPT]`) bên trái và "System Message Attack" bên phải (được ghi chú là một trong những cách phá mô hình hiệu quả nhất). | **Đạt** |
| `adv_test_12` | `chart` | P.3 `Figure 2` `[45,55,565,680]` | `[p003_chart_01, p004_text_02]` | Nêu đúng trục y là `-Mean Log Pass Rate` (0 đến 5), trục x là compute huấn luyện chuẩn hóa theo `GPT-4 = 1`, trên `23` bài toán thuộc `3rd easiest difficulty bucket` của HumanEval. | **Đạt** |
| `adv_test_13` | `chart` | P.8 `Figure 5` `[45,55,565,450]` | `[p008_chart_01]` | Nêu đúng GPT-4 vượt GPT-3.5 English (`70.1%`) trên `24/26` ngôn ngữ; cao nhất là Italian (`84.1%`) và Afrikaans (`84.1%`); 2 ngôn ngữ dưới `70.1%` là Marathi (`66.7%`) và Telugu (`62.0%`). | **Đạt** |
| `adv_test_14` | `chart` | P.10 `Figure 6` `[60,310,550,610]` | `[p010_chart_01, p010_text_02]` | Nêu đủ 9 danh mục trên trục x (`learning, technology, writing, history, math, science, recommendation, code, business`), đối sánh với `chatgpt-v2, chatgpt-v3, chatgpt-v4`, vượt GPT-3.5 mới nhất `19` điểm phần trăm. | **Đạt** |
| `adv_test_15` | `chart` | P.11 `Figure 7 & Table 4` `[55,65,555,650]` | `[p011_chart_01, p011_text_01]` | Nêu đúng độ chính xác TruthfulQA mc1 tăng từ `~30%` (gpt-4-base 0-shot) lên `~41%` (5-shot) và `~60%` (sau RLHF), đồng thời nêu câu trả lời sai `"Presley"` thay vì `"Perkins"` (Elvis Perkins) ở Table 4. | **Đạt** |
| `adv_test_16` | `chart` | P.14 `Figure 9` `[95,55,515,330]` | `[p014_chart_01, p013_text_05]` | Nêu đúng 2 nhóm `Sensitive Prompts` và `Disallowed Prompts` trên 3 mô hình (`text-davinci-003`, `gpt-3.5-turbo`, `gpt-4`) và mức giảm `82%` phản hồi nội dung bị cấm so với GPT-3.5. | **Đạt** |
| `adv_test_17` | `multi_hop` | P.99 Diagram + P.100 Trace | `[p100_text_01, p100_text_02]` | Kết hợp sơ đồ ReAct và trace: sau khi thất bại sửa đổi Dasatinib, `Literature Answer` trả về `imatinib, nilotinib, bosutinib`; tác tử sửa đổi `imatinib` thành dẫn xuất mới không có bằng sáng chế và đặt mua từ `ZINC`. | **Đạt** |
| `adv_test_18` | `multi_hop` | P.5 `Table 1` + P.28 `Table 8` | `[p005_table_01, p028_table_01]` | Kết hợp điểm USNCO 2022 ở Table 1 (`36/60` có vision, `38/60` không vision, `24/60` ở GPT-3.5) với tỷ lệ phần trăm ở Table 8 (`51.7%` base vs `63.3%` post-RLHF). | **Đạt** |
| `adv_test_19` | `multi_hop` | P.63 `Fig 6 & 7` + P.64 Text | `[p063_chart_01, p064_text_01]` | Nêu đủ 3 câu hỏi Yes/No của rubric RBRM ở Fig 6, mức giảm độc hại RealToxicityPrompts từ `6.48%` xuống `0.73%`, và tỷ lệ người chấm ưu tiên GPT-4-launch (`70.2%` so với GPT-3.5 RLHF, `61.1%` so với GPT-3.5-Turbo RLHF). | **Đạt** |
| `adv_test_20` | `multi_hop` | P.3 `Figure 2` + P.4 Text | `[p003_chart_01, p004_text_02]` | Kết hợp P.4 và P.3: toàn bộ trừ `15` bài khó nhất của HumanEval được chia thành `6` difficulty buckets; Figure 2 vẽ `3rd easiest bucket`, còn GPT-4 thấp hơn dự đoán trên `easiest bucket`. | **Đạt** |
| `adv_test_21` | `text` | P.24 `p024_text_03` + `p024_text_04` | `[p024_text_03, p024_text_04]` | Nêu đúng lỗi giới hạn độ dài phản hồi (`response length bug`) trên đề thi AMC 10 & AMC 12 đã được sửa và chạy lại, cùng nhiệt độ lấy mẫu `temperature 0.6` cho câu hỏi tự luận. | **Đạt** |
| `adv_test_22` | `text` | P.25 `p025_text_04` + `p025_text_05` | `[p025_text_04, p025_text_05]` | Nêu đúng cách tính ELO Codeforces qua `10` kỳ thi (`~6` bài/kỳ, `10` lần thử/bài) và 3 mốc snapshot: `March 1, 2023` (MCQ), `February 23, 2023` (free-response), `December 16, 2022` (USABO). | **Đạt** |
| `adv_test_23` | `text` | P.64 `p064_text_03` + `p064_text_04` | `[p064_text_03, p064_text_04]` | Trình bày đúng quy trình 4 bước dùng chính GPT-4 phát hiện và viết lại câu trả lời không còn ảo giác để đưa vào tập huấn luyện RM, nâng TruthfulQA từ `~30%` lên `~60%`. | **Đạt** |
| `adv_test_24` | `text` | P.66 `p066_text_02` + `p066_text_03` | `[p066_text_02, p066_text_03]` | Nêu đúng `Moderation API endpoint` và 2 vai trò của GPT-4: (1) tăng tốc xây dựng taxonomy phân loại nội dung bằng cách tìm lỗ hổng chính sách, (2) hỗ trợ gán nhãn dữ liệu huấn luyện bộ phân loại. | **Đạt** |
| `adv_test_25` | `unanswerable` | Bẫy chi phí USD & điện năng MWh | `[] (Abstain)` | Từ chối đúng lúc (`abstained = True`): Section 2 không công bố chi phí đám mây USD hay điện năng MWh. | **Đạt** |
| `adv_test_26` | `unanswerable` | Bẫy ViT patch size & cross-attention | `[] (Abstain)` | Từ chối đúng lúc (`abstained = True`): Báo cáo không công bố kích thước patch ảnh hay số lớp cross-attention của bộ mã hóa thị giác. | **Đạt** |
| `adv_test_27` | `unanswerable` | Bẫy điểm BLEU WMT 2014 ở Table 2 | `[] (Abstain)` | Từ chối đúng lúc (`abstained = True`): Table 2 không đánh giá dịch máy WMT 2014 EN-DE/EN-FR. | **Đạt** |
| `adv_test_28` | `unanswerable` | Bẫy lương theo giờ của giám khảo chấm bài | `[] (Abstain)` | Từ chối đúng lúc (`abstained = True`): Appendix A trang 24 không tiết lộ thù lao USD/giờ của nhà thầu chấm tự luận. | **Đạt** |
| `adv_test_29` | `unanswerable` | Bẫy giá tiền USD & SKU của ZINC (P.100) | `[] (Abstain)` | Từ chối đúng lúc (`abstained = True`): Trace trang 100 chỉ ghi `"Purchased from ZINC"`, không có giá tiền, mã SKU hay số ngày giao hàng. | **Đạt** |
| `adv_test_30` | `unanswerable` | Bẫy tiếng Việt/Tagalog/Amharic trên MMLU | `[] (Abstain)` | Từ chối đúng lúc (`abstained = True`): Figure 5 chỉ kiểm tra 26 ngôn ngữ (từ Italian tới Telugu), không có tiếng Việt, Tagalog hay Amharic. | **Đạt** |

### 5. Đường dẫn trace, nhãn, chi phí, thời gian, lỗi và giới hạn
- **Đường dẫn đầy đủ của mã nguồn, chỉ mục, benchmark và kết quả kiểm toán:**
  - Mã nguồn pipeline đa phương thức: `src/advanced_challenge.py` (chạy bằng lệnh `python src/advanced_challenge.py --step all`).
  - Chỉ mục đa phương thức & thống kê trang bỏ qua: `data/advanced_benchmark/multimodal_chunks.json` (`303` đơn vị) và `data/advanced_benchmark/index_manifest.json`.
  - Ảnh cắt gốc (`crops`) của toàn bộ bảng, hình, sơ đồ và biểu đồ: `artifacts/advanced/crops/*.png` (`21` file ảnh PNG gắn tọa độ `bbox`).
  - Bộ câu hỏi Benchmark `48` ca (`18 dev` + `30 test`): `data/advanced_benchmark/cases.json`.
  - Kết quả đối sánh tầng Retrieval & Context: `artifacts/advanced/retrieval_comparison.json`.
  - Kết quả tổng hợp và nhật ký kiểm toán từng ca:
    - Tập `dev`: `artifacts/advanced/dev_Adv_Baseline_TextOnly_TFIDF_b1200_summary.json`, `artifacts/advanced/dev_Adv_Selected_Multimodal_Hybrid_b3200_summary.json`, và thư mục trace từng câu `artifacts/advanced/traces/dev/`.
    - Tập `test`: `artifacts/advanced/test_Adv_Selected_Multimodal_Hybrid_b3200_summary.json`, bảng kiểm toán chi tiết 30 câu `artifacts/advanced/test_manual_audit_30_cases.md`, và thư mục trace từng câu `artifacts/advanced/traces/test/Adv_Selected_Multimodal_Hybrid_b3200/adv_test_01..30.json`.
- **Thống kê Lỗi, Thời gian, Token và Chi phí trên tập `test` (`30` ca):**
  - **Số lỗi hệ thống / Timeout / Provider error:** `0 / 30` (`0%`).
  - **Tổng lượng token tiêu thụ:** `39,046` tokens trên `30` ca (trung bình **`1,301.5` tokens / ca**).
  - **Tổng chi phí suy luận ước tính:** **`\$0.008123`** cho toàn bộ `30` ca `test` (trung bình **`\$0.000271 / ca`**).
  - **Độ trễ trung bình mỗi ca (`avg_latency_ms`):** `9,256.5 ms / ca` (bao gồm cả thời gian giãn cách tránh giới hạn tốc độ API).
- **Phân tích lỗi trên `dev` và giới hạn của hệ thống đa phương thức:**
  - **Phân tích lỗi đồng xuất hiện hai phương thức trên cùng một trang (`adv_dev_04` trên tập `dev`):** Trên trang 9 (`Table 3` — cáp VGA-to-Lightning), phần mô tả các panel vừa nằm trong đơn vị trực quan `gpt4-2023-techreport_p009_image_diagram_01`, vừa xuất hiện trong khối văn bản trích xuất tự động `gpt4-2023-techreport_p009_text_01`. Khi cả hai cùng nằm trong ngữ cảnh, mô hình trả lời đúng toàn bộ chi tiết nhưng chỉ gắn thẻ `[gpt4-2023-techreport_p009_text_01]`. Hướng khắc phục triệt để trong tương lai là hợp nhất (deduplicate) các đoạn text trùng lắp nằm trọn bên trong khung bao `bbox` của một đơn vị `image_diagram` hoặc `table` đã được trích xuất cấu trúc.
  - **Giới hạn với biểu đồ bitmap không có nhãn số (`Trang 98 — System Card Figure 11`):** Với các biểu đồ chỉ nhúng dưới dạng ảnh phẳng mà tác giả không in số cụ thể trên đỉnh cột và không kèm bảng phụ lục, hệ thống chỉ có thể trích xuất xu hướng so sánh giữa các mô hình (`GPT-4-launch > GPT-3.5-Turbo > GPT-3.5`) chứ không thể tự suy diễn ra con số thập phân chính xác nếu không có dữ liệu gốc.
