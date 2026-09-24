# Báo cáo: điểm mơ hồ / mâu thuẫn trong docs Unsloth

Ngày đọc nguồn: 2026-09-24. File này dành cho việc review, không nằm trong website (VitePress chỉ build thư mục `docs/`) và chưa được commit.

- **Thao tác**: ảnh hưởng tới việc người đọc làm theo, gồm giá trị mặc định, VRAM, phần cứng, lệnh cài/chạy, cổng, bảo mật, code không chạy được.
- **Mô tả**: chỉ lệch về lời mô tả, gồm typo ngoài lệnh, tên gọi, link hỏng, số marketing không có điều kiện đo, thông tin cũ.
- **Trên web**: "Có" nghĩa là điểm đã hiện trên trang, bằng hộp `Docs chưa thống nhất`, hộp info/warning, nhãn [Nhận định], hoặc ghi chú "cần kiểm tra lại".

## Tổng hợp

| Mức độ | Tổng | Đã đưa lên web | Chưa đưa lên |
| --- | --- | --- | --- |
| Thao tác | 138 | 136 | 2 |
| Mô tả | 77 | 33 | 44 |
| **Tổng** | **215** | **169** | **46** |

## A. Ảnh hưởng thao tác

### tong-quan, lo-trinh-hoc

| # | Điểm | Trên web |
| --- | --- | --- |
| A1 | Server-side tools khi mở mạng: api "unsloth run bind 0.0.0.0 → tắt" vs README "on by default" | Có |
| A2 | Lệnh mở Studio: `unsloth studio -H 0.0.0.0 -p 8888` vs README `unsloth studio` vs dev `unsloth studio -p 8888`; không có cổng mặc định | Có |
| A3 | Bind mặc định: docs đưa -H 0.0.0.0 vào lệnh cơ bản vs README khuyên -H 127.0.0.1 | Có |
| A4 | Tạo mật khẩu: studio/start trên trình duyệt vs README hỏi tại terminal khi mở mạng | Có |
| A5 | Cổng API "8000 or 8888"; ví dụ 8888; README Docker gọi 8888 là cổng JupyterLab | Có |
| A6 | Định dạng file tải lên: studio/start PDF/DOCX/JSONL/JSON/CSV/Parquet; new/studio 3 danh sách khác nhau (TXT, YAML); desktop PDF/CSV/JSON; docs.md PDF/CSV/DOCX | Có |

### tong-quan, lo-trinh-hoc, index

| # | Điểm | Trên web |
| --- | --- | --- |
| A7 | Train theo phần cứng: new/studio "on NVIDIA" vs cùng trang macOS train được; README AMD train được; không nguồn nói rõ Intel/CPU | Có |

### tong-quan

| # | Điểm | Trên web |
| --- | --- | --- |
| A8 | /v1/responses có trong đoạn giới thiệu nhưng không trong bảng Endpoints | Có |
| A9 | cli.py (new-ui-prototype/) vs lệnh unsloth studio/run/start | Có |

### lo-trinh-hoc

| # | Điểm | Trên web |
| --- | --- | --- |
| A10 | Bật LAN: desktop nút Start thẻ "LAN access" hoặc -H 0.0.0.0; README Settings > API keys > LAN access; api -H 0.0.0.0 -p 8888 | Có |

### cai-dat

| # | Điểm | Trên web |
| --- | --- | --- |
| A11 | Mac có train được: requirements "ALL supported"; Studio "Like CPU… MLX training now works" & "training and all features"; trang Mac "run and train MLX" | Có |
| A12 | Core trên Mac: OS chỉ Linux/Windows, "Apple/Silicon/MLX is in the works" | Có |
| A13 | Docker chưa hỗ trợ Mac ("working on Mac compatibility") | Có |
| A14 | Gói cần cho Mac: requirements Homebrew/git/cmake/openssl/Python vs Studio chỉ macOS 12+ & môi trường Python | Có |
| A15 | Studio train AMD/Intel: "works on … AMD, Intel" vs "use Core to train on AMD and Intel" vs yêu cầu chỉ ghi NVIDIA | Có |
| A16 | GPU AMD: "RDNA 3/3.5/4 (RX 6000–9000)" vs Docker "RDNA1+ và CDNA"; ROCm 6.0+ vs image ROCm 7.2 | Có |
| A17 | ROCm 7.2/7.3+: index rocm7.2 vs "7.2+ → rocm7.1, no wheels" | Có |
| A18 | bitsandbytes ≤0.49.2 lỗi NaN nhưng fallback >=0.49.1 | Có |
| A19 | Extra AMD: unsloth[rocm72-torch291] (Python 3.12 Windows) vs unsloth[amd] | Có |
| A20 | Nút "Download for Linux" trang AMD trỏ URL Windows | Có |
| A21 | Python: 3.11–<3.14 / 3.11–3.13 / conda 3.12 / uv 3.13 / venv 3.10–3.13 / AMD 3.10 / Intel 3.10+ | Có |
| A22 | pip nâng cao: văn bản cu118/121/124, torch ≤2.5 vs script CUDA ≤13.0, torch 2.9.x; khuyên -ampere nhưng script tắt | Có |
| A23 | CUDA toolkit Linux/WSL: "12.4+ recommended" vs "Optional; nvcc auto-detected" vs "installed inside WSL distro" | Có |
| A24 | Git: Linux `sudo apt install git` vs "Usually preinstalled"; Windows winget tự cài vs "Installed by setup script" | Có |
| A25 | NVIDIA Container Toolkit trên Windows: apt 1.17.8-1 vs "No separate … needed" | Có |
| A26 | docker run: cổng 8888/2222 vs 8888/8000/2222 vs 8000/8888; mount /workspace/work vs /workspace/host; driver 570.26+ | Có |
| A27 | Lệnh chạy Studio & bảo mật: -H 0.0.0.0 -p 8888 vs mặc định 127.0.0.1; --secure, --disable-tools; cổng 8888 vs 8000 | Có |
| A28 | Typo lệnh: `–break-system-packages` (en dash) | Có |
| A29 | Intel: lệnh powershell Set-ItemProperty ngoặc kép lồng, thiếu ngoặc; `setvars.bat" -` | Có |
| A30 | Bảng VRAM "absolute minimum", không ghi context/batch; bước nhảy 14B→27B | Có (mô tả + [Nhận định]) |

### inference

| # | Điểm | Trên web |
| --- | --- | --- |
| A31 | Đường dẫn HF cache / LM Studio trên Windows viết khác nhau trong studio/chat | Có |
| A32 | Context length: studio/chat "no longer necessary" vs unsloth-start --context-length 32768 vs api -c 131072 vs codex giảm ở Settings | Có |
| A33 | Sampling mặc định UI Studio không có trong nguồn | Có ("cần kiểm tra lại") |
| A34 | Port "typically 8000 or 8888" (api, python-sdk); ví dụ 8888; codex gọi 8888 là default; opencode /v1/ có dấu / | Có |
| A35 | Bảng endpoints api thiếu /v1/responses (văn bản api, curl, codex có) | Có |
| A36 | Model ID có/không tiền tố unsloth/ giữa curl, api, codex, claude-code (bash vs PowerShell), python-sdk | Có |
| A37 | Server-side tools: api "on by default" ở localhost vs curl/python-sdk opt-in enable_tools:true vs trang agent nói Studio nuốt tool call; ví dụ LAN claude-code thiếu --disable-tools, opencode có | Có |
| A38 | unsloth studio --secure: không nói chính sách tools | Có ("cần kiểm tra lại") |
| A39 | Tên model qwen3.8-27B-GGUF-GGUF vs Qwen3.8-27B-GGUF; thiếu `\` sau --model | Có |
| A40 | codex: `unsloth start Code` vs `unsloth start codex` | Có |
| A41 | Reasoning effort: api --chat-template-kwargs '{"reasoning_effort":"medium"}' vs unsloth-start --reasoning-effort medium | Có |
| A42 | api: comment "# Disable reasoning" nhưng --reasoning on | Có |
| A43 | codex: profile trong config.toml [profiles.unsloth_api] vs file ~/.codex/unsloth_api.config.toml; --profile vs -p; codex --oss llama_cpp vs --profile llama_cpp; provider llama_cpp thiếu requires_openai_auth | Có |
| A44 | Lệnh vLLM: `\` thừa (connections) / `\ --dtype auto` xuống dòng sai (connections/vllm) | Có |
| A45 | Base URL Ollama http://localhost:11434/v1 vs http://localhost:11434 | Có |
| A46 | enabled_tools: api có "terminal" vs SDK/curl python, bash, web_search | Có |
| A47 | tool-calling-guide: for…else đặt has_tool_calls=False sau vòng đầu; original_messages_len không dùng | Có ([Nhận định]) |
| A48 | curl ảnh qua /v1/messages: http://localhost/v1/messages thiếu port, thiếu model (không dùng ví dụ này) | Không |
| A49 | claude-code PowerShell thiếu bước đặt ANTHROPIC_API_KEY rỗng như bash | Không |

### model-catalog

| # | Điểm | Trên web |
| --- | --- | --- |
| A50 | Qwen3.5 27B 4-bit: 17GB (bảng) / 18GB (mục 27B) / 22GB (mở đầu) | Có |
| A51 | Qwen3.5 35B-A3B 4-bit: 22GB (bảng, mở đầu) / 24GB (mục 35B-A3B) | Có |
| A52 | Qwen3.5 35B-A3B 16-bit: BF16 70GB / F16 ~72GB | Có |
| A53 | Qwen3.5 397B: BF16 810GB / ~807GB | Có |
| A54 | Qwen3.8 27B: 17GB (mở đầu) / 16–19GB (bảng 4-bit) | Có |
| A55 | Qwen3.8 typo lệnh tải 2.4T UD-IQ1_S `--local-dir ...-GGFF` vs lệnh chạy ...-GGUF | Có |
| A56 | Gemma 4 E2B/E4B 4-bit: 4GB & 5.5–6GB (bảng) / 5GB (mở đầu) | Có |
| A57 | Gemma 4 E2B/E4B 16-bit: 10GB & 16GB (bảng) / 15GB (mở đầu) | Có |
| A58 | gpt-oss context max 131,072 nhưng lệnh 120b --ctx-size 262114 | Có |
| A59 | DeepSeek-V4 Q8: file 162GB cần ≥169GB / bảng 4-bit 162GB, Q8 169GB / benchmark Q8 161.9GB, Q4 155.1GB / "Q8 chỉ lớn hơn Q4 7GB" | Có |
| A60 | DeepSeek-V4 3-bit: 110GB / 110–135GB / "fits on 128GB" | Có |
| A61 | DeepSeek-V4 "run and trained in Unsloth" nhưng không có hướng dẫn fine-tune | Có |
| A62 | Kimi K3 2-bit: 861.3GB / bảng XXS 726GB, XL 880GB / UD-IQ2_XXS 711.1GB | Có |
| A63 | Kimi K3 UD-Q4_K_XL "requires 1.56TB" vs "Q8 lớn hơn Q4 50GB" | Có |
| A64 | Kimi K3 Q8: 1.6TB (bảng) / 1.56TB | Có |
| A65 | Kimi K3 "Mac Studio connected to a 128GB RAM device" vs cần ≥610GB | Có |
| A66 | Nemotron 3 Nano-4B "5GB" vs 4-bit ~3GB / 8-bit 5GB | Có |
| A67 | Nemotron 3 context 1M vs mặc định 262,144, OOM khi 1M | Có |
| A68 | Nemotron 3 "16-bit LoRA ~60GB" không rõ 4B hay 30B | Có |
| A69 | Mistral 3.5 "Vision for GGUFs it now supported for now. Support will come later." | Có |
| A70 | Mistral 3.5 "~64GB RAM" (3-bit) vs khuyên Dynamic 4-bit 80GB | Có |
| A71 | Mistral 3.5 "run or fine-tune" không có mục fine-tune | Có |
| A72 | Llama 4 Maverick 1.78-bit "fits in 2x48GB" vs "2 RTX 4090s (2x24GB)" | Có |
| A73 | Llama 4 "Full unquantized 113GB" không rõ Scout hay chung; Maverick 422GB | Có |
| A74 | Llama 4 bảng Maverick Q2_K_XL "151B" (typo đơn vị) | Có |
| A75 | Llama 4 "Run & Fine-tune" không có hướng dẫn fine-tune | Có |

### fine-tuning

| # | Điểm | Trên web |
| --- | --- | --- |
| A76 | Mặc định Studio (batch 4, GA 8, epochs 3, alpha 32, dropout 0.05) vs Core tutorial (batch 2, GA 4, max_steps 60, alpha 16, dropout 0) | Có (bảng 21 tham số) |
| A77 | gradient_accumulation_steps 4 (guide/tutorial) vs 8 (hyperparameters guide) | Có |
| A78 | QLoRA tiết kiệm VRAM "4×"/"75%"/"over 75%"; Unsloth "70%" vs ">70%/>75%" | Có |
| A79 | Mất độ chính xác 4-bit: "1-2%" / "marginally" / "negligible" / "largely recovered" | Có |
| A80 | Ngưỡng loss 0.5–1.0 (guide) vs <0.2 overfit (hyperparameters) | Có |
| A81 | Tên eval: evaluation_steps (guide) vs eval_steps (checkpoint) vs "Eval Steps" (Studio) | Có |
| A82 | Benchmark "H100 and Blackwell" nhưng VRAM 80GB; bảng context 8B không ghi GPU; 70B A100 | Có |
| A83 | "3GB VRAM" không điều kiện; Colab 16GB vs 15GB | Có |
| A84 | use_rslora: "set lora_alpha = 16 automatically" vs alpha/sqrt(r) | Có |
| A85 | bias: "Leave this as 0" nhưng code "none" | Có |
| A86 | Rank: 8–128 vs "4–64" vs slider Studio 4–128 | Có |
| A87 | Warmup: Studio 5 steps vs guide 5–10% tổng bước; train on completions: Studio false vs guide khuyên dùng | Có |
| A88 | Context dài hơn "4×" (guide/tutorial) vs 12×/13× (benchmarks) | Có |

### fine-tuning, du-lieu

| # | Điểm | Trên web |
| --- | --- | --- |
| A89 | "Start with Instruct" vs >1.000 dòng dùng base; tối thiểu 100 dòng vs mốc <300 dòng | Có |

### reinforcement-learning

| # | Điểm | Trên web |
| --- | --- | --- |
| A90 | Không có định dạng dữ liệu DPO/ORPO/KTO (chosen/rejected, nhãn KTO); ORPO/KTO không có code | Có ("cần kiểm tra lại") |
| A91 | Code DPO không tự chạy: max_seq_length, YOUR_DATASET_HERE chưa định nghĩa | Có (warning) |
| A92 | tokenizer= trong DPOTrainer có thể không khớp TRL mới ([Nhận định]) | Có |
| A93 | 15GB VRAM → 17B (rl-guide) vs 16GB → 16B (tutorial) | Có (warning) |
| A94 | 5GB cho ≤1.5B (rl-guide) vs Qwen3-1.7B FP8 trên 5GB (fp8-rl) | Có (warning) |
| A95 | 5GB cho ≤1.5B nhưng khuyên model ≥1.5B (rl-guide) | Có (warning) |
| A96 | Thời gian: ≥12 giờ (rl-guide) vs 300 steps ~30 phút (tutorial) | Có (warning) |
| A97 | loss_type: gspo/grpo/dr_grpo vs bnpo/grpo/dr_grpo/dapo | Có (warning) |
| A98 | Giảm VRAM 50–90% (memory-efficient) vs 90% (rl-guide) | Có (warning) |
| A99 | Context dài hơn 1.2–1.7x vs ví dụ Llama-3.1-8B 1.13x | Có (warning) |
| A100 | Standby tiết kiệm "30%+" (fp8-rl) vs 15% (thí nghiệm T4) | Có (warning) |

### du-lieu

| # | Điểm | Trên web |
| --- | --- | --- |
| A101 | Data Recipes chỉ "PDFs or CSVs"; Seed không liệt kê định dạng; DOCX/JSONL/Parquet chỉ ở tab Local; nháp "need to add more here", mục trống | Có |
| A102 | JSON ví dụ: ChatML dấu phẩy thừa; Alpaca cặp khóa rời | Có |
| A103 | Cột Alpaca "Instruction/Input/Output" hoa (datasets-guide) vs thường (Studio, tutorial) | Có |
| A104 | Ví dụ raw text chỉ `"text": ...` thiếu {} | Có |
| A105 | standardize_sharegpt: "Always call this!" vs "Only use if sharegpt but model expects ChatML" | Có |
| A106 | Vision: train text trước ảnh vs ví dụ inference ảnh trước text | Có |
| A107 | Comment mapping chỉ 8 template vs CHAT_TEMPLATES nhiều hơn | Có |

### export-deploy

| # | Điểm | Trên web |
| --- | --- | --- |
| A108 | llama-server guide: tải Devstral-2-123B UD-Q2_K_XL nhưng serve Devstral-Small-2-24B UD-Q4_K_XL | Có |
| A109 | Tools mặc định khi mở mạng: lan & cloudflare "on by default" vs api "unsloth run bind 0.0.0.0 thì tắt" | Có |
| A110 | Menu Remote access 4 cách ghi (Settings → API → Remote & LAN / API → Remote access / Remote & LAN → Remote access / API thẻ Remote access) | Có |
| A111 | NVFP4: hai danh sách GPU (RTX 5050–5090, B200, RTX PRO 6000 vs RTX 50X, DGX Spark, B200, B300) | Có |
| A112 | NVFP4 DGX Spark không flashinfer_b12x: "2x SLOWER" vs "much slower" vs "2.5x slower" | Có |
| A113 | NVFP4 Marlin: "2.5x slower" vs bảng (decode 105.6 vs 125.9; throughput 2,127 vs 6,863) | Có |
| A114 | NVFP4 Qwen3.6-35B-A3B "1.7x faster" vs bảng 1.56×/1.79× | Có |
| A115 | NVFP4 Gemma 4 "at most 1.44x" vs bảng 31B 1.45× | Có |
| A116 | NVFP4: lệnh SGLang unsloth/Gemma-4-31B-NVFP4 vs bảng unsloth/gemma-4-31B-it-NVFP4 | Có |
| A117 | NVFP4: đổi tên model "Qwen3.6-35-A3B-NVFP4" (thiếu B) vs repo Qwen3.6-35B-A3B-NVFP4 | Có |
| A118 | NVFP4: lệnh vllm serve … --speculative-config thiếu `\` | Có |
| A119 | saving-to-ollama: toàn bộ code nằm trong ảnh (kể cả bản gốc); chỉ có `ollama serve` | Có (info "cần kiểm tra lại") |
| A120 | NVFP4: không có cách tự xuất NVFP4 từ model fine-tune | Có (info) |

### ung-dung-rag

| # | Điểm | Trên web |
| --- | --- | --- |
| A121 | (cùng mâu thuẫn tools mặc định, ở mục Unsloth API) | Có |
| A122 | Code embedding SentenceTransformer("<your-unsloth-finetuned-model") thiếu `>` | Có |
| A123 | Port API: localhost:PORT / "8000 or 8888" / 8888 | Có |
| A124 | Embedding: không có code train, không mô tả định dạng dữ liệu | Có (info) |
| A125 | api: không nói unsloth run load được GGUF local không | Có ("cần kiểm tra lại") |

### kien-thuc-nen/phan-loai-mo-hinh

| # | Điểm | Trên web |
| --- | --- | --- |
| A126 | Base vs Instruct: "start with Instruct" vs >1.000 dòng nên fine-tune base (what-model-should-i-use) | Có |

### kien-thuc-nen/dense-va-moe

| # | Điểm | Trên web |
| --- | --- | --- |
| A127 | faster-moe tự mâu thuẫn: Qwen3-30B-A3B "1.8x"/"~1.7x" (B200)/"1.77x" (H100); GLM-4.7-Flash "2.1x on RTX PRO 6000" vs "2.6x"; "LoRA rank = 64" vs cột Rank = 8; tiêu đề ">35%" VRAM vs bảng 2–15% | Có |
| A128 | VRAM fine-tune gpt-oss-20b: 12.8 GB (faster-moe) vs 14 GB (trang gpt-oss) | Có |
| A129 | Qwen3.5 fine-tune: không khuyên QLoRA "no matter MoE or dense" vs comment code "dense 27B is fine" | Có |

### kien-thuc-nen/tham-so-va-bo-nho

| # | Điểm | Trên web |
| --- | --- | --- |
| A130 | Bảng VRAM fine-tune của Unsloth không ghi model/context/batch; VramEstimator lệch +36% (QLoRA 14B), +19% (QLoRA 70B) | Có (hộp cảnh báo dưới component) |

### kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa

| # | Điểm | Trên web |
| --- | --- | --- |
| A131 | Mức mất độ chính xác của QLoRA 4-bit: "1-2%" / "negligible" / "largely recovered" / "marginally less accurate" | Có |
| A132 | NVFP4 trên SGLang: qwen3.8 ghi "vLLM only (SGLang is not supported)" nhưng có hướng dẫn SGLang v0.5.19 | Có |

### kien-thuc-nen/qua-trinh-huan-luyen

| # | Điểm | Trên web |
| --- | --- | --- |
| A133 | Hyperparameters guide: "keep alpha/rank at least = 1" (alpha ≥ r) vs mục Underfitting "Rank should at least equal to the alpha" (r ≥ alpha) | Có |
| A134 | Warmup: 5–10% tổng bước (guide) vs 5 steps (Studio) vs warmup_steps = 10 (code Colab/Windows install) | Có |
| A135 | Effective batch mặc định Studio 4×8 = 32 nằm ngoài khoảng 4–16 khuyến nghị của hyperparameters guide | Có |
| A136 | `evaluation_steps` (fine-tuning guide) không có trong TRL SFTConfig (chỉ có eval_steps) | Có ("cần kiểm tra lại") |

### kien-thuc-nen/lora-va-qlora

| # | Điểm | Trên web |
| --- | --- | --- |
| A137 | faster-moe dùng lora_alpha = lora_rank*2 "speeds up training" — thêm một biến thể khuyến nghị alpha | Có |

### kien-thuc-nen/suy-luan-va-sampling

| # | Điểm | Trên web |
| --- | --- | --- |
| A138 | Qwen3.5 presence_penalty: hint mặc định tắt (0.0–2.0) vs bảng 1.5 vs lệnh mẫu không truyền | Có |

## B. Chỉ lệch mô tả

### index

| # | Điểm | Trên web |
| --- | --- | --- |
| B1 | "2x nhanh, ít hơn 70% VRAM" không điều kiện đo; bảng notebook README 1.5x–2x, 20%–80% | Có (bảng) |
| B2 | docs.md "open-source framework" vs README "first desktop app" | Không |

### tong-quan

| # | Điểm | Trên web |
| --- | --- | --- |
| B3 | Inference Core "2x faster" không điều kiện đo | Không |
| B4 | Quan hệ Desktop–Studio: "cài Studio dễ nhất qua Desktop" vs install "ba cách tách biệt" | Không ([Nhận định]) |
| B5 | Unsloth Start: docs.md & README có DeepSeek Harness (dsh), new/studio không có | Không |
| B6 | Thiếu thông tin: Core trên macOS, license Desktop, export của Core | Không ("cần kiểm tra lại" trên trang) |
| B7 | Danh sách phương pháp train khác nhau theo trang (desktop / studio/start / new/studio) | Không |

### tong-quan, lo-trinh-hoc

| # | Điểm | Trên web |
| --- | --- | --- |
| B8 | Colab "up to 22B" trên T4 không rõ quant/phương pháp | Không |

### cai-dat

| # | Điểm | Trên web |
| --- | --- | --- |
| B9 | Export trên CPU "coming very soon" | Có |
| B10 | Không có yêu cầu hệ thống riêng cho Desktop | Có |
| B11 | Link "#method-2-wsl" sai (WSL là Method #3) | Không |
| B12 | get-started/install: thẻ AMD trỏ /docs/basics/amd | Không |

### inference

| # | Điểm | Trên web |
| --- | --- | --- |
| B13 | python-sdk "three OpenAI-compatible dialects" tính cả Anthropic Messages | Không |
| B14 | Số liệu tool calling "by 50%", "+50%", "30%–80%" không điều kiện đo, 10 lần chạy | Không |
| B15 | UI Connections: "Connected"/"Connection"/"External"; nút "Add Connection"/"Add Provider"/"Add Connected" | Có (ghi chú ngắn) |
| B16 | Anchor #auto-parameter-tuning không khớp heading | Không |
| B17 | Model Arena song song "being worked on" | Có |
| B18 | mcp: link llama.cpp hỏng (llama.cpphttps://...) | Không |

### model-catalog

| # | Điểm | Trên web |
| --- | --- | --- |
| B19 | Qwen3.8 typo "Qwen3.8-2.T" trong văn bản | Không |
| B20 | Gemma 4 "up to 256K" nhưng E2B/E4B 128K | Không (trang ghi context từng bản) |
| B21 | gpt-oss không ghi số tham số kích hoạt | Có ("—") |
| B22 | DeepSeek-V4-Pro không có yêu cầu bộ nhớ | Có ("—") |
| B23 | Kimi K3 1-bit 594GB vs 553.2GiB (khác đơn vị) | Không |
| B24 | GLM-5.3 UD-IQ2_M 239GB / cần ≥245GB / chạy tốt 256GB (khớp) | Không |
| B25 | GLM-5.3 không dùng chữ MoE, chỉ "40B active" | Có (ký hiệu MoE*) |
| B26 | Mistral 3.5 không có trong catalog | Không |
| B27 | Llama 4 không giải thích 17B-16E/128E | Có ("—") |
| B28 | Granite notebook 4.0, đổi tên sang 4.1 | Có |
| B29 | Granite 4.1, Qwen3-VL, DeepSeek-OCR 2 không có yêu cầu bộ nhớ | Có |
| B30 | Catalog không có nhóm FP8 | Có |
| B31 | Mistral Medium 3.5, Granite 4.1, DeepSeek-OCR 2 không có trong catalog | Không |
| B32 | "Muse Glimmer" link sang deepseek-v4, xuất hiện 2 lần | Không |
| B33 | DeepSeek-V4 Flash-0731 cột GGUF link changelog | Không |
| B34 | Cột 4-bit Kimi K3 & DSV4-Flash trỏ safetensors gốc | Không |
| B35 | Qwen3.6, Gemma 4 26B-A4B NVFP4 ở bảng New nhưng "—" ở bảng họ | Không |
| B36 | Lẫn tên Unsloth Studio / Desktop | Không |

### fine-tuning

| # | Điểm | Trên web |
| --- | --- | --- |
| B37 | "as of January 2025 leading 70B Llama 3.3" cũ | Không |
| B38 | Multi-GPU "official support soon" | Có |
| B39 | QAT/ExecuTorch "coming soon"; torchao==0.14.0 ghim | Không |
| B40 | Tên model lạ trong notebooks (Qwen3.8 27B, Muse Glimmer, DiffusionGemma) | Không |

### reinforcement-learning

| # | Điểm | Trên web |
| --- | --- | --- |
| B41 | Không có code Python reward function trong nguồn; ví dụ chỉ là mô tả quy tắc (trình bày dạng bảng) | Có |
| B42 | "10% faster RL", "2x torch.compile" không điều kiện đo | Có (ghi chú trong bảng) |
| B43 | Link GitBook hỏng .../pages/nw2c1elNySGBBav8WP9B (tutorial GRPO) | Không |
| B44 | fp8-rl "We'll share details in a new blog soon" | Không |

### du-lieu

| # | Điểm | Trên web |
| --- | --- | --- |
| B45 | Khối unsloth_template hỏng do GitBook | Không |
| B46 | Link notebook "ChatML"/"Ollama" bị tráo | Không |
| B47 | "NVIDIA DataDesigner" vs "NVIDIA Nemo Data Designer" | Không |

### export-deploy

| # | Điểm | Trên web |
| --- | --- | --- |
| B48 | FP8: không có trang export FP8 riêng | Có (info) |
| B49 | NVFP4 ghi "Dynamic 2.0" nhưng link dynamic-3.0 | Không |
| B50 | Dynamic 3.0 ">10% top-1" chỉ đo Qwen3.8-27B; số 2.0 đo Gemma 3 27B | Có (bảng điều kiện đo) |
| B51 | Dynamic 3.0 nhắc cả Desktop lẫn Studio | Không |
| B52 | Dynamic 3.0 link Llama 4 domain cũ docs.unsloth.ai | Không |
| B53 | studio/export bảng Export Type có dòng trống | Không |

### ung-dung-rag

| # | Điểm | Trên web |
| --- | --- | --- |
| B54 | Tốc độ embedding "1.8–3.3x" vs QLoRA 1.8–2.6x vs LoRA 16-bit 1.2–3.3x | Có (ghi đủ trong câu) |
| B55 | Docs không nêu model embedding hỗ trợ tiếng Việt | Có ([Nhận định]) |
| B56 | Embedding typo "Lacker's other fast classes" | Không |

### kien-thuc-nen/token-va-context

| # | Điểm | Trên web |
| --- | --- | --- |
| B57 | Hệ số "context dài hơn" khác nhau theo trang: 4× (fine-tuning guide), 2× (embedding), ">8×"/"10×" (gpt-oss) — mỗi số một ngữ cảnh | Không |
| B58 | Colab install: Unsloth "supports RoPE Scaling internally, so choose any" max_seq_length — không nói chất lượng khi vượt context gốc | Có ([Nhận định]) |

### kien-thuc-nen/phan-loai-mo-hinh

| # | Điểm | Trên web |
| --- | --- | --- |
| B59 | Gemma 4 "E2B/E4B": chữ "E" (Dense + PLE) không được giải thích | Có ("cần kiểm tra lại") |

### kien-thuc-nen/dense-va-moe

| # | Điểm | Trên web |
| --- | --- | --- |
| B60 | faster-moe: ngưỡng Split LoRA s<32K (bộ nhớ) vs crossover ≈16K (FLOPs), chiều bất đẳng thức ngược nhau | Không |
| B61 | GLM-4.7-Flash 64 routed + 1 shared expert chỉ có trong faster-moe, không đối chiếu được | Có ("cần kiểm tra lại") |

### kien-thuc-nen/tham-so-va-bo-nho

| # | Điểm | Trên web |
| --- | --- | --- |
| B62 | "UD" = Unsloth Dynamic không được docs định nghĩa; hậu tố "_XL" không định nghĩa | Có ([Nhận định] / "cần kiểm tra lại") |
| B63 | Qwen3.5/3.8-27B kiến trúc lai (16/64 layer full attention); dung lượng trạng thái Gated DeltaNet không có nguồn | Có ("cần kiểm tra lại") |

### kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa

| # | Điểm | Trên web |
| --- | --- | --- |
| B64 | NVFP4 page ghi "Unsloth Dynamic 2.0" vs qwen3.8 GGUF dùng Dynamic 3.0 | Không |
| B65 | Bảng thiếu điều kiện: gguf-benchmarks Qwen3.5 không ghi model; KLD Gemma 3 không ghi size/baseline; NVFP4 "92–97% top-1" không rõ Qwen3.8 hay 3.6; bảng tốc độ NVFP4 vs BF16 không ghi GPU | Có (ghi chú điều kiện) |
| B66 | FP8 trên Ada (RTX 40, L4): chỉ docs Unsloth nói; NVIDIA TE primer chỉ nêu Hopper (FP8) và Blackwell (NVFP4/MXFP8) | Có ("cần kiểm tra lại") |

### kien-thuc-nen/lora-va-qlora

| # | Điểm | Trên web |
| --- | --- | --- |
| B67 | Unsloth Dynamic 4-bit có dùng NF4 hay không — docs QLoRA không nói rõ | Có ("cần kiểm tra lại") |
| B68 | LoRA hot swapping "up to 4 adapters" chỉ suy ra từ ví dụ --max-loras 4 | Có ([Nhận định]) |

### kien-thuc-nen/rl-va-preference

| # | Điểm | Trên web |
| --- | --- | --- |
| B69 | RL guide: GRPO lấy trung bình reward "across multiple different questions" vs ví dụ của chính trang, paper DeepSeekMath và TRL (cùng một câu hỏi) | Có |
| B70 | GRPO beta: Unsloth advanced docs & TRL mặc định 0.0 vs RL guide mô tả KL với beta > 0 | Có (bảng) |
| B71 | Unsloth: GRPO "removes the Reward Model" vs TRL cho phép truyền reward model | Có |
| B72 | DPO ref_model=None với LoRA: docs không nói reference là model gốc tắt adapter | Có ([Nhận định]) |

### kien-thuc-nen/suy-luan-va-sampling

| # | Điểm | Trên web |
| --- | --- | --- |
| B73 | Gemma 4 không có min_p/penalty khuyến nghị; gpt-oss không có min_p trong khuyến nghị nhưng lệnh 120B dùng --min-p 0.0 | Có ("Docs không ghi") |

### (không trang nào)

| # | Điểm | Trên web |
| --- | --- | --- |
| B74 | desktop/api: typo qwen3.8-27B-GGUF-GGUF, thiếu `\` (không chép lên trang này) | Không |
| B75 | api: comment "Disable reasoning" vs --reasoning on | Không |
| B76 | api: link hỏng "see ." | Không |
| B77 | README link Windows ARM64 ghim v0.1.811-beta | Không |
## C. Kiểm tra VramEstimator với bảng VRAM của Unsloth

Cấu hình khi chạy thử: context 2048, batch 2, LoRA r = 16 gắn đủ 7 module, `adamw_8bit`, phần phụ 10%. Mỗi dòng dùng kiến trúc từ `config.json` chính thức của model gần nhất với dòng tương ứng trong bảng [unsloth-requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements).

| Dòng bảng Unsloth | Kiến trúc thử | QLoRA docs → ước tính | Lệch | LoRA docs → ước tính | Lệch |
| --- | --- | --- | --- | --- | --- |
| 3B | Llama 3.2 3B | 3.5 → 3.33 GB | −5% | 8 → 8.63 GB | +8% |
| 8B | Llama 3.1 8B | 6 → 6.78 GB | +13% | 22 → 20.0 GB | −9% |
| 14B | Qwen3 14B | 8.5 → 11.6 GB | **+36%** | 33 → 36.0 GB | +9% |
| 32B | Qwen3 32B | 26 → 23.5 GB | −10% | 76 → 77.6 GB | +2% |
| 70B | Llama 3.3 70B | 41 → 48.7 GB | **+19%** | 164 → 165 GB | +1% |

- Nếu đặt phần phụ bằng 0%, độ lệch nằm trong khoảng −24% đến +24%.
- Hai chỗ lệch lớn (QLoRA 14B và 70B) được ghi rõ trên trang [Tham số & bộ nhớ](docs/kien-thuc-nen/tham-so-va-bo-nho.md). Công thức không được chỉnh cho khớp.
- Bảng docs không ghi model, context hay batch dùng để đo, nên không xác định được lệch là do công thức hay do điều kiện đo.
- Mức phần phụ 10% không có nguồn và được gắn nhãn [Nhận định] trên trang.
