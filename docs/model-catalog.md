---
title: Danh sách model hỗ trợ
description: Bảng tra các họ model tiêu biểu mà Unsloth cung cấp (Qwen, Gemma, gpt-oss, DeepSeek, GLM, Kimi, Nemotron, Mistral, Granite, Llama), kèm yêu cầu bộ nhớ và khả năng fine-tune theo docs.
---

# Danh sách model hỗ trợ

Trang này giúp bạn tra nhanh các họ model Unsloth cung cấp, mỗi model cần bao nhiêu bộ nhớ và có fine-tune được không. Đọc xong, bạn chọn được model vừa với máy và mục tiêu của mình.

::: tip Tóm tắt
- **Dùng khi:** Bạn cần chọn model để chạy hoặc fine-tune và muốn biết nó có vừa GPU/RAM của mình không.
- **Kết quả:** Bảng 13 họ model tiêu biểu kèm kiến trúc, yêu cầu bộ nhớ, khả năng fine-tune theo docs, và gợi ý cách chọn.
- **Nên biết trước:** Ký hiệu "35B-A3B", tổng tham số và tham số kích hoạt: [Dense & MoE](/kien-thuc-nen/dense-va-moe). Vì sao model 27B cần khoảng 17 GB: [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho). 4-bit, 8-bit, BF16, NVFP4, MXFP4, `UD-Q4_K_XL`: [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa). Temperature, top_p, top_k: [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling). Context window: [Token & context](/kien-thuc-nen/token-va-context). LoRA, QLoRA: [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora). GRPO: [RL & preference](/kien-thuc-nen/rl-va-preference).
:::

::: warning Dữ liệu có hạn dùng
Dữ liệu truy cập ngày **2026-09-24**. Catalog thay đổi nhanh: model mới ra, quant mới được thêm, số liệu bộ nhớ được cập nhật lại. Trước khi tải, bạn luôn nên đối chiếu với [Unsloth Model Catalog gốc](https://unsloth.ai/docs/get-started/unsloth-model-catalog) và trang hướng dẫn của từng model.
:::

Unsloth Model Catalog là danh mục các bản model mà Unsloth đưa lên Hugging Face. Catalog có ba kiểu bản chính:

- Bản GGUF (định dạng file model cho llama.cpp) đã lượng tử hóa kiểu Dynamic.
- Bản 4-bit.
- Bản NVFP4.

Trang này chọn ra 13 họ model có trang hướng dẫn riêng. Với mỗi họ, trang tóm tắt thông số và yêu cầu bộ nhớ đúng như docs ghi.

## Cách đọc bảng

Mục này giải thích các cột và ký hiệu trong bảng model. Bạn nên đọc trước để hiểu con số bộ nhớ đang nói về loại bộ nhớ nào.

Quy ước trong bảng:

- **Tổng tham số / Tham số kích hoạt**: chép theo docs. Model MoE (Mixture of Experts, kiến trúc chỉ kích hoạt một phần mạng cho mỗi token) có hai con số. Model dense (mọi tham số đều tham gia tính toán) chỉ có một.
- **Kiến trúc**: chỉ ghi "dense" hoặc "MoE" khi docs nói rõ. Ký hiệu **MoE\*** nghĩa là docs chỉ ghi số tham số kích hoạt (vd "13B active", hậu tố "A3B") mà không dùng chữ MoE. **[Nhận định]** Trang này xếp chúng vào MoE vì chỉ model MoE mới có khái niệm tham số kích hoạt.
- **Yêu cầu bộ nhớ**: hầu hết trang docs tính theo **tổng bộ nhớ**. Tổng bộ nhớ = RAM (bộ nhớ hệ thống) + VRAM (bộ nhớ card đồ họa), hoặc là unified memory (bộ nhớ dùng chung cho CPU và GPU, như trên Mac). Con số luôn đi kèm một mức quant (mức lượng tử hóa) cụ thể.
- **"—"**: docs được dùng làm nguồn không ghi thông tin này. Trang này không tự điền thay.
- **⚠**: các chỗ trong docs ghi khác nhau. Ô bảng liệt kê đủ các giá trị. Chi tiết và link nằm trong hộp "Docs chưa thống nhất" ở mục của họ model đó.

Nhiều trang docs lặp lại một quy tắc chung: tổng bộ nhớ khả dụng nên **lớn hơn kích thước file quant** bạn tải. Nếu không đủ, llama.cpp vẫn chạy được nhờ offload (đẩy một phần sang RAM hoặc ổ đĩa), nhưng sinh token chậm hơn.

**Nguồn:** https://unsloth.ai/docs/get-started/unsloth-model-catalog, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/gemma-4

## Bảng model tiêu biểu

Bảng này giúp bạn tra nhanh: model nào lớn bao nhiêu, cần bao nhiêu bộ nhớ, và docs có nói rõ là fine-tune được hay không.

| Tên | Tổng tham số | Tham số kích hoạt | Kiến trúc | Loại | Yêu cầu bộ nhớ theo docs | Fine-tune với Unsloth? | Docs |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 27B | — | — | Text + vision, hybrid thinking | 4-bit: 16–19 GB / 17 GB ⚠ (xem cảnh báo mục Qwen3.8); 56 GB cho BF16 | — | [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) |
| Qwen3.8-2.4T-A95B | 2.4T | 95B | MoE\* | LLM, thinking-only | 397 GB cho Dynamic 1-bit XXXS; tutorial khuyên ≥ 450 GB RAM cho quant này; BF16 4.9 TB | — | [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) |
| Qwen3.5-9B | 9B | — | — | Multimodal, hybrid reasoning | 6.5 GB RAM+VRAM cho 4-bit; 19 GB cho BF16 | Có (trang có mục "Fine-tune Qwen3.5") | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Qwen3.5-35B-A3B | 35B | 3B | MoE\* | Multimodal, hybrid reasoning | 4-bit: 22 GB / 24 GB ⚠; BF16: 70 GB / ~72 GB ⚠ (xem cảnh báo mục Qwen3.5) | Có | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Qwen3.5-397B-A17B | 397B | 17B | MoE\* | Multimodal, hybrid reasoning | 214 GB cho 4-bit (`UD-Q4_K_XL`); 180 GB cho 3-bit; BF16: 810 GB / ~807 GB ⚠ | Có | [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5) |
| Gemma 4 E4B | — | — | Dense + PLE | Text, image, audio | 4-bit: 5.5–6 GB / 5 GB ⚠; 16-bit: 16 GB / 15 GB ⚠ (xem cảnh báo mục Gemma 4) | Có | [Gemma 4](https://unsloth.ai/docs/models/gemma-4) |
| Gemma 4 26B-A4B | 26B | 4B | MoE | Text, image | 16–18 GB RAM+VRAM cho 4-bit; 52 GB cho BF16 | Có | [Gemma 4](https://unsloth.ai/docs/models/gemma-4) |
| Gemma 4 31B | 31B | — | Dense | Text, image | 17–20 GB RAM+VRAM cho 4-bit; 62 GB cho BF16 | Có | [Gemma 4](https://unsloth.ai/docs/models/gemma-4) |
| gpt-oss-20b | 20B | — (chọn 4/32 expert mỗi token) | MoE | Text, reasoning | Chạy: ≥ 14 GB (unified hoặc RAM) cho Dynamic 4-bit; QLoRA: 14 GB VRAM | Có (QLoRA, BF16 LoRA, GRPO) | [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| gpt-oss-120b | 120B | — (chọn 4/128 expert mỗi token) | MoE | Text, reasoning | Chạy: ≥ 66 GB cho quant 1-bit; QLoRA: 65 GB VRAM (context: xem cảnh báo mục gpt-oss ⚠) | Có | [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune) |
| DeepSeek-V4-Flash-0731 | 284B | 13B | MoE\* | LLM (coding, agent, chat) | File `UD-IQ3_XXS` 103 GB, cần 110 GB / 110–135 GB / máy 128 GB ⚠; `UD-Q8_K_XL`: file 162 GB, cần ≥ 169 GB ⚠ (xem cảnh báo mục DeepSeek-V4) | Chưa rõ ⚠: trang ghi "run and trained in Unsloth" nhưng không có hướng dẫn fine-tune | [DeepSeek-V4](https://unsloth.ai/docs/models/deepseek-v4) |
| DeepSeek-V4-Pro-0813 | 1.6T | 49B | MoE\* | LLM | — | — | [DeepSeek-V4](https://unsloth.ai/docs/models/deepseek-v4) |
| GLM-5.3 | 744B | 40B | MoE\* | LLM, luôn bật thinking | 223 GB cho 1-bit; `UD-IQ2_M` 239 GB trên đĩa, chạy tốt trên máy 256 GB RAM; 810 GB cho 8-bit | — | [GLM-5.3](https://unsloth.ai/docs/models/glm-5.3) |
| Kimi K3 | 2.8T | 104B | MoE (trọng số MoE ở MXFP4) | Text + vision, thinking-only | `UD-IQ1_S` 594 GB, cần ≥ 610 GB RAM+VRAM; Q8 lossless: 1.6 TB / 1.56 TB ⚠ (xem cảnh báo mục Kimi K3) | — | [Kimi K3](https://unsloth.ai/docs/models/kimi-k3) |
| Nemotron-3-Nano-4B | 4B | — | "Hybrid MoE" | LLM (coding, math, agent) | ~3 GB cho 4-bit; 5 GB cho 8-bit; đoạn mở đầu ghi "5 GB" không nói quant ⚠ | Có (vừa GPU Colab miễn phí) | [Nemotron 3](https://unsloth.ai/docs/models/nemotron-3) |
| Nemotron-3-Nano-30B-A3B | 30B | 3B | MoE\* | LLM | ~24 GB RAM cho 4-bit; 36 GB cho 8-bit | Có; "LoRA 16-bit của Nemotron 3 Nano ~60 GB VRAM", không rõ 4B hay 30B ⚠ | [Nemotron 3](https://unsloth.ai/docs/models/nemotron-3) |
| Mistral-Medium-3.5-128B | 128B | — | Dense | Text + image vào, text ra (vision qua GGUF: chưa rõ ⚠) | 64 GB RAM+VRAM cho 3-bit; 80 GB cho 4-bit; 128–170 GB cho 8-bit (đoạn mở đầu: "~64GB RAM") | Chưa rõ ⚠: mô tả trang nhắc "run or fine-tune", không có hướng dẫn fine-tune | [Mistral 3.5](https://unsloth.ai/docs/models/mistral-3.5) |
| Granite-4.1-3B | 3B | — | Dense | Text (chat, RAG, tool calling) | — | Có | [Granite 4.1](https://unsloth.ai/docs/models/ibm-granite-4.1) |
| Granite-4.1-8B | 8B | — | Dense | Text | — | Có | [Granite 4.1](https://unsloth.ai/docs/models/ibm-granite-4.1) |
| Granite-4.1-30B | 30B | — | Dense | Text | — | Có | [Granite 4.1](https://unsloth.ai/docs/models/ibm-granite-4.1) |
| Llama-4-Scout (17B-16E) | 109B | — | MoE (docs nhắc các lớp MoE) | Text + vision | Bản 1.78-bit 33.8 GB, vừa GPU 24 GB VRAM (~20 token/s); bản gốc 113 GB (docs không ghi rõ là Scout) ⚠ | Chưa rõ ⚠: tiêu đề trang "Run & Fine-tune", có bản bnb-4bit, nhưng trang không có hướng dẫn fine-tune | [Llama 4](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) |
| Llama-4-Maverick (17B-128E) | 402B | — | MoE (xen kẽ lớp dense và MoE) | Text + vision | Bản 1.78-bit 122 GB: "vừa 2×48 GB VRAM" / "tốt nhất 2× RTX 4090 (2×24 GB)" ⚠; bản gốc 422 GB | Như trên | [Llama 4](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) |
| Qwen3-VL-8B | 8B | — | Dense | Vision, video, OCR | — | Có (notebook SFT và GRPO miễn phí trên Colab) | [Qwen3-VL](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) |
| Qwen3-VL-30B-A3B | 30B | 3B | MoE | Vision, video, OCR | — | Có (docs nói hỗ trợ fine-tune và RL cho Qwen3-VL) | [Qwen3-VL](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) |
| Qwen3-VL-235B-A22B | 235B | 22B | MoE | Vision, video, OCR | — | Có (docs nêu rõ gồm cả 235B) | [Qwen3-VL](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune) |
| DeepSeek-OCR 2 | 3B | — | — | OCR / hiểu tài liệu (image-to-text) | — | Có (notebook Colab miễn phí) | [DeepSeek-OCR 2](https://unsloth.ai/docs/models/tutorials/deepseek-ocr-2) |

::: info Về cột "Fine-tune"
Hầu như trang model nào cũng có một đoạn giới thiệu chung về Unsloth Desktop/Studio ("Train LLMs 2x faster with 70% less VRAM"). Trang này **không** coi đoạn quảng bá đó là bằng chứng rằng model fine-tune được. Cột này chỉ ghi "Có" khi trang của chính model đó nói rõ fine-tune được.
:::

**Nguồn:** https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune, https://unsloth.ai/docs/models/deepseek-v4, https://unsloth.ai/docs/models/glm-5.3, https://unsloth.ai/docs/models/kimi-k3, https://unsloth.ai/docs/models/nemotron-3, https://unsloth.ai/docs/models/mistral-3.5, https://unsloth.ai/docs/models/ibm-granite-4.1, https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune, https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune, https://unsloth.ai/docs/models/tutorials/deepseek-ocr-2

## Các nhóm trong catalog

Mỗi model trong catalog có nhiều định dạng tải về. Định dạng nào hợp với bạn tùy vào việc bạn muốn chạy model hay fine-tune nó, và bạn có loại GPU nào.

Catalog gốc chia bảng theo **họ model**: New & recommended, DeepSeek, Llama, Gemma, Qwen, GLM, Mistral, Phi, Other. Mỗi dòng có các cột định dạng tải về như sau:

| Nhóm / cột | Ý nghĩa theo docs | Dùng khi |
|---|---|---|
| **GGUF** (Unsloth [Dynamic](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)) | File cho [Unsloth Desktop](https://unsloth.ai/docs/desktop) và llama.cpp. Bản Dynamic không hạ mọi lớp xuống cùng mức bit: vd 4-bit vẫn giữ các lớp quan trọng ở 8 hoặc 16-bit (trang Qwen3.5). | Chạy local trên CPU/GPU/Mac |
| **4-bit / Instruct (4-bit)** | Safetensors 4-bit "cho inference hoặc fine-tune qua Unsloth". Nhiều link có tên repo đuôi `bnb-4bit` / `unsloth-bnb-4bit`; trang Qwen3-VL gọi là "4-bit BnB Unsloth Dynamic". | Nạp vào Unsloth để fine-tune (QLoRA) |
| **NVFP4** | Quant 4-bit của NVIDIA. Trang Qwen3.8 ghi: nhanh hơn BF16 khoảng 1.5×, **chỉ chạy trên GPU Blackwell** (RTX 50, DGX Spark, B200, B300), hiện chỉ dùng qua vLLM (và SGLang v0.5.19). | Serve trên GPU Blackwell |
| **MTP** (vd Qwen3.6-27B-MTP-GGUF) | GGUF có bật MTP (multi-token prediction, đoán trước nhiều token) để suy luận nhanh hơn; trang Qwen3.8 dặn chừa thêm 1–2 GB bộ nhớ. | Muốn tăng tốc decode |
| **QAT** (Gemma 4) | Bản quantization-aware training; trang Gemma 4 ghi giảm yêu cầu bộ nhớ khoảng 3× mà vẫn giữ chất lượng. | Máy ít bộ nhớ |

Catalog **không** có nhóm FP8 riêng. Trang Gemma 4 và GLM-5.3 có nhắc thêm bản MLX (định dạng cho Apple Silicon), nhưng bảng catalog không có cột MLX.

::: warning Phần cứng
- NVFP4 cần GPU Blackwell. Với GPU đời cũ, docs khuyên dùng GGUF.
- Qwen3.5 GGUF và GGUF có vision của Mistral 3.5 **không chạy được trong Ollama**. Lý do: file vision `mmproj` nằm tách riêng. Docs khuyên dùng backend tương thích llama.cpp.
- Trang Mistral 3.5 và Granite 4.1 cảnh báo **không dùng CUDA 13.2** vì có thể sinh ra chữ vô nghĩa.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/unsloth-model-catalog, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune, https://unsloth.ai/docs/models/mistral-3.5, https://unsloth.ai/docs/models/ibm-granite-4.1

## Chi tiết từng họ model

Mỗi họ model bên dưới được thu gọn. Bấm vào tên họ model để mở phần chi tiết: các bản, context, sampling, bộ nhớ và chỗ docs chưa thống nhất.

:::: details Qwen3.8
Qwen3.8 là họ model mới nhất của Qwen trong catalog, gồm 27B, 2.4T-A95B và Max. **[Nhận định]** Bản 27B chạy được trên máy cá nhân; bản 2.4T cần máy rất nhiều RAM.

- **Bản 27B:** có vision và reasoning. Đây là model **hybrid thinking**, tức có cả chế độ thinking và non-thinking. GGUF dùng Unsloth Dynamic V3.0.
- **Bản 2.4T-A95B:** là model **thinking-only**. Bản đầy đủ cần 4.9 TB; bản Dynamic 1-bit còn 397 GB.

Thông số chính:

- Context window (độ dài ngữ cảnh tối đa): 27B là `262,144` token (mở rộng tới 1M bằng YaRN); 2.4T tới `1,010,000`.
- Sampling khuyến nghị cho 27B:
  - Thinking: `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0`.
  - Non-thinking: `temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`.
- Có `reasoning_effort` (`xhigh` mặc định, `medium`, `low`, none) và "Preserve Thinking" (giữ thinking trace của lượt trước).
- Phần cứng: 27B 4-bit chạy trên 16–19 GB VRAM (RTX 5080, 4090) hoặc Mac 24 GB RAM. Bản NVFP4 chạy trên 24 GB VRAM và cần GPU Blackwell.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B |
|---|---|---|
| Bộ nhớ cho Qwen3.8-27B | "runs locally on 17GB RAM/VRAM", [đoạn mở đầu](https://unsloth.ai/docs/models/qwen3.8) | 4-bit: 16–19 GB (tổng RAM+VRAM), [bảng "Qwen3.8-27B Requirements"](https://unsloth.ai/docs/models/qwen3.8#recommended-settings) |
| Thư mục tải bản `UD-IQ1_S` của 2.4T | Lệnh `hf download` ghi `--local-dir unsloth/Qwen3.8-2.4T-A95B-GGFF`, [mục "Run Qwen3.8 in llama.cpp"](https://unsloth.ai/docs/models/qwen3.8#run-qwen3.8-in-llama.cpp) | Lệnh `llama-cli` ngay sau đó đọc từ `unsloth/Qwen3.8-2.4T-A95B-GGUF/UD-IQ1_S/...`, [cùng mục](https://unsloth.ai/docs/models/qwen3.8#run-qwen3.8-in-llama.cpp) |

Nếu chép nguyên hai lệnh này, thư mục tải về sẽ khác với đường dẫn mà lệnh chạy đọc vào.
:::

**Nguồn:** https://unsloth.ai/docs/models/qwen3.8
::::

:::: details Qwen3.5
Qwen3.5 là họ model đa phương thức (multimodal: nhận cả text và ảnh) với hybrid reasoning. Họ này có nhiều cỡ, từ bản nhỏ chạy trên laptop tới bản rất lớn, và trang docs có mục riêng để fine-tune Qwen3.5 với Unsloth.

Họ model chia hai nhóm:

- Nhóm Medium: 27B, 35B-A3B, 122B-A10B, 397B-A17B.
- Nhóm Small: 0.8B, 2B, 4B, 9B.

Qwen3.5 hỗ trợ 201 ngôn ngữ.

- Context: `262,144` token (mở rộng tới 1M qua YaRN). Nên để độ dài output `32,768` token cho đa số truy vấn.
- Sampling ở chế độ thinking:
  - Tác vụ chung: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5`.
  - Code cần chính xác: `temperature=0.6, presence_penalty=0.0`.
- Nhóm Small **tắt reasoning mặc định**. Bật bằng `--chat-template-kwargs '{"enable_thinking":true}'`.
- Bảng bộ nhớ 4-bit (theo bảng "Usage Guide"; các mục khác ghi khác, xem hộp cảnh báo bên dưới): 0.8B/2B 3.5 GB, 4B 5.5 GB, 9B 6.5 GB, 27B 17 GB, 35B-A3B 22 GB, 122B-A10B 70 GB, 397B-A17B 214 GB.
- Bản 397B chạy được trên một GPU 24 GB + 256 GB RAM nhờ MoE offloading, đạt 25+ token/s.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B | Chỗ C |
|---|---|---|---|
| 27B, bộ nhớ cho 4-bit | 17 GB, [bảng "Usage Guide"](https://unsloth.ai/docs/models/qwen3.5) | "works great on a 18GB RAM / Mac device", [mục "Qwen3.5-27B"](https://unsloth.ai/docs/models/qwen3.5#qwen3.5-27b) | "The 35B and 27B models work on a 22GB Mac / RAM device", [đoạn mở đầu](https://unsloth.ai/docs/models/qwen3.5) |
| 35B-A3B, bộ nhớ cho 4-bit | 22 GB, [bảng "Usage Guide"](https://unsloth.ai/docs/models/qwen3.5) | "works great on a 24GB RAM / Mac device", [mục "Qwen3.5-35B-A3B"](https://unsloth.ai/docs/models/qwen3.5#qwen3.5-35b-a3b) | 22 GB, [đoạn mở đầu](https://unsloth.ai/docs/models/qwen3.5) |
| 35B-A3B, 16-bit | BF16 70 GB, [bảng "Usage Guide"](https://unsloth.ai/docs/models/qwen3.5) | "around 72GB at full F16", [mục "Qwen3.5-35B-A3B"](https://unsloth.ai/docs/models/qwen3.5#qwen3.5-35b-a3b) | — |
| 397B-A17B, bản đầy đủ | BF16 810 GB, [bảng "Usage Guide"](https://unsloth.ai/docs/models/qwen3.5) | "~807GB on disk", [mục "Qwen3.5-397B-A17B"](https://unsloth.ai/docs/models/qwen3.5#qwen3.5-397b-a17b) | — |
:::

**Nguồn:** https://unsloth.ai/docs/models/qwen3.5
::::

:::: details Gemma 4
Gemma 4 là họ model mở của Google DeepMind (giấy phép Apache-2.0). Họ này có nhiều cỡ, từ bản nhỏ nhận được cả audio tới bản 31B mạnh nhất, và trang docs hướng dẫn fine-tune Gemma 4 trong Unsloth Studio.

Gemma 4 là model hybrid thinking, hỗ trợ hơn 140 ngôn ngữ, có cả dense lẫn MoE. Các bản và đầu vào mỗi bản nhận:

| Bản | Kiến trúc | Đầu vào |
|---|---|---|
| E2B, E4B | Dense + PLE | Text, ảnh, audio |
| 12B Unified | Dense | Text, ảnh, audio |
| 26B-A4B | MoE | Text, ảnh |
| 31B | Dense | Text, ảnh |

- Context: 128K cho E2B/E4B; `262,144` cho 12B, 26B-A4B, 31B.
- Sampling: `temperature=1.0, top_p=0.95, top_k=64`. Muốn bật thinking, thêm token `<|think|>` vào đầu system prompt.
- Audio chỉ có ở 12B, E2B, E4B, tối đa 30 giây.
- Bảng bộ nhớ 4-bit (theo bảng "Hardware requirements"; đoạn mở đầu ghi khác cho E2B/E4B, xem hộp cảnh báo bên dưới): E2B 4 GB, E4B 5.5–6 GB, 12B 7–8 GB, 26B-A4B 16–18 GB, 31B 17–20 GB.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A: [bảng "Hardware requirements"](https://unsloth.ai/docs/models/gemma-4) | Chỗ B: [đoạn mở đầu](https://unsloth.ai/docs/models/gemma-4) |
|---|---|---|
| E2B, 4-bit | 4 GB | "Run on 5GB RAM (4-bit)" (chung cho E2B và E4B) |
| E4B, 4-bit | 5.5–6 GB | 5 GB (chung cho E2B và E4B) |
| E2B, 16-bit | 10 GB (BF16/FP16) | "15GB (full 16-bit)" (chung cho E2B và E4B) |
| E4B, 16-bit | 16 GB (BF16/FP16) | 15 GB (chung cho E2B và E4B) |
:::

**Nguồn:** https://unsloth.ai/docs/models/gemma-4
::::

:::: details gpt-oss
gpt-oss gồm hai model mở của OpenAI (Apache 2.0): gpt-oss-20b và gpt-oss-120b. Cả hai đều là MoE, chuyên về reasoning và function calling. **[Nhận định]** Đây là họ có số liệu fine-tune khá đầy đủ trong docs.

Unsloth đã sửa lỗi chat template (định dạng Harmony) và hỗ trợ ba cách fine-tune: QLoRA, LoRA BF16 và RL (GRPO).

- Context: tối thiểu nên dùng `16,384`, tối đa `131,072`. Một lệnh ví dụ cho 120b lại dùng giá trị khác, xem hộp cảnh báo bên dưới.
- Sampling (OpenAI khuyến nghị): `temperature=1.0, top_p=1.0, top_k=0`. Có `reasoning_effort` low/medium/high.
- Bộ nhớ khi fine-tune:
  - QLoRA: 20b cần 14 GB VRAM, 120b cần 65 GB VRAM.
  - LoRA BF16: 20b cần 44 GB, 120b cần 210 GB.
  - Các cách train khác (upcast lên bf16) cần tối thiểu 65 GB VRAM cho bản 20b.
- Dataset: muốn giữ khả năng reasoning thì trộn ít nhất 75% mẫu reasoning và 25% mẫu không reasoning.
- Phần cứng: trọng số gốc ở MXFP4. Trên GPU cũ như T4 (không hỗ trợ bfloat16), Unsloth dùng float32 để tránh tràn số.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B |
|---|---|---|
| Context length | "Maximum context length window: 131,072", [mục "Recommended Settings"](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#recommended-settings) | Hướng dẫn lệnh gpt-oss-120b bảo sửa "`--ctx-size` 262114 for context length", [mục "Run gpt-oss-120b", bước 5](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune#run-gpt-oss-120b) |
:::

**Nguồn:** https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune
::::

:::: details DeepSeek-V4
DeepSeek-V4 là họ model MoE rất lớn, context 1M. Ngay cả bản nhỏ nhất trong bảng bên dưới (3-bit `UD-IQ3_XXS`) cũng cần ≥ 110 GB RAM.

Họ này gồm ba bản:

- V4-Flash: 284B tổng, 13B kích hoạt.
- V4-Pro: 1.6T, 49B kích hoạt.
- Flash-Vision-Exp.

Trọng số gốc của Flash lưu các routed expert (96% model) ở MXFP4. Vì vậy bản `UD-Q8_K_XL` của Unsloth giữ nguyên trọng số bit-for-bit, và docs gọi bản này là **lossless** (không mất chất lượng).

- Context tối đa: `1,048,576`. Chế độ Think Max cần context ít nhất 384K.
- Ba chế độ: Non-think, Think High (mặc định), Think Max.
- Sampling: `temperature=1.0, top_p=1.0`. Với Flash-0731 trong tác vụ agent, dùng `top_p=0.95`.
- Phần cứng: docs khuyên dùng `UD-IQ3_XXS` (103 GB) với ít nhất 110 GB RAM. Các chỗ khác trong trang ghi 110–135 GB hoặc máy 128 GB, xem hộp cảnh báo bên dưới.
- DSpark (speculative decoding, tăng tốc decode tới ~2×) cần thêm ~10 GB khi bật.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B | Chỗ C |
|---|---|---|---|
| `UD-Q8_K_XL` (lossless) | File 162 GB, nên có ≥ 169 GB RAM/VRAM, [mục "Usage Guide"](https://unsloth.ai/docs/models/deepseek-v4#usage-guide) | Bảng phần cứng: cột "4-bit (near Lossless)" 162 GB, cột "Q8_K_XL (Lossless)" 169 GB, [mục "Usage Guide"](https://unsloth.ai/docs/models/deepseek-v4#usage-guide) | Bảng GGUF Benchmarks: Q8 161.9 GB, Q4 155.1 GB, [mục "GGUF Benchmarks"](https://unsloth.ai/docs/models/deepseek-v4#gguf-benchmarks); đoạn mở đầu: Q8 "only 7GB larger" than Q4 |
| 3-bit `UD-IQ3_XXS` | File 103 GB, cần ≥ 110 GB RAM, [mục "Usage Guide"](https://unsloth.ai/docs/models/deepseek-v4#usage-guide) | Bảng phần cứng: 3-bit 110–135 GB (bật DSpark: 120–145 GB), [cùng mục](https://unsloth.ai/docs/models/deepseek-v4#usage-guide) | "fits on a 128GB RAM device", [mục "Run DeepSeek-V4-Flash Tutorials"](https://unsloth.ai/docs/models/deepseek-v4#run-deepseek-v4-flash-tutorials) |
| Fine-tune | "can now be run and trained in Unsloth", [mục "Unsloth Guide"](https://unsloth.ai/docs/models/deepseek-v4#unsloth-studio-guide) | Trang không có hướng dẫn, notebook hay số VRAM để fine-tune | — |
:::

**Nguồn:** https://unsloth.ai/docs/models/deepseek-v4
::::

:::: details GLM-5.3
GLM-5.3 là model MoE lớn của Z.ai, luôn bật thinking. Model có 744B tham số tổng, 40B kích hoạt, context 1M, và dùng chung base model với GLM-5.2.

GLM-5.3 có ba mức thinking: Low, High, Max. Thinking **không tắt được**. Nếu muốn chạy bản nhỏ hơn, hãy xem trang riêng của GLM-5.3-Flash.

- Context tối đa: `1,048,576`.
- Sampling: mặc định `temperature=1.0, top_p=0.95`. Tác vụ agent dài dùng `top_p=1.0`. Docs khuyên đặt `clear_thinking=true` cho hội thoại nhiều lượt.
- Chất lượng bản lượng tử hóa, đo bằng top-1 accuracy (tỉ lệ trùng token dự đoán hàng đầu với model gốc):
  - Dynamic 1-bit đạt ~76% và nhỏ hơn 85%.
  - 2-bit đạt ~81%.
- Phần cứng: `UD-IQ2_M` (239 GB) chạy tốt trên máy 256 GB RAM, như 2× DGX Spark hoặc Mac Studio.

**Nguồn:** https://unsloth.ai/docs/models/glm-5.3
::::

:::: details Kimi K3
Kimi K3 là model MoE rất lớn của Moonshot AI, chỉ có chế độ thinking. Model có 2.8T tham số tổng, 104B kích hoạt, vision gốc, context 1M, trọng số MoE ở MXFP4.

Kimi K3 là model **thinking-only**: luôn bật `preserve_thinking`, mặc định ở mức thinking max, và không có chế độ instant. Muốn dùng vision, bạn cần bản fork llama.cpp của Unsloth.

- Context: tới `1,048,576`.
- Sampling: mặc định `temperature=1.0, top_p=0.95`. Tác vụ agent dùng `top_p=1.0`.
- Docs khuyên dùng `UD-IQ1_S` (594 GB, ~78.9% top-1 accuracy) vì bản này cân bằng giữa dung lượng và chất lượng.
- Phần cứng: nếu model vừa bộ nhớ, trên B200 đạt ~20 token/s. Quy tắc chung: RAM+VRAM ≈ kích thước quant. Thiếu bộ nhớ thì vẫn chạy, nhưng chậm hơn nhiều do phải offload xuống ổ đĩa.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B |
|---|---|---|
| Bản 2-bit | "Dynamic 2-bit 861.3GB", [đoạn mở đầu](https://unsloth.ai/docs/models/kimi-k3) | Bảng phần cứng: 2-bit XXS 726 GB, 2-bit XL 880 GB, [đầu trang](https://unsloth.ai/docs/models/kimi-k3); bảng Quantization Analysis: `UD-IQ2_XXS` 711.1 GB |
| `UD-Q4_K_XL` | "near full precision and requires 1.56 TB RAM/VRAM", [mục "Quantization Analysis"](https://unsloth.ai/docs/models/kimi-k3) | Q8 "50GB larger than Q4", [đoạn mở đầu](https://unsloth.ai/docs/models/kimi-k3) |
| `UD-Q8_K_XL` (lossless) | Q8 (Lossless) 1.6 TB, [bảng phần cứng](https://unsloth.ai/docs/models/kimi-k3) | "the 1.56 TB lossless `UD-Q8_K_XL`", [mục "Quantization Analysis"](https://unsloth.ai/docs/models/kimi-k3) |
| Phần cứng tối thiểu | "can run on a NVIDIA DGX Station, or Mac Studio connected to a 128GB RAM device", [đoạn mở đầu](https://unsloth.ai/docs/models/kimi-k3) | `UD-IQ1_S` "will require at least 610GB RAM", [mục "Run Kimi K3 Guide"](https://unsloth.ai/docs/models/kimi-k3) |
:::

**Nguồn:** https://unsloth.ai/docs/models/kimi-k3
::::

:::: details NVIDIA Nemotron 3
Nemotron 3 là họ model của NVIDIA, hướng đến coding, math và agent. Bản nhỏ nhất vừa GPU Colab miễn phí khi fine-tune. **[Nhận định]** Vì vậy đây là lựa chọn dễ thử cho người mới.

Họ này gồm Nano-4B, Nano-30B-A3B và Super-120B-A12B, context 1M. Docs gọi Nano-4B là "hybrid MoE". Unsloth hỗ trợ fine-tune mọi model Nemotron, gồm cả Super và Nano, và hỗ trợ RL qua NeMo Gym.

- Sampling:
  - Chat chung: `temperature=1.0, top_p=1.0`.
  - Tool calling: `temperature=0.6, top_p=0.95`.
- Context mặc định là `262,144`. Đặt 1M có thể gây lỗi CUDA OOM (hết bộ nhớ GPU).
- Fine-tune:
  - Bản 4B vừa GPU Colab miễn phí.
  - Bản 30B không vừa; có notebook cho A100 80 GB.
  - Router của MoE mặc định không được fine-tune.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B |
|---|---|---|
| Bộ nhớ Nano-4B | "runs on 5GB of RAM, VRAM, or unified memory" (không nói quant), [đoạn mở đầu](https://unsloth.ai/docs/models/nemotron-3) | 4-bit ~3 GB, 8-bit 5 GB, [mục "Run Nemotron-3-Nano-4B"](https://unsloth.ai/docs/models/nemotron-3#run-nemotron-3-nano-4b) |
| Context length | "1M-token context", [đoạn mở đầu](https://unsloth.ai/docs/models/nemotron-3); `max_new_tokens` tối đa 1M, [mục "Usage Guide"](https://unsloth.ai/docs/models/nemotron-3) | Mặc định 262,144; đặt 1M "may trigger CUDA OOM and crash", [mục llama.cpp của 4B và 30B](https://unsloth.ai/docs/models/nemotron-3#run-nemotron-3-nano-30b-a3b) |
| VRAM LoRA 16-bit | "16-bit LoRA fine-tuning of Nemotron 3 Nano will use around 60GB VRAM", không nói 4B hay 30B, [mục "Fine-tuning Nemotron 3 and RL"](https://unsloth.ai/docs/models/nemotron-3#fine-tuning-nemotron-3-and-rl) | Cùng đoạn: 4B vừa GPU Colab miễn phí, 30B cần notebook A100 80 GB |
:::

**Nguồn:** https://unsloth.ai/docs/models/nemotron-3
::::

:::: details Mistral 3.5
Mistral-Medium-3.5-128B là một model **dense** lớn. Bản 3-bit đã cần 64 GB bộ nhớ (xem dòng Phần cứng bên dưới). Docs khuyên bạn bắt đầu với GGUF Dynamic 4-bit.

Model này đa phương thức: nhận text và ảnh, trả về text. Model có hybrid reasoning và context 256K.

- Context tối đa: `262,144`.
- Temperature theo mức reasoning:
  - `reasoning_effort="high"` dùng `temperature=0.7`.
  - `reasoning_effort="none"` dùng `temperature` từ 0.0 tới 0.7 tùy tác vụ.
- Phần cứng: 3-bit cần 64 GB, 4-bit cần 80 GB. Cần thêm bộ nhớ khi context dài, batch lớn hoặc có ảnh.
- Ngày 1/5/2026, GGUF được phát hành lại để sửa lỗi phân tích YaRN (lỗi không do Unsloth).

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B |
|---|---|---|
| Vision qua GGUF | "Vision for GGUFs it now supported for now. Support will come later.", [mục "Usage Guide"](https://unsloth.ai/docs/models/mistral-3.5) | Cập nhật 1/5/2026 ghi đã sửa lỗi file `mmproj` (file vision); mục tutorial chỉ cảnh báo GGUF vision không chạy trong Ollama, [mục "Run Mistral 3.5 Tutorials"](https://unsloth.ai/docs/models/mistral-3.5) |
| Bộ nhớ để bắt đầu | "Run locally on ~64GB RAM", [đoạn mở đầu](https://unsloth.ai/docs/models/mistral-3.5) | Bảng: 64 GB là mức 3-bit, 4-bit cần 80 GB; docs khuyên bắt đầu bằng Dynamic 4-bit, [mục "Usage Guide" và "Run Mistral 3.5 Tutorials"](https://unsloth.ai/docs/models/mistral-3.5) |
| Fine-tune | Mô tả trang: "to run or fine-tune locally on your device", [đầu trang](https://unsloth.ai/docs/models/mistral-3.5) | Trang không có mục, lệnh hay notebook fine-tune |
:::

**Nguồn:** https://unsloth.ai/docs/models/mistral-3.5
::::

:::: details IBM Granite 4.1
Granite 4.1 là họ model **dense** cỡ nhỏ và vừa của IBM, gồm 3B, 8B và 30B. Unsloth hỗ trợ fine-tune cả ba cỡ, và docs gọi 3B, 8B là điểm khởi đầu tốt.

Các model được train trên 15T token. Chúng nhắm tới instruction following, tool calling, chat, RAG (tra cứu tài liệu rồi trả lời) và coding. Notebook mẫu train một trợ lý hỗ trợ khách hàng. Notebook này viết cho Granite-4.0; bạn đổi tên model sang 4.1 là dùng được.

- Sampling cho kết quả tất định (deterministic): `temperature=0.0, top_p=1.0, top_k=0`.
- Context: tối thiểu nên dùng `16,384`, tối đa `131,072`.
- Docs không ghi yêu cầu bộ nhớ.

**Nguồn:** https://unsloth.ai/docs/models/ibm-granite-4.1
::::

:::: details Llama 4
Llama 4 gồm hai model của Meta: Scout (109B) và Maverick (402B). Cả hai đều hỗ trợ text và vision.

GGUF Dynamic của Unsloth chỉ hạ bit mạnh ở các lớp MoE, còn attention được giữ ở 4 hoặc 6-bit. Maverick xen kẽ lớp dense và lớp MoE.

- Sampling (Meta khuyến nghị): `temperature=0.6, top_p=0.9, min_p=0.01`.
- Docs ghi Llama 4 hỗ trợ context 10M. Lệnh ví dụ dùng `--ctx-size 16384`.
- Docs khuyên dùng bản 2.42-bit (`IQ2_XXS`) trở lên để có kết quả tốt nhất.
- Phần cứng:
  - Scout 1.78-bit vừa GPU 24 GB.
  - Maverick 1.78-bit: docs ghi hai cấu hình khác nhau (2×48 GB và 2× RTX 4090), xem hộp cảnh báo bên dưới.

::: warning Docs chưa thống nhất
| Thông số | Chỗ A | Chỗ B |
|---|---|---|
| Phần cứng cho Maverick 1.78-bit (122 GB) | "fits in 2x48GB VRAM GPUs", [đoạn mở đầu](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) | "it's best to have 2 RTX 4090s (2 x 24GB)", [mục tutorial Maverick](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) |
| Dung lượng bản không lượng tử | "The full unquantized version requires 113GB" (không nói là Scout hay chung), [đoạn mở đầu](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) | Maverick "went from 422GB", [cùng đoạn](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) |
| Maverick `Q2_K_XL` | Bảng ghi "151B" (không rõ đơn vị), [bảng Maverick](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune) | Các dòng khác trong bảng dùng GB (140 GB, 193 GB) |
| Fine-tune | Tiêu đề trang: "Llama 4: How to Run & Fine-tune" | Trang không có mục fine-tune, chỉ nhắc các bản upload bnb-4bit/8bit của Scout |
:::

**Nguồn:** https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune
::::

:::: details Qwen3-VL
Qwen3-VL là họ model vision của Qwen: đọc được ảnh, video và làm OCR. Unsloth hỗ trợ fine-tune và RL cho họ này, kèm notebook Colab miễn phí.

Họ này có bản instruct và bản thinking. Về kiến trúc, 2B, 4B, 8B, 32B là dense; 30B và 235B là MoE. Context 256K (mở rộng tới 1M). Phần hỗ trợ fine-tune gồm cả 32B, 235B, fine-tune cho video và cho object detection (nhận diện vật thể).

- Sampling:
  - Instruct: `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
  - Thinking: `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0`.
- Docs ghi train nhanh hơn 1.7×, tốn ít VRAM hơn 60%, context dài hơn 8×. Có hai notebook Colab miễn phí cho 8B: một cho SFT, một cho GRPO/GSPO.
- Bản thinking từng bị lỗi chat template sau lượt hội thoại thứ hai. Unsloth đã sửa và upload lại quant.

**Nguồn:** https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune
::::

:::: details DeepSeek-OCR 2
DeepSeek-OCR 2 là model 3B của DeepSeek để đọc ảnh và tài liệu, phát hành 27/1/2026. Model hợp với tài liệu có bố cục phức tạp như nhiều cột hoặc bảng.

Bộ mã hóa DeepEncoder V2 đọc ảnh theo thứ tự logic giống người, nhờ vậy đọc tốt hơn các bố cục phức tạp. Bạn fine-tune được qua notebook miễn phí. Docs ghi train nhanh hơn 1.4×, tốn ít VRAM hơn 40%, context dài hơn 5×.

- Thiết lập khuyến nghị: `temperature=0.0`, `max_tokens=8192`, `ngram_size=30`, `window_size=90`.
- Cần dùng bản upload đã chỉnh của Unsloth để chạy hoặc train trên `transformers` mới.
- Docs không ghi yêu cầu bộ nhớ.

**Nguồn:** https://unsloth.ai/docs/models/tutorials/deepseek-ocr-2
::::

## Chọn model thế nào

Mục này gom các gợi ý chọn model rải rác trong docs thành một danh sách câu hỏi. Bạn đi lần lượt từ mục đích sử dụng, tới bộ nhớ, rồi tới mức quant.

1. **Chạy hay fine-tune?** Theo catalog, GGUF dùng để chạy trong Unsloth Desktop và llama.cpp. Bản Instruct 4-bit safetensors dùng để chạy hoặc fine-tune qua Unsloth.
2. **So bộ nhớ trước.** Tổng RAM+VRAM nên lớn hơn file quant. Thiếu thì vẫn chạy nhưng chậm (Qwen3.5, Gemma 4, gpt-oss, Kimi K3).
3. **Dense hay MoE cùng cỡ.**
   - Qwen3.5: chọn 27B nếu cần chính xác hơn một chút; chọn 35B-A3B nếu cần suy luận nhanh hơn nhiều.
   - Gemma 4: 26B-A4B cân bằng tốc độ và độ chính xác, hợp khi ít RAM; 31B mạnh nhất nhưng chậm hơn.
4. **Theo thiết bị.** Gemma 4 E2B/E4B nhắm tới điện thoại và laptop. Model lớn nhắm tới PC có GPU NVIDIA RTX.
5. **Fine-tune local.**
   - Granite-4.1-3B và 8B là điểm khởi đầu tốt; 8B là "lựa chọn mặc định" cân bằng chất lượng và bộ nhớ.
   - Nemotron-3-Nano-4B và Qwen3-VL-8B có notebook Colab miễn phí.
   - gpt-oss-20b QLoRA vừa 14 GB VRAM.
6. **Chọn quant.**
   - Qwen3.5: dùng ít nhất `UD-Q2_K_XL`.
   - Llama 4: dùng `IQ2_XXS` trở lên.
   - DeepSeek-V4 và Kimi K3: `UD-Q8_K_XL` là bản lossless.
   - Gemma 4: bắt đầu với 8-bit cho model nhỏ và Dynamic 4-bit cho model lớn.
7. **Cần OCR hoặc vision?**
   - Tài liệu: DeepSeek-OCR 2.
   - Vision, video, OCR: Qwen3-VL.
   - Ngoài ra Gemma 4, Qwen3.5, Qwen3.8-27B, Kimi K3, Mistral Medium 3.5 cũng nhận ảnh.

**[Nhận định]** Nếu bạn mới bắt đầu và có một GPU 16–24 GB, các model vừa tầm và có con số rõ ràng trong docs là Qwen3.5-9B/27B, Gemma 4 E4B/26B-A4B/31B và gpt-oss-20b (cả để chạy lẫn QLoRA). Các model hàng trăm GB trở lên (GLM-5.3, Kimi K3, DeepSeek-V4, Qwen3.8-2.4T) chủ yếu để tham khảo nếu bạn không có máy nhiều RAM.

**[Nhận định]** Nếu mục tiêu là fine-tune, bạn nên ưu tiên model mà trang docs có notebook hoặc số VRAM cho training cụ thể (gpt-oss, Nemotron 3, Qwen3-VL, Granite 4.1, DeepSeek-OCR 2), thay vì model chỉ có hướng dẫn chạy.

**Nguồn:** https://unsloth.ai/docs/get-started/unsloth-model-catalog, https://unsloth.ai/docs/models/tutorials, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/gemma-4, https://unsloth.ai/docs/models/ibm-granite-4.1, https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune

## Đọc tiếp

- [Fine-tuning](/fine-tuning/) — Chọn được model rồi thì học cách train thêm cho model đó.
- [Chạy model và gọi API](/inference/) — Tải và chạy model đã chọn trong Studio hoặc qua API.
- [Cài đặt và phần cứng](/cai-dat) — Kiểm tra máy của bạn có đủ GPU, RAM cho model đã chọn không.
