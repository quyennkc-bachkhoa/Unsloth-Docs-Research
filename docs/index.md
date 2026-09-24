---
layout: home
title: "Unsloth Research"
description: "Nghiên cứu Unsloth bằng tiếng Việt: chạy, fine-tune và triển khai LLM trên phần cứng của bạn, kèm nguồn cho từng thông số."
---

## Unsloth là gì

Unsloth là công cụ giúp bạn chạy và huấn luyện LLM (Large Language Model, mô hình ngôn ngữ lớn) ngay trên máy của mình. Đây là framework mã nguồn mở (open-source), đi kèm một giao diện người dùng cũng mã nguồn mở.

Docs chia Unsloth thành ba cách dùng:

| Dạng | Mô tả ngắn theo docs |
| --- | --- |
| **Unsloth Desktop** | App native cho macOS, Windows, Linux (bản Beta), xây trên Tauri |
| **Unsloth Studio** | Web UI chạy trên trình duyệt, không cần viết code (bản Beta) |
| **Unsloth Core** | Gói Python gốc, dùng code để train và inference |

Unsloth không chỉ làm việc với LLM văn bản. Bạn còn chạy và train được các loại model khác:

- model diffusion để sinh ảnh hoặc video;
- model audio, gồm TTS (chuyển chữ thành giọng nói) và nhận dạng giọng nói;
- model embedding và model vision.

::: tip Kiến thức nền
Chưa phân biệt được LLM, model embedding hay multimodal? Xem [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh).
:::

Chi tiết từng dạng: [Tổng quan kiến trúc](/tong-quan).

**Nguồn:** https://unsloth.ai/docs, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/get-started/install, https://github.com/unslothai/unsloth

## Giải quyết vấn đề gì

Unsloth gom cả chuỗi công việc "chạy model → chuẩn bị dữ liệu → train → deploy" vào một công cụ chạy local. Bảng sau ghép từng khó khăn thường gặp với thứ Unsloth cung cấp, theo docs:

| Vấn đề | Unsloth cung cấp (theo docs) |
| --- | --- |
| Train LLM tốn VRAM (bộ nhớ card đồ họa) và chậm | Trang docs chính ghi fine-tune nhanh hơn 2 lần, ít hơn 70% VRAM, "không giảm độ chính xác". README ghi mức khác theo từng model (xem hộp bên dưới) |
| Không muốn viết code train | Dùng Studio hoặc Desktop: tải tài liệu lên rồi bấm train (LoRA, full fine-tuning, pretraining). Trang Desktop ghi định dạng PDF, CSV, JSON; các trang khác liệt kê định dạng khác, xem [Tổng quan](/tong-quan) |
| Không có dataset sẵn | Data Recipes biến PDF, CSV, DOCX thành dataset qua một workflow dạng graph-node |
| Muốn dùng model local trong agent hoặc ứng dụng | Unsloth có API tương thích OpenAI và Anthropic. Lệnh `unsloth start claude` nối Claude Code, Codex với model local |
| Muốn mang model đã train đi chỗ khác | Export sang GGUF, safetensors, NVFP4, FP8... để chạy bằng llama.cpp, Ollama, vLLM, LM Studio |
| Lo dữ liệu rời khỏi máy | Không thu thập telemetry; có thể chạy hoàn toàn offline |

Docs còn cho biết Unsloth hỗ trợ inference và training cho hơn 500 model. Đội Unsloth từng phối hợp sửa lỗi nghiêm trọng cho các họ model như gpt-oss, Qwen3, Llama 4, Gemma, Phi-4.

::: warning Docs chưa thống nhất
Các trang ghi mức tăng tốc và mức tiết kiệm VRAM khi train khác nhau. Không nguồn nào nêu điều kiện đo:

| Thông số | Nguồn A: [docs](https://unsloth.ai/docs), [new/studio](https://unsloth.ai/docs/new/studio), [desktop](https://unsloth.ai/docs/desktop) | Nguồn B: bảng Free Notebooks, [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Tốc độ train | 2x nhanh hơn | 1.5x (Gemma 4 E2B, Qwen3.5 4B, Orpheus-TTS 3B) đến 2x (gpt-oss 20B, Llama 3.1 8B...) |
| Bộ nhớ | Ít hơn 70% VRAM | Ít hơn 20% (embeddinggemma 300M) đến 80% (gpt-oss 20B GRPO) |
:::

::: tip Kiến thức nền
VRAM và cách ước lượng bộ nhớ cho một model: xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho). Định dạng GGUF, FP8, NVFP4: xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

**Nguồn:** https://unsloth.ai/docs, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/new/studio, https://github.com/unslothai/unsloth

## Dành cho ai

Unsloth có lối vào riêng cho từng kiểu người dùng, từ người không viết code đến người tự viết code train:

- **Bạn muốn chạy thử LLM local mà không viết code:** docs khuyên cài app Desktop. Chọn model và mức quantization (lượng tử hóa) vừa với máy, rồi chat ngay.
- **Bạn là lập trình viên, muốn đưa model local vào ứng dụng:** gọi endpoint `/v1/chat/completions` (dùng OpenAI SDK) hoặc `/v1/messages` (dùng Anthropic SDK). Bạn cũng có thể nối agent như Claude Code, Codex, OpenCode.
- **Bạn mới bắt đầu fine-tune:** đọc mục "Fine-tuning for Beginners" trong docs. Mục này trả lời các câu như nên chọn model instruct hay base, dataset cần lớn cỡ nào, GPU có đủ VRAM không.
- **Bạn muốn tự viết code train:** dùng Unsloth Core (gói Python) hoặc notebook Colab miễn phí.

Về nền tảng, docs liệt kê macOS, Windows, Linux, WSL; GPU NVIDIA, AMD, Intel, Mac và CPU. Phần cứng cũ có thể không được hỗ trợ tốt. Các trang docs ghi khác nhau về việc train được trên phần cứng nào (xem hộp "Docs chưa thống nhất" ở [Tổng quan kiến trúc](/tong-quan)).

**[Nhận định]** Website này viết cho người đã biết Python và FastAPI nhưng mới với AI. Phần "Unsloth" tập trung vào thao tác. Các khái niệm (token, LoRA, quantization...) được tách sang route Kiến thức nền.

**Nguồn:** https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://github.com/unslothai/unsloth
