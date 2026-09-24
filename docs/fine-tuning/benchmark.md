---
title: Hiệu năng và benchmark
description: "Con số tốc độ, VRAM và context tối đa Unsloth công bố, kèm điều kiện đo."
---

# Hiệu năng và benchmark

Trang này gom các con số tốc độ, bộ nhớ và độ dài context mà Unsloth công bố. Bạn nên đọc kỹ điều kiện đo trước khi dùng các số này để chọn phần cứng.

**Bảng tốc độ.** Điều kiện đo:

- GPU H100 và Blackwell.
- Alpaca dataset; batch size 2; gradient accumulation 4; rank = 32.
- QLoRA trên mọi lớp linear (q, k, v, o, gate, up, down).
- Mốc so sánh 1× là Hugging Face + FA2 (Flash Attention 2).

| Model | VRAM | Tốc độ Unsloth | Giảm VRAM | Context dài hơn | Hugging Face + FA2 |
| --- | --- | --- | --- | --- | --- |
| Llama 3.3 (70B) | 80GB | 2× | trên 75% | 13× | 1× |
| Llama 3.1 (8B) | 80GB | 2× | trên 70% | 12× | 1× |

**Context tối đa — Llama 3.1 (8B) Instruct.** Cấu hình: QLoRA 4-bit trên mọi lớp linear, rank = 32, batch size 1. Mọi chuỗi được pad tới độ dài tối đa để mô phỏng tải long-context. Docs không ghi rõ loại GPU cho bảng này.

| VRAM GPU | Context Unsloth | Hugging Face + FA2 |
| --- | --- | --- |
| 8 GB | 2,972 | OOM |
| 12 GB | 21,848 | 932 |
| 16 GB | 40,724 | 2,551 |
| 24 GB | 78,475 | 5,789 |
| 40 GB | 153,977 | 12,264 |
| 48 GB | 191,728 | 15,502 |
| 80 GB | 342,733 | 28,454 |

**Context tối đa — Llama 3.3 (70B) Instruct.** Đo trên A100 80GB, cùng cấu hình (QLoRA 4-bit, rank 32, batch 1).

| VRAM GPU | Context Unsloth | Hugging Face + FA2 |
| --- | --- | --- |
| 48 GB | 12,106 | OOM |
| 80 GB | 89,389 | 6,916 |

Theo docs, lượng VRAM tiết kiệm được đến từ hai thứ: thuật toán gradient checkpointing của Unsloth và thuật toán CCE của Apple.

::: warning Docs chưa thống nhất: con số VRAM, context và điều kiện đo
Các trang docs công bố những con số khác nhau cho cùng một chỉ số. Trang này liệt kê đủ:

**Mức giảm VRAM của Unsloth:**
- "70% less memory" (không kèm điều kiện đo) — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- "trên 70%" (Llama 3.1 8B) và "trên 75%" (Llama 3.3 70B) — [Unsloth Benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)

**VRAM tối thiểu / GPU miễn phí:**
- Fine-tune hoặc RL "with just 3GB VRAM" (không ghi model, cấu hình) — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- Llama-3 8B 4-bit fine-tune được trên "a free 16GB memory GPU" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- "Colab's free 15 GB VRAM tier" — [Unsloth Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)

**Context dài hơn bao nhiêu lần:**
- "Unsloth enables 4× longer context fine-tuning" — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide); "4x longer context lengths than the best" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- 12× (Llama 3.1 8B) và 13× (Llama 3.3 70B) — [Unsloth Benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks)

**Điều kiện đo trong trang benchmark:** bảng tốc độ ghi "Tested on H100 and Blackwell GPUs" nhưng cột VRAM chỉ ghi 80GB và không tách theo GPU; bảng context Llama 3.1 (8B) không ghi GPU; bảng context Llama 3.3 (70B) ghi "80GB A100" — [Unsloth Benchmarks](https://unsloth.ai/docs/basics/unsloth-benchmarks). Cần kiểm tra lại trước khi dùng các số này để ước tính phần cứng.
:::

::: warning Đo tốc độ đúng cách
`torch.compile` thường mất khoảng 5 phút (hoặc lâu hơn) để khởi động và biên dịch. Vì vậy lúc đầu Unsloth trông có vẻ chậm. Bạn chỉ nên đo throughput **sau** khi đã nạp xong.
:::

**Nguồn:** https://unsloth.ai/docs/basics/unsloth-benchmarks, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/unsloth-notebooks
