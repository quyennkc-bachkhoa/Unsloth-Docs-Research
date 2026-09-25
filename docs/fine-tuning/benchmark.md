---
title: Hiệu năng và benchmark
description: "Con số tốc độ, VRAM và context tối đa Unsloth công bố, kèm điều kiện đo."
---

# Hiệu năng và benchmark

Trang này gom các con số tốc độ, bộ nhớ và độ dài context mà Unsloth công bố. Bạn nên đọc kỹ điều kiện đo trước khi dùng các số này để chọn phần cứng.

::: tip Tóm tắt
- **Dùng khi:** bạn cần ước lượng Unsloth nhanh hơn bao nhiêu, tiết kiệm bao nhiêu VRAM, hay train được context dài tới đâu trên GPU của mình.
- **Kết quả:** có các bảng tốc độ và context tối đa kèm điều kiện đo, và biết chỗ nào các trang docs đưa số khác nhau.
- **Nên biết trước:** QLoRA và rank là gì ở [Chọn cách train và model](/fine-tuning/chon-cach-train).
:::

## Bảng tốc độ

Bảng tốc độ được đo trong điều kiện sau:

- GPU H100 và Blackwell.
- Alpaca dataset; batch size 2; gradient accumulation 4; rank = 32.
- QLoRA trên mọi lớp linear (q, k, v, o, gate, up, down).
- Mốc so sánh 1× là Hugging Face + FA2 (Flash Attention 2).

| Model | VRAM | Tốc độ Unsloth | Giảm VRAM | Context dài hơn | Hugging Face + FA2 |
| --- | --- | --- | --- | --- | --- |
| Llama 3.3 (70B) | 80GB | 2× | trên 75% | 13× | 1× |
| Llama 3.1 (8B) | 80GB | 2× | trên 70% | 12× | 1× |

**Nguồn:** https://unsloth.ai/docs/basics/unsloth-benchmarks

## Context tối đa

Hai bảng dưới cho biết độ dài context tối đa train được theo VRAM GPU, so với Hugging Face + FA2.

### Llama 3.1 (8B) Instruct

Cấu hình: QLoRA 4-bit trên mọi lớp linear, rank = 32, batch size 1. Mọi chuỗi được pad tới độ dài tối đa để mô phỏng tải long-context. Docs không ghi rõ loại GPU cho bảng này.

| VRAM GPU | Context Unsloth | Hugging Face + FA2 |
| --- | --- | --- |
| 8 GB | 2,972 | OOM |
| 12 GB | 21,848 | 932 |
| 16 GB | 40,724 | 2,551 |
| 24 GB | 78,475 | 5,789 |
| 40 GB | 153,977 | 12,264 |
| 48 GB | 191,728 | 15,502 |
| 80 GB | 342,733 | 28,454 |

### Llama 3.3 (70B) Instruct

Đo trên A100 80GB, cùng cấu hình (QLoRA 4-bit, rank 32, batch 1).

| VRAM GPU | Context Unsloth | Hugging Face + FA2 |
| --- | --- | --- |
| 48 GB | 12,106 | OOM |
| 80 GB | 89,389 | 6,916 |

Theo docs, lượng VRAM tiết kiệm được đến từ hai thứ: thuật toán gradient checkpointing của Unsloth và thuật toán CCE của Apple.

**Nguồn:** https://unsloth.ai/docs/basics/unsloth-benchmarks

## Đọc số liệu cho đúng

Trước khi dùng các con số trên, bạn nên biết các trang docs không thống nhất với nhau, và đo tốc độ quá sớm sẽ cho kết quả sai.

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

::: warning Lỗi thường gặp
**Đo tốc độ quá sớm.** `torch.compile` thường mất khoảng 5 phút (hoặc lâu hơn) để khởi động và biên dịch. Vì vậy lúc đầu Unsloth trông có vẻ chậm. Bạn chỉ nên đo throughput **sau** khi đã nạp xong.
:::

**Nguồn:** https://unsloth.ai/docs/basics/unsloth-benchmarks, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/unsloth-notebooks

## Đọc tiếp

- [Kỹ thuật nâng cao](/fine-tuning/mo-rong) — train nhiều GPU, vision, continued pretraining và QAT khi đã quen fine-tune cơ bản.
- [Cài đặt và phần cứng](/cai-dat) — đối chiếu các con số VRAM với phần cứng bạn đang có.
- [Chọn cách train và model](/fine-tuning/chon-cach-train) — chọn QLoRA, LoRA hay full fine-tuning theo lượng VRAM.
