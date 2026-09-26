---
title: Host model trên máy chủ
description: "Host model đã fine-tune trên máy chủ 2×L40S cho cả đơn vị: chọn engine, chia 2 GPU, model cỡ nào vừa 96 GB."
---

# Host model trên máy chủ

Trang này chỉ nói những gì riêng cho máy **2×L40S (96 GB)**. Cách xuất model và lệnh chạy từng engine đã có ở [Export và deploy](/export-deploy/).

::: tip Tóm tắt
- **L40S:** mỗi card 48 GB, không có NVLink, hỗ trợ FP8. NVFP4 cần GPU Blackwell.
- **vLLM:** docs khuyên pipeline parallel cho GPU không NVLink, nhưng một lần đo trên 2×L40S cho tensor parallel nhanh hơn.
- **Ollama:** mặc định xử lý 1 request mỗi lúc cho mỗi model.
:::

## Engine

| | vLLM | Ollama / llama-server |
| --- | --- | --- |
| Định dạng | safetensors 16-bit, FP8 | GGUF |
| Nhiều người cùng lúc | Continuous batching | Ollama: `OLLAMA_NUM_PARALLEL`, mặc định **1** request mỗi model |

Cả vLLM và Ollama đều có API tương thích OpenAI. Lệnh chạy: [Chạy model đã xuất](/export-deploy/chay-model).

**Nguồn:** https://docs.vllm.ai/en/latest/, https://docs.ollama.com/faq, https://docs.ollama.com/api/openai-compatibility

## L40S có gì đặc biệt

**[Nguồn ngoài]** Theo trang NVIDIA: mỗi card 48 GB GDDR6, băng thông bộ nhớ 864 GB/s, **không có NVLink**, kết nối PCIe Gen4 x16 (64 GB/s hai chiều).

- **FP8:** docs vLLM ghi FP8 chạy trên GPU compute capability từ 8.9 (Ada Lovelace, Hopper, Blackwell). L40S là GPU Ada Lovelace.
- **NVFP4:** docs Unsloth ghi NVFP4 cần GPU Blackwell. Với GPU đời cũ, docs gợi ý dùng GGUF. Xem [NVFP4 và FP8](/export-deploy/nvfp4-fp8).

**Nguồn:** https://www.nvidia.com/en-us/data-center/l40s/, https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/, https://unsloth.ai/docs/basics/nvfp4

## Chia model qua hai card

Model lớn hơn 48 GB thì phải chia qua hai card. Với vLLM, docs và số đo thực tế nói ngược nhau:

| Nguồn | Nói gì |
| --- | --- |
| [Docs vLLM](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) | GPU không NVLink ("ví dụ L40S") nên dùng pipeline parallel |
| [Diễn đàn vLLM](https://discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848) | Đo trên 2×L40S: tensor parallel 0.95 req/s, pipeline parallel 0.72 req/s |

Lệnh cho hai cách:

```bash
vllm serve ./model-merged --tensor-parallel-size 2
vllm serve ./model-merged --pipeline-parallel-size 2
```

Ollama: model vừa một card thì load lên một card, không vừa thì chia cho mọi card.

**Nguồn:** https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html, https://discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848, https://docs.ollama.com/gpu

## Model cỡ nào vừa 96 GB

VRAM phải chứa trọng số model và KV cache (bộ nhớ ngữ cảnh của từng người đang hỏi). vLLM mặc định dùng 92% VRAM (`--gpu-memory-utilization 0.92`), tức khoảng 88 GB trên 96 GB.

**[Ước tính]** Tính từ `config.json` của model, giả định 30 người dùng cùng lúc và khoảng 4 GB overhead (chưa đo):

| Model | Trọng số | Ngữ cảnh mỗi người |
| --- | --- | --- |
| 70B BF16 | ~140 GB | Không vừa |
| 70B FP8 | ~70 GB | ~1.5K token |
| 70B 4-bit | ~40 GB | ~4.5K token |
| 32B BF16 | ~64 GB | ~2.7K token |
| 32B FP8 | ~32 GB | ~7K token (KV cache FP8: ~14K) |

Số thật xem ở dòng log "maximum concurrency" khi vLLM khởi động.

**Nguồn:** https://docs.vllm.ai/en/latest/configuration/engine_args/, https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/, https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json, https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/resolve/main/config.json

## Đọc tiếp

- [Tổng quan và kết luận nhanh](/tro-ly-van-ban/)
- [Chạy model đã xuất](/export-deploy/chay-model)
- [Fine-tune bằng Unsloth](/tro-ly-van-ban/fine-tune)
