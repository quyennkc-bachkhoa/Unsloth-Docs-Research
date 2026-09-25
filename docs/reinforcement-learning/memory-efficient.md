---
title: Memory-efficient RL
description: "Vì sao RL tốn bộ nhớ GPU, các kỹ thuật Unsloth dùng để giảm (chia sẻ trọng số với vLLM, Standby, FP8 RL) và các con số kèm điều kiện đo."
---

# Memory-efficient RL

RL tốn nhiều bộ nhớ GPU hơn fine-tune thường. Mục này giải thích vì sao, rồi liệt kê các kỹ thuật Unsloth dùng để giảm mức tốn đó.

**Vì sao RL tốn bộ nhớ:** GRPO sinh văn bản rất nhiều, và việc sinh này chạy bằng vLLM (engine inference, tức engine chạy suy luận). Vì vậy GPU phải giữ cùng lúc hai "bộ":

1. Inference engine: trọng số model và KV cache.
2. Training engine: trọng số model, activation, gradient, optimizer state.

Theo docs, các framework khác thường chia GPU 80GB theo tỉ lệ 50/50 cho hai engine.

::: tip Kiến thức nền
KV cache là gì? Xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling). FP8/BF16: xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

**Unsloth tối ưu gì:**

| Kỹ thuật | Tác dụng theo docs |
| --- | --- |
| Chia sẻ vùng trọng số với vLLM | Bỏ việc giữ hai bản trọng số model. Ví dụ GPU 80GB giải phóng 16GB; tiết kiệm khoảng 5GB với Llama 3.1 8B và 3GB với Llama 3.2 3B |
| Unsloth Standby | RL xen kẽ inference và training nên dùng lại chung một vùng nhớ: xóa KV cache khi train nhưng giữ trọng số đang chia sẻ. Ví dụ GPU 80GB: 16GB trọng số chia sẻ + 64GB dùng chung cho cả hai engine |
| Linear kernel tiết kiệm bộ nhớ cho GRPO | Giảm bộ nhớ từ 8 lần trở lên, bớt 68.5GB |
| Unsloth gradient checkpointing | Chuyển activation sang RAM hệ thống, chỉ chậm hơn 1%, bớt 52GB |
| FP8 RL (`load_in_fp8 = True`) | Ít hơn 60% VRAM, context dài hơn 10 lần so với các cách làm FP8 RL khác; inference nhanh hơn khoảng 1.4 lần |

**Con số chính (ghi rõ điều kiện):**

| Kết quả | Điều kiện |
| --- | --- |
| 54.3GB so với 510.8GB (ít hơn 90%) | Llama 3.1 8B, context 20K, 8 generation mỗi prompt, so với cách làm chuẩn + Flash Attention 2 |
| Context 6,144 so với 3,600 trước đây (dài hơn 1.7 lần) | Qwen3-32B LoRA 16-bit, 1 GPU H100 80GB, có Standby |
| Context 47,500 so với 42,000 (dài hơn 1.13 lần) | Llama-3.1-8B QLoRA 4-bit |
| Tiết kiệm 2GiB (15%) | Qwen3 4B trên T4, `vllm_gpu_util 0.7`, 2 generation, có Standby so với không |
| Fit 10K context so với 6K khi không có Standby | A100 40GB, Qwen-2.5-3B-Instruct, 8 generation, LoRA 16-bit |
| RL nhanh hơn 10%, thời gian `torch.compile` nhanh hơn 2 lần | Công bố chung, không nêu cấu hình cụ thể |

Lưu ý khi đọc bảng: với GRPO, context 6,144 của Qwen3-32B thực chất là 6,144 × 2 generation = 12,288.

**Cách bật Standby.** Docs dặn đặt biến này trước mọi lệnh import Unsloth:

```python
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
```

::: warning Docs chưa thống nhất
Các chỗ trong docs ghi mức tiết kiệm bộ nhớ và mức tăng context khác nhau:

| Thông số | Nguồn A | Nguồn B |
| --- | --- | --- |
| Mức giảm VRAM của Unsloth khi làm RL | "reduces VRAM usage by 50–90%" so với các cách làm dùng FA2 ([Memory Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)) | "over 90%" / "90% less", đo với Llama 3.1 8B, context 20K, 8 generation ([RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)) |
| Mức tiết kiệm khi bật Standby | Comment trong code: "Unsloth standby saves 30%+ memory for RL" ([FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)) | 2GiB, tức 15%, với Qwen3 4B trên T4, `vllm_gpu_util 0.7`, 2 generation ("can be higher for longer sequences") ([Memory Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)) |
| Mức tăng context | "1.2 to 1.7x increased context lengths" ([Memory Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl), phần mở đầu) | Llama-3.1-8B QLoRA 4-bit: 47,500 so với 42,000, tức 1.13x ([Memory Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl), cùng trang) |
:::

Khi có Standby, bạn chỉ cần đặt `gpu_memory_utilization` ở 0.9 hoặc 0.95. Không còn phải dò từ 30% đến 95% như trước. Đừng đặt 100%, vì cần chừa chỗ cho các tensor nhỏ. Theo docs, mọi notebook GRPO của Unsloth đã bật sẵn Standby.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning
