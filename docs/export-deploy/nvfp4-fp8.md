---
title: Định dạng trọng số NVFP4 và FP8
description: "Hai cách biểu diễn trọng số bằng ít bit: NVFP4 (4-bit cho GPU Blackwell, chạy bằng vLLM/SGLang) và FP8 (chế độ lúc train RL)."
---

# Định dạng trọng số NVFP4 và FP8

NVFP4 và FP8 là hai cách lưu trọng số model bằng ít bit hơn BF16, để model nhẹ hơn và chạy nhanh hơn. Với Unsloth, NVFP4 (4-bit) là các bản quant làm sẵn để chạy trên GPU NVIDIA Blackwell, còn FP8 (8-bit) là chế độ bạn bật khi train RL.

::: tip Tóm tắt
- **Dùng khi:** bạn có GPU Blackwell và muốn chạy bản quant NVFP4, hoặc muốn train RL với FP8 để tiết kiệm VRAM.
- **Kết quả:** biết GPU nào dùng được NVFP4, VRAM cần cho từng model, lệnh chạy bằng vLLM và SGLang; biết cách bật FP8 khi load model và GPU nào hỗ trợ.
- **Nên biết trước:** [Export và deploy](/export-deploy/) (bảng các định dạng export), [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient) (phần FP8 RL).
:::

## NVFP4: 4-bit cho GPU Blackwell {#nvfp4}

Unsloth Dynamic NVFP4 là định dạng quant 4-bit dành cho **GPU NVIDIA Blackwell**. Nếu GPU của bạn đời cũ hơn, docs khuyên dùng GGUF.

::: warning Docs chưa thống nhất
Cùng một trang docs đưa ra hai danh sách GPU khác nhau:

| Thông số | Đoạn mở đầu ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) | Mục Performance Analysis ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) |
| --- | --- | --- |
| GPU được nêu | RTX 5050-5090, B200, RTX PRO 6000 "and more" | "Blackwell GPUs like RTX 50X, DGX Spark, B200, B300" |

Cả hai danh sách đều nói là Blackwell. Với GPU không nằm trong danh sách nào: cần kiểm tra lại.
:::

Unsloth làm những việc sau trong quant này:

- Giữ các layer quan trọng ở FP8 (W8A8) hoặc BF16.
- Phần còn lại dùng W4A4 (không phải W4A16) để tận dụng FP4 tensor core.
- Kèm FP8 KV cache calibration, cho context dài gấp 2.
- Có sẵn MTP tensors (phục vụ speculative decoding) trong quant.

Bảng docs ghi yêu cầu VRAM (bộ nhớ card đồ họa) như sau:

| Model | VRAM cần | Tăng tốc (docs) |
| --- | ---: | --- |
| Gemma 4 E2B | 7 GB | 1.12× so với BF16 |
| Gemma 4 E4B | 9 GB | 1.22× so với BF16 |
| Gemma 4 12B Unified | 11 GB | 1.26× so với BF16 |
| Gemma 4 26B A4B | 26 GB | 1.41× so với BF16 |
| Gemma 4 31B | 32 GB | 1.45× so với BF16 |
| Qwen3.6 27B | 24 GB | 2.5× so với NVFP4 khác |
| Qwen3.6 35B A3B | 32 GB | 1.56× so với NVFP4 khác |
| Qwen3.6 35B A3B Fast | 32 GB | 1.79× so với NVFP4 khác |

Điều kiện đo: 1x B200, 128 concurrency (128 request đồng thời). Về độ chính xác của Qwen3.6-27B trên MMLU-Pro / GPQA / AIME 2025: bản Unsloth đạt 86.25 / 86.34 / 93.12, bản BF16 đạt 85.96 / 88.13 / 93.33.

### Chạy bằng vLLM

Cài vLLM trong một venv riêng:

```bash
uv venv unsloth-nvfp4-env --python 3.13
source unsloth-nvfp4-env/bin/activate
uv pip install "vllm>=0.25.0" "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" \
    --torch-backend=auto
```

```shell
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
```

Bật MTP / speculative decoding. Decode sẽ nhanh hơn, còn throughput giảm chút:

```bash
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

::: warning Docs chưa thống nhất
Lệnh trên chép nguyên văn từ [nvfp4](https://unsloth.ai/docs/basics/nvfp4). Dòng đầu của nó **không có** `\` ở cuối. Trong khi đó, các lệnh nhiều dòng khác trong cùng trang (lệnh `uv pip install`, lệnh `sglang.launch_server`) đều có `\` để nối dòng. Nếu bạn chép y nguyên vào shell, dòng `--speculative-config` có thể bị chạy như một lệnh riêng — cần kiểm tra lại.
:::

Trên DGX Spark, bạn phải dùng backend `flashinfer_b12x`:

```shellscript
export CUTE_DSL_ARCH=sm_121a
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --moe-backend flashinfer_b12x
```

### Chạy bằng SGLang

Qwen3.6:

```bash
python -m sglang.launch_server --model-path unsloth/Qwen3.6-27B-NVFP4 --speculative-algorithm NEXTN \
     --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Gemma 4:

```bash
python -m sglang.launch_server --model-path unsloth/Gemma-4-31B-NVFP4 --speculative-algorithm NEXTN \
     --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

::: warning Docs chưa thống nhất
Tên model trong lệnh và hướng dẫn khác với tên repo trong bảng của cùng trang:

| Thông số | Trong lệnh / hướng dẫn ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) | Trong bảng Overview ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) |
| --- | --- | --- |
| Gemma 4 31B | `unsloth/Gemma-4-31B-NVFP4` (lệnh SGLang) | `unsloth/gemma-4-31B-it-NVFP4` |
| Qwen3.6 35B A3B | "you can change model name to `Qwen3.6-35-A3B-NVFP4`" | `unsloth/Qwen3.6-35B-A3B-NVFP4` |

Trước khi chạy, hãy kiểm tra tên repo chính xác trên Hugging Face.
:::

::: warning Lỗi thường gặp
Đừng tự chọn MoE backend: trên GPU thường, **không** set MoE backend. Hãy để vLLM tự chọn (docs giải thích: Marlin không hỗ trợ tốt W4A4). DGX Spark thì ngược lại: phải set `--moe-backend flashinfer_b12x`. Nếu gặp lỗi Torchcodec, cài `ffmpeg` (`sudo apt-get install -y ffmpeg`) rồi chạy lại vLLM.
:::

::: warning Docs chưa thống nhất
Cùng trang [nvfp4](https://unsloth.ai/docs/basics/nvfp4) ghi mức tốc độ khác nhau cho cùng một cấu hình hoặc backend:

| Thông số | Nguồn A (link) | Nguồn B (link) |
| --- | --- | --- |
| DGX Spark không dùng `flashinfer_b12x` | "2x SLOWER inference" — mục DGX Spark Tutorial ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) | "much slower" (mục vLLM Tutorial) và "2.5x slower inference" (mục Marlin vs Flashinfer…) ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) |
| Marlin với W4A4 | "2.5x performance degradation" / "Marlin is 2.5x slower" ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) | Bảng benchmark: unsloth 27B W4A4 marlin 105.6 decode tok/s, 2,127 thr out tok/s; cute-DSL (auto) 125.9 decode tok/s, 6,863 thr out tok/s ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) |
| Qwen3.6-35B-A3B nhanh hơn | "1.7x faster on 32GB VRAM" — mục Performance Analysis ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) | 1.56× (NVFP4) và 1.79× (NVFP4-Fast) — bảng Overview ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) |
| Gemma 4 nhanh hơn BF16 tối đa | "at most a 1.44x throughput boost" — mục Gemma 4 and others ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) | 1.45× (31B) — bảng Overview ([nvfp4](https://unsloth.ai/docs/basics/nvfp4)) |
:::

::: info Export NVFP4 từ model tự fine-tune
Trang NVFP4 chỉ hướng dẫn chạy các quant do Unsloth upload (collection `unsloth/nvfp4` trên Hugging Face). Không thấy hàm hay tùy chọn nào để xuất NVFP4 cho model bạn tự fine-tune — cần kiểm tra lại.
:::

**Nguồn:** https://unsloth.ai/docs/basics/nvfp4

## FP8: chế độ dùng lúc train {#fp8}

Với Unsloth, FP8 là một chế độ dùng lúc train, không phải một bước export. Bạn bật nó khi load model.

::: info Docs không có trang export FP8 riêng
Trong các nguồn đã đọc, không có trang hướng dẫn "xuất model fine-tune sang FP8". FP8 chỉ xuất hiện ở trang **FP8 Reinforcement Learning**. Tức là FP8 được dùng khi **train** (RL/GRPO), không phải một bước export.
:::

Trang FP8 RL nói những điểm sau:

- **Cách bật:** đặt `load_in_fp8 = True` trong `FastLanguageModel.from_pretrained`. Nếu có bản Float8 sẵn, Unsloth tự map sang bản đó. Nếu không, Unsloth convert on-the-fly (chuyển đổi ngay lúc load).
- **GPU hỗ trợ:** H100, L4, RTX 50x, RTX 40x, H200, B200 và "any NVIDIA GPU (consumer or data center grade) released after the RTX 4090". GPU T4 miễn phí của Colab **không** hỗ trợ FP8.
- **Model có sẵn:** Unsloth upload sẵn model **FP8 Dynamic** và **FP8 Block** trên Hugging Face. Bạn dùng chúng để train FP8 hoặc serve bằng vLLM hay SGLang. FP8 Dynamic train nhanh hơn và tốn ít VRAM hơn FP8 Block, đổi lại độ chính xác giảm nhẹ.

```python
from unsloth import FastLanguageModel
fp8_model = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct", # Can be any model name!
    load_in_fp8 = True, # Can be "block" for block FP8, True for row FP8, False
)
```

Chi tiết về FP8 RL, xem [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning

## Đọc tiếp

- [Chạy model đã xuất](/export-deploy/chay-model) — trang kế tiếp: chạy model trên Ollama, llama-server, vLLM hoặc LM Studio.
- [Xuất file GGUF](/export-deploy/gguf) — lựa chọn thay thế khi GPU của bạn đời cũ hơn Blackwell.
- [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient) — FP8 RL nằm trong bộ kỹ thuật giảm VRAM khi train RL.
