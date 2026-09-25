---
title: Export & deploy
description: Unsloth hỗ trợ xuất model sau fine-tune sang những định dạng nào (LoRA, merged 16-bit, GGUF, NVFP4…) và cách đưa lên Ollama, llama-server, vLLM, LM Studio, LAN, Cloudflare.
---

# Export & deploy

Sau khi fine-tune (tinh chỉnh) xong, bạn có một checkpoint (bản lưu model trong lúc train). Checkpoint này gồm model gốc cộng với LoRA adapter (phần trọng số nhỏ học thêm). Việc tiếp theo là xuất (export) model ra một định dạng, rồi đưa nó lên một engine (phần mềm chạy model) để dùng.

Trang này chỉ trả lời ba câu hỏi: **Unsloth xuất được ra định dạng gì, bằng lệnh nào, và định dạng đó chạy trên engine nào**.

::: tip Kiến thức nền
Trang này không giải thích FP16/BF16, FP8, FP4/NVFP4, GGUF quant (Q4_K_M, Q8_0…) hay Dynamic quants là gì. Phần đó nằm ở [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa). Về LoRA adapter và merge, xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora). Về chat template, xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

## Tổng quan luồng export

Sơ đồ dưới cho thấy từ một model đã fine-tune, bạn có thể xuất theo những nhánh nào, và mỗi nhánh dẫn tới engine nào.

<div class="dg">
<div class="dg-stages">
<div class="dg-stage"><div class="dg-node is-main">Model sau fine-tune<small>base + LoRA</small></div></div>
<div class="dg-stage">
<div class="dg-rows">
<div class="dg-map"><div class="dg-node">LoRA adapter<small><code>save_pretrained</code> / <code>save_method='lora'</code></small></div><div class="dg-chips"><span class="dg-chip">vLLM / SGLang</span></div></div>
<div class="dg-map"><div class="dg-node">Merged 16-bit<small><code>save_method='merged_16bit'</code></small></div><div class="dg-chips"><span class="dg-chip">vLLM / SGLang</span><span class="dg-chip is-ghost">llama.cpp <code>convert_hf_to_gguf.py</code> → GGUF</span></div></div>
<div class="dg-map is-single"><div class="dg-node">Merged 4-bit<small><code>save_method='merged_4bit'</code></small></div></div>
<div class="dg-map"><div class="dg-node is-main">GGUF<small><code>save_pretrained_gguf</code></small></div><div class="dg-chips"><span class="dg-chip">llama.cpp / llama-server</span><span class="dg-chip">Ollama</span><span class="dg-chip">LM Studio</span><span class="dg-chip">Unsloth Studio / API</span></div></div>
</div>
</div>
</div>
<div class="dg-title">Ngoài luồng export</div>
<div class="dg-map" style="--dg-mapw: 260px"><div class="dg-node is-ghost">Quant NVFP4 do Unsloth upload<small>không phải export từ fine-tune</small></div><div class="dg-chips"><span class="dg-chip">vLLM / SGLang</span></div></div>
</div>

[Nhận định] Sơ đồ được gom từ nhiều trang docs. Nhánh NVFP4 được vẽ tách riêng vì một lý do: nguồn chỉ hướng dẫn **chạy** các quant NVFP4 mà Unsloth đã upload sẵn. Nguồn không có hướng dẫn tự xuất NVFP4 từ model bạn fine-tune.

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/nvfp4

## Bảng định dạng export

Bảng này đặt các định dạng export cạnh nhau. Với mỗi định dạng, bạn thấy engine nào chạy được nó, cách xuất trong Studio và trong Core API (thư viện Python của Unsloth), cùng lưu ý chính.

| Định dạng | Dùng cho engine nào | Cách xuất | Ghi chú |
| --- | --- | --- | --- |
| LoRA adapter | vLLM (docs có trang LoRA Hot Swapping), Unsloth | Studio: **LoRA Only**. Core: `model.save_pretrained(...)` + `tokenizer.save_pretrained(...)`, hoặc `save_pretrained_merged(..., save_method = "lora")`, `push_to_hub_merged(..., save_method = "lora")` | Chỉ chứa trọng số adapter (~100MB theo trang Ollama). Khi chạy vẫn cần model gốc |
| Merged 16-bit (safetensors) | vLLM, SGLang, transformers. Cũng là đầu vào khi bạn tự convert sang GGUF | Studio: **Merged Model**. Core: `save_pretrained_merged(..., save_method = "merged_16bit")`, `push_to_hub_merged(..., save_method = "merged_16bit")` | LoRA đã được gộp vào trọng số gốc |
| Merged 4-bit | Hugging Face (inference online), DPO training | Core: `save_method = "merged_4bit"`, sau đó `merged_4bit_forced` nếu chắc chắn | Docs **không khuyến khích**, trừ khi bạn biết rõ mục đích |
| GGUF | llama.cpp, llama-server, Ollama, LM Studio, Unsloth Studio | Studio: **GGUF / llama.cpp**. Core: `save_pretrained_gguf(...)`, `push_to_hub_gguf(...)`. Hoặc làm thủ công qua `convert_hf_to_gguf.py` | Cần chọn `quantization_method` (xem mục GGUF) |
| NVFP4 (Unsloth Dynamic NVFP4) | vLLM, SGLang trên GPU Blackwell | Không thấy hàm export NVFP4 trong nguồn — cần kiểm tra lại | Nguồn chỉ có các quant NVFP4 do Unsloth upload sẵn |
| FP8 | vLLM, SGLang (theo trang FP8 RL) | Không có trang export FP8 riêng — xem mục FP8 | Unsloth có upload sẵn bản FP8 Dynamic và bản FP8 Block |

::: info safetensors vs .bin trong Colab
Trang Troubleshooting ghi: trong Colab, Unsloth lưu file `.bin` vì cách này nhanh hơn ~4 lần. Muốn ép lưu `.safetensors`, bạn đặt `safe_serialization = None`. Ví dụ: `model.save_pretrained(..., safe_serialization = None)` hoặc `model.push_to_hub(..., safe_serialization = None)`.
:::

**Nguồn:** https://unsloth.ai/docs/new/studio/export, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama, https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning, https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference

## Export trong Unsloth Studio

Unsloth Studio (giao diện web và desktop của Unsloth) cho bạn export bằng vài lần bấm chọn. Studio xuất được checkpoint bạn đã train. Studio cũng convert được một model bất kỳ sang GGUF, Safetensors hoặc LoRA. Quy trình gồm bốn bước:

1. **Select Training Run** — chọn lượt train. Mỗi run là một phiên train và có thể có nhiều checkpoint.
2. **Select Checkpoint** — chọn checkpoint cần xuất. Checkpoint cuối thường là model hoàn chỉnh, nhưng bạn xuất checkpoint nào cũng được.
3. **Export Methods** — chọn một trong ba kiểu:

   | Kiểu export | Kết quả |
   | --- | --- |
   | Merged Model | Model **16-bit**, LoRA adapter đã gộp vào trọng số gốc |
   | LoRA Only | **Chỉ trọng số adapter**, khi chạy cần model gốc |
   | GGUF / llama.cpp | Chuyển sang **GGUF** để chạy trong Unsloth, llama.cpp, Ollama hoặc LM Studio |

4. **Nơi lưu**:
   - **Export / Save Locally** — tải file về máy.
   - **Push to Hub** — đẩy lên Hugging Face Hub. Bước này cần Hugging Face write token. Nếu bạn đã đăng nhập Hugging Face CLI thì có thể để trống ô token.

**Nguồn:** https://unsloth.ai/docs/new/studio/export

## GGUF

GGUF là định dạng file của llama.cpp. Ollama, LM Studio và Unsloth Studio đều đọc được GGUF. Bạn có hai cách tạo file GGUF: gọi hàm có sẵn của Unsloth (Core API), hoặc tự convert thủ công bằng llama.cpp.

### Lưu GGUF bằng Core API

Lưu cục bộ:

```python
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "f16")
```

Đẩy lên Hugging Face Hub:

```python
model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method = "q8_0")
```

### Các `quantization_method` được hỗ trợ

Tham số `quantization_method` quyết định file GGUF được lượng tử hóa ra sao. Docs liệt kê các giá trị sau (chép nguyên văn):

```python
# https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19
ALLOWED_QUANTS = \
{
    "not_quantized"  : "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized" : "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized"      : "Recommended. Slow conversion. Fast inference, small files.",
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.wv and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    "iq2_xxs" : "2.06 bpw quantization",
    "iq2_xs"  : "2.31 bpw quantization",
    "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
}
```

Trang LM Studio gợi ý cách chọn nhanh:

- `q4_k_m`: thường là mặc định khi chạy local.
- `q8_0`: gần như giữ nguyên chất lượng.
- `f16`: lớn và chậm nhất, nhưng không lượng tử hóa.

Trang Ollama lưu ý thêm: export `Q8_0` thì nhanh. Nếu bạn bật nhiều dòng quant cùng lúc, bạn sẽ phải chờ rất lâu (quá trình convert mất 5–10 phút).

### Lưu GGUF thủ công

Cách thủ công gồm ba bước: lưu model dạng merged 16-bit, build llama.cpp, rồi dùng script của llama.cpp để convert sang GGUF.

Bước 1 — lưu merged 16-bit:

```python
model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)
```

Bước 2 — build llama.cpp từ source:

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
```

Bước 3 — convert sang F16. Bạn cũng có thể chọn BF16 hoặc Q8_0:

```bash
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-F16.gguf --outtype f16 \
    --split-max-size 50G
```

```bash
# For BF16:
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-BF16.gguf --outtype bf16 \
    --split-max-size 50G
    
# For Q8_0:
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-Q8_0.gguf --outtype q8_0 \
    --split-max-size 50G
```

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama

## Unsloth Dynamic 3.0 GGUFs

Mục này nói về các file GGUF **do Unsloth tự lượng tử hóa và upload** lên Hugging Face. Bạn nhận ra chúng qua tiền tố `UD-`, ví dụ `UD-Q4_K_XL`, `UD-Q2_K_XL`. Đây không phải một giá trị `quantization_method` mà bạn chọn khi tự save model.

Theo docs, Unsloth làm những việc sau:

- **Dynamic v2.0**:
  - Không lượng tử hóa đồng đều mọi layer. Unsloth chọn kiểu quant riêng cho từng layer.
  - Mỗi model có một cấu hình riêng, ví dụ Gemma 3 khác Llama 4.
  - Dùng bộ calibration (dữ liệu hiệu chuẩn) hơn 1.5M token, được chọn lọc thủ công để tối ưu cho hội thoại.
  - Áp dụng cho cả model MoE và non-MoE.
  - Có thêm các quant Q4_NL, Q5.1, Q5.0, Q4.1, Q4.0 cho Apple Silicon và ARM.
- **Dynamic v3.0**:
  - Dùng bộ imatrix calibration chất lượng cao hơn, nhắm tới agentic coding, chat và đa ngôn ngữ.
  - Cải thiện cách chọn layer.
  - Không train trên bộ calibration, không dùng QAT hay QAD. Đây chỉ là post-training quantization (lượng tử hóa sau huấn luyện).
  - Với model lớn, Unsloth vẫn dùng UD-2 cũ.
- Các file này chạy được trên llama.cpp, Unsloth Studio / Unsloth Desktop và "hầu hết inference engine".

Số liệu benchmark bên dưới luôn phải đọc kèm điều kiện đo:

| Tuyên bố | Điều kiện đo theo docs |
| --- | --- |
| Dynamic v3.0 tốt hơn **trên 10% top-1 accuracy ở cùng kích thước** so với mọi provider khác | Model Qwen3.8-27B, so sánh KLD top-1 |
| `UD-Q2_K_XL` chính xác hơn ~8% top-1 so với bản tốt nhất kế tiếp, nặng 9.83GB | Qwen3.8-27B |
| `UD-IQ1_S` 6.2GB giữ ~72% top-1, nhỏ hơn 89% | Qwen3.8-27B, không kèm MTP |
| Divergence-300 @32 rơi từ ~25% (UD-Q2_K_XL) xuống dưới 8–10% (UD-IQ2_S) | 300 prompt không nằm trong calibration, greedy decode 32 token so với BF16 |
| Q2_K_XL giảm KLD ~7.5% | Gemma 3 27B, Dynamic v2.0 so với baseline imatrix, test trên Wikipedia |
| Dynamic 4-bit nhỏ hơn 2GB và +1% accuracy so với bản QAT của Google | Gemma 3 27B, MMLU 5-shot: Q4_K_XL 71.47 (15.64GB) vs Google QAT 70.64 (17.2GB) |
| Dynamic 3-bit DeepSeek V3.1 đạt 75.6% | Aider Polyglot |

::: warning Không dùng 1-bit cho agentic
Docs khuyến cáo không dùng quant thấp hơn `UD-Q2_K_XL` (tức 1-bit) cho tool calling hoặc agent. Các lỗi hay gặp:
- Dễ lặp vô hạn. Khi đó dùng `presence_penalty = 1.5` trở lên.
- Trả lời rỗng nếu tắt thinking.
- Gọi tool sai.

Chỉ dùng các quant này cho câu hỏi kiến thức ngắn. Lựa chọn tốt nhất vẫn là `UD-Q2_K_XL`.
:::

**Nguồn:** https://unsloth.ai/docs/basics/dynamic-3.0-ggufs

## NVFP4 (Unsloth Dynamic NVFP4)

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

::: warning Đừng tự chọn MoE backend
Trên GPU thường, **không** set MoE backend. Hãy để vLLM tự chọn (docs giải thích: Marlin không hỗ trợ tốt W4A4). DGX Spark thì ngược lại: phải set `--moe-backend flashinfer_b12x`. Nếu gặp lỗi Torchcodec, cài `ffmpeg` (`sudo apt-get install -y ffmpeg`) rồi chạy lại vLLM.
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

## FP8

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

Chi tiết về FP8 RL, xem [Memory-efficient RL](/reinforcement-learning/memory-efficient).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning

## Deploy: Ollama

Unsloth đưa model sang Ollama qua đường GGUF. Điểm tiện là Unsloth **tự tạo `Modelfile`**. Đây là file cấu hình Ollama cần, trong đó có chat template bạn đã dùng lúc fine-tune.

Quy trình theo docs:

1. Cài Ollama (trong notebook Colab).
2. Export model sang GGUF. Chỉ bật `True` cho **một** dòng quant, thường là dòng đầu `Q8_0`.
3. Chạy Ollama ở chế độ nền. Nếu không dùng Colab, bạn chỉ cần chạy `ollama serve` trong terminal.
4. Dùng `Modelfile` do Unsloth sinh ra để tạo model Ollama, rồi gọi inference.

::: info Code của trang Ollama bị thiếu
Đã đối chiếu cả bản gốc chưa lọc của trang. Các bước cài Ollama, export GGUF, in `Modelfile`, tạo model và gọi inference đều chỉ có dạng ảnh, không có khối code văn bản. Lệnh duy nhất ở dạng văn bản là `ollama serve`. Chi tiết xem tutorial "Finetune Llama-3 and Use In Ollama" — cần kiểm tra lại.
:::

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama

## Deploy: llama-server (OpenAI endpoint)

`llama-server` là server của llama.cpp. Nó phục vụ file GGUF qua một endpoint tương thích OpenAI, nên bạn gọi model bằng thư viện `openai` như gọi API thông thường.

Đầu tiên, build llama.cpp như ở mục GGUF. Nếu chỉ chạy CPU hoặc Mac/Metal, đổi `-DGGML_CUDA=ON` thành `-DGGML_CUDA=OFF`. Ví dụ trong docs dùng Devstral 2. Bước tải model:

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Devstral-2-123B-Instruct-2512-GGUF",
    local_dir = "Devstral-2-123B-Instruct-2512-GGUF",
    allow_patterns = ["*UD-Q2_K_XL*", "*mmproj-F16*"],
)
```

Chạy server:

```bash
./llama.cpp/llama-server \
    --model Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf \
    --mmproj Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf \
    --alias "unsloth/Devstral-Small-2-24B-Instruct-2512" \
    --threads -1 \
    --n-gpu-layers 999 \
    --prio 3 \
    --min-p 0.01 \
    --ctx-size 16384 \
    --port 8001 \
    --jinja
```

::: warning Docs chưa thống nhất
Trên cùng trang [llama-server & OpenAI endpoint](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint), bước tải và bước serve dùng **hai model khác nhau**. Vì vậy nếu chép y nguyên, server sẽ không tìm thấy file `--model`:

| Thông số | Bước tải `snapshot_download` ([nguồn](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint)) | Bước `llama-server` ([nguồn](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint)) |
| --- | --- | --- |
| Model | `unsloth/Devstral-2-123B-Instruct-2512-GGUF` | `Devstral-Small-2-24B-Instruct-2512` |
| Thư mục | `Devstral-2-123B-Instruct-2512-GGUF` | `Devstral-Small-2-24B-Instruct-2512-GGUF` |
| Quant | `*UD-Q2_K_XL*` | `UD-Q4_K_XL.gguf` |

Khi làm theo, hãy cho `--model`/`--mmproj` trỏ đúng file bạn đã thực sự tải về.
:::

Gọi server từ Python (sau khi chạy `pip install openai`):

```python
from openai import OpenAI
import json
openai_client = OpenAI(
    base_url = "http://127.0.0.1:8001/v1",
    api_key = "sk-no-key-required",
)
completion = openai_client.chat.completions.create(
    model = "unsloth/Devstral-Small-2-24B-Instruct-2512",
    messages = [{"role": "user", "content": "What is 2+2?"},],
)
print(completion.choices[0].message.content)
```

::: warning `--jinja` chèn thêm system message
Khi bật `--jinja` và model hỗ trợ tools, llama-server tự nối thêm câu "Respond in JSON format, either with tool_call … or with response …" vào system message. Câu chèn thêm này có thể làm hỏng model fine-tune. Nếu tắt bằng `--no-jinja`, bạn mất hỗ trợ `tools`. Docs khuyên thêm một prompt tool calling riêng cho mọi fine-tune.
:::

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint

## Deploy: vLLM

vLLM chạy model ở dạng merged 16-bit. Quy trình gồm ba bước: cài vLLM, lưu model dạng merged 16-bit, rồi serve thư mục model đó.

Cài đặt (GPU NVIDIA):

```bash
pip install --upgrade pip
pip install uv
uv pip install -U vllm --torch-backend=auto
```

Với GPU AMD, dùng Docker image nightly `rocm/vllm-dev:nightly`.

Lưu model cho vLLM (merged 16-bit):

```python
model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "merged_16bit")
## OR to upload to HuggingFace:
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
```

Serve model ở một terminal khác:

```bash
vllm serve finetuned_model
```

Nếu lệnh trên không chạy, dùng đường dẫn đầy đủ:

```bash
vllm serve /mnt/disks/daniel/finetuned_model
```

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide

## Deploy: LM Studio

LM Studio là ứng dụng chạy file GGUF. Quy trình có ba bước: export GGUF từ Unsloth, import vào LM Studio, rồi load model để chat hoặc bật API tương thích OpenAI.

Import file GGUF bằng CLI `lms`:

```bash
lms import /path/to/model.gguf
```

Nếu bạn đã `push_to_hub_gguf`, có thể tải thẳng từ Hugging Face:

```bash
# Download from HF by repo name
lms get hf_username/my_model_gguf

# Pick a quantization with @
lms get hf_username/my_model_gguf@Q4_K_M
```

Load model và bật server. Địa chỉ mặc định thường là `http://localhost:1234/v1`:

```bash
lms load <model-identifier> --gpu=auto --context-length=8192
```

```bash
lms server start --port 1234
```

```bash
curl http://localhost:1234/v1/models
```

Mẹo debug template: chạy `lms log stream` để xem prompt thô mà LM Studio gửi vào model.

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio

## Truy cập qua LAN và remote (Cloudflare)

Unsloth Studio có sẵn hai chế độ cho thiết bị khác dùng model đang chạy trên máy bạn. Chế độ LAN dành cho thiết bị cùng mạng nội bộ. Chế độ remote qua Cloudflare dành cho truy cập từ bất kỳ đâu qua internet.

### LAN (mạng nội bộ)

Thiết bị dùng cùng Wi-Fi hoặc cùng mạng dây truy cập qua địa chỉ dạng `http://192.168.1.42:8888`. Chế độ này không cần internet, và dữ liệu không rời mạng nội bộ.

- Bật lúc khởi động: thêm `-H 0.0.0.0`.

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

- Bật khi Studio đang chạy: mở trang cài đặt Remote & LAN (đường dẫn menu xem hộp bên dưới), vào thẻ **LAN access**, bấm **Start**. Trạng thái **Online** nghĩa là địa chỉ đã phản hồi. Toggle **Start automatically** giúp tự bật LAN mỗi lần khởi động.

### Remote qua Cloudflare tunnel

Unsloth tạo một link HTTPS kiểu `https://<random>.trycloudflare.com`. Bạn không cần tài khoản Cloudflare, domain hay mở port trên router.

```bash
unsloth studio --secure -p 8888
```

Cờ `--secure` giữ Unsloth bind ở `127.0.0.1` và chỉ publish ra ngoài qua tunnel. Nếu tunnel lỗi, Unsloth **thoát** chứ không fallback sang port thô. Khi Studio đang chạy, bạn tìm thẻ **Remote access**, bấm **Start**, rồi copy **Remote URL** hoặc quét mã QR.

::: warning Docs chưa thống nhất
Đường dẫn menu tới thẻ LAN access và thẻ Remote access được ghi khác nhau ở các chỗ:

| Chỗ trong docs | Đường dẫn menu |
| --- | --- |
| Trang LAN, đoạn mở đầu và Quickstart ([lan](https://unsloth.ai/docs/basics/lan)) | Settings → API → **Remote & LAN** (thẻ LAN access) |
| Trang Cloudflare, đoạn mở đầu ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Settings → API → **Remote access** |
| Trang Cloudflare, mục Quickstart ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Settings → **Remote & LAN** → Remote access |
| Trang Cloudflare, mục "UI: start a link…" ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Settings → API → tìm thẻ **Remote access** |

Nếu không thấy menu theo một đường dẫn, hãy thử các đường dẫn còn lại.
:::

Đặt mật khẩu khi launch headless (không có terminal để nhập):

```bash
UNSLOTH_STUDIO_PASSWORD='your-strong-password' unsloth studio --secure
```

### Mỗi kiểu launch mở ra tới đâu

| Lệnh | Port thô | URL Cloudflare công khai |
| --- | --- | --- |
| `unsloth studio` | chỉ máy này | không |
| `unsloth studio -H 0.0.0.0` | cả mạng nội bộ | không |
| `unsloth studio -H 0.0.0.0 --cloudflare` | cả mạng nội bộ | **có** (kém riêng tư nhất) |
| `unsloth studio --secure` | chỉ máy này | **có**, là lối vào duy nhất |

::: warning Bảo mật khi mở Unsloth ra mạng
- **LAN là HTTP thường, không mã hóa.** Ai cùng phân đoạn mạng đều đọc được dữ liệu. Trên mạng không do bạn kiểm soát, hãy dùng `--secure` (HTTPS).
- **Tool phía server chạy dưới quyền user của bạn.** Ai có API key và truy cập được server là chạy được code trên máy bạn. Docs khuyên thêm `--disable-tools` khi mở ra mạng và giữ kín API key. Tool bật hay tắt mặc định: xem hộp "Docs chưa thống nhất" bên dưới.
- Ai có **URL + mật khẩu** là đăng nhập được (user `unsloth`). URL tunnel đổi mỗi lần start. Hãy coi URL là bí mật.
- `-H 0.0.0.0` vẫn để port thô mở. Chỉ `--secure` mới đóng port này.
- Lần đầu publish công khai, nếu admin vẫn dùng mật khẩu tự sinh, Unsloth bắt đổi mật khẩu trước. Tránh truyền `--password VALUE` trên dòng lệnh, vì mật khẩu sẽ lộ trong `ps` và shell history.
- Khi tunnel bật, các MCP server stdio cục bộ bị tắt (trừ khi bạn đặt `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1`).
:::

::: warning Docs chưa thống nhất
Các trang ghi khác nhau về việc tool phía server (web search, Python, terminal) **bật hay tắt mặc định** khi mở Unsloth ra mạng:

| Thông số | Trang LAN ([lan](https://unsloth.ai/docs/basics/lan)) | Trang Cloudflare ([remote access](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth)) | Trang API ([api](https://unsloth.ai/docs/basics/api)) |
| --- | --- | --- | --- |
| Lệnh được nói tới | Trang hướng dẫn `unsloth studio -H 0.0.0.0`; mục Security không ghi rõ lệnh | Trang hướng dẫn `unsloth studio --secure` / `--cloudflare`; mục Security không ghi rõ lệnh | `unsloth run` |
| Tool mặc định | "on by default" | "on by default" | Bind `127.0.0.1`: bật; bind `0.0.0.0` hoặc địa chỉ không phải loopback: **tắt** |
| Cách ép | `--disable-tools` | `--disable-tools` | `--enable-tools` / `--disable-tools` (trên `0.0.0.0`, `--enable-tools` hỏi y/N) |

Khi mở ra mạng, hãy luôn truyền `--disable-tools` một cách tường minh thay vì dựa vào mặc định.
:::

**Nguồn:** https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth

## Troubleshooting phổ biến

Bảng này gom các lỗi hay gặp khi export và deploy, kèm nguyên nhân và cách xử lý mà docs đưa ra.

| Triệu chứng | Nguyên nhân / cách xử lý theo docs |
| --- | --- |
| Chạy trong Unsloth tốt, sang Ollama, vLLM hay llama.cpp thì ra rác hoặc lặp vô hạn | Nguyên nhân phổ biến nhất là **chat template sai**: phải dùng đúng template lúc train. Kiểm tra `eos token`. Kiểm tra engine có thêm thừa hay thiếu token "start of sequence". Dùng conversational notebook của Unsloth để ép template |
| LM Studio ra rác hoặc lặp | Template không khớp. Vào **My Models** → bánh răng → **Prompt Template**, đặt đúng template đã train |
| Save GGUF hoặc vLLM 16-bit bị crash (OOM, hết bộ nhớ) | Giảm `maximum_memory_usage` (mặc định `0.75`), ví dụ xuống `0.5` |
| LM Studio không thấy model | Dùng `lms import`, hoặc đặt file đúng cấu trúc `~/.lmstudio/models/publisher/model/model-file.gguf` |
| LM Studio OOM hoặc chạy chậm | Dùng quant nhỏ hơn (`Q4_K_M`), giảm context, chỉnh GPU offload |
| `Unsloth Studio is already running on port 8888` | Chạy `unsloth studio stop` hoặc đổi `--port` |
| Thiết bị khác không vào được địa chỉ LAN | Hai thiết bị phải cùng mạng. Kiểm tra Guest Wi-Fi, AP isolation, VPN và firewall của máy |
| `--secure` thoát ngay | Tunnel bị lỗi. Sửa kết nối mạng trước (đừng vội dùng `--no-secure` vì nó mở port thô) |
| `cloudflared is unavailable` | Kiểm tra máy có truy cập ra `github.com` không, hoặc tự cài `cloudflared` vào `PATH` |

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio, https://unsloth.ai/docs/basics/lan, https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth
