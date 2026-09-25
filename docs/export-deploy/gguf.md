---
title: Xuất file GGUF
description: "Lưu model sang GGUF bằng Core API, các quantization_method được hỗ trợ và cách convert thủ công bằng llama.cpp."
---

# Xuất file GGUF

GGUF là định dạng file của llama.cpp mà Ollama, LM Studio và Unsloth Studio đều đọc được. Bạn có hai cách tạo file GGUF: gọi hàm có sẵn của Unsloth (Core API), hoặc tự convert thủ công bằng llama.cpp.

::: tip Tóm tắt
- **Dùng khi:** bạn muốn chạy model đã fine-tune bằng llama.cpp, Ollama, LM Studio hoặc Unsloth Studio.
- **Kết quả:** tạo được file GGUF bằng Core API hoặc thủ công, chọn được `quantization_method` phù hợp, và phân biệt được file GGUF tự xuất với các file `UD-` do Unsloth làm sẵn.
- **Nên biết trước:** [Export và deploy](/export-deploy/) (bảng so sánh các định dạng export).
:::

## Lưu GGUF bằng Core API

Chỉ cần một dòng code để lưu GGUF, cục bộ hoặc đẩy thẳng lên Hugging Face Hub. Lưu cục bộ:

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

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf

## Các `quantization_method` được hỗ trợ

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

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama

## Lưu GGUF thủ công {#luu-gguf-thu-cong}

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

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf

## File GGUF Unsloth làm sẵn (UD) {#dynamic-gguf}

Ngoài file bạn tự xuất, còn có các file GGUF **do Unsloth tự lượng tử hóa và upload** lên Hugging Face. Bạn nhận ra chúng qua tiền tố `UD-`, ví dụ `UD-Q4_K_XL`, `UD-Q2_K_XL`. Đây không phải một giá trị `quantization_method` mà bạn chọn khi tự save model.

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

::: warning Lỗi thường gặp
Không dùng 1-bit cho agentic: docs khuyến cáo không dùng quant thấp hơn `UD-Q2_K_XL` (tức 1-bit) cho tool calling hoặc agent. Các lỗi hay gặp:
- Dễ lặp vô hạn. Khi đó dùng `presence_penalty = 1.5` trở lên.
- Trả lời rỗng nếu tắt thinking.
- Gọi tool sai.

Chỉ dùng các quant này cho câu hỏi kiến thức ngắn. Lựa chọn tốt nhất vẫn là `UD-Q2_K_XL`.
:::

**Nguồn:** https://unsloth.ai/docs/basics/dynamic-3.0-ggufs

## Đọc tiếp

- [Định dạng trọng số NVFP4 và FP8](/export-deploy/nvfp4-fp8) — trang kế tiếp: hai định dạng dành cho GPU mới, chạy bằng vLLM hoặc SGLang.
- [Chạy model đã xuất](/export-deploy/chay-model#ollama) — nạp file GGUF vừa tạo vào Ollama, llama-server hoặc LM Studio.
- [Lỗi thường gặp](/export-deploy/loi-thuong-gap) — khi file GGUF chạy ra kết quả lạ hoặc không chạy.
