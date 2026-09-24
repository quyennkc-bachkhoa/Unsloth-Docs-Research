---
title: Độ chính xác & lượng tử hóa
description: FP32, BF16, FP16, FP8, NVFP4/MXFP4, INT8/INT4; lượng tử hóa PTQ và QAT; các mức quant GGUF; Unsloth Dynamic; GPU hỗ trợ FP8/FP4; QLoRA và 4-bit; đánh đổi độ chính xác, bộ nhớ, tốc độ.
---

# Độ chính xác & lượng tử hóa

Trang này giải thích các kiểu số (BF16, FP8, NVFP4…) và các mức quant (Q4_K_M, UD-Q4_K_XL, bnb-4bit…) xuất hiện khắp docs Unsloth, để bạn chọn được file model và cờ `load_in_*` phù hợp với GPU. Đọc sau trang [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), trước khi tải model ở [Model catalog](/model-catalog) hoặc xuất model ở [Export & deploy](/export-deploy).

## Số dấu phẩy động: FP32, BF16, FP16

**Khái niệm.** Số dấu phẩy động (floating point) gồm 3 phần: 1 bit dấu (S, sign), các bit mũ (E, exponent) quyết định **khoảng giá trị** (range: số lớn nhất, nhỏ nhất biểu diễn được) và các bit định trị (M, mantissa) quyết định **độ mịn** (precision: phân biệt được hai số gần nhau đến đâu). Docs Unsloth minh họa: nhiều bit định trị thì biểu diễn được 0.121332, ít bit thì bị làm tròn thành 0.1.

| Kiểu | Bố cục S-E-M | Byte | Đặc điểm |
|---|---|---|---|
| FP32 | 1-8-23 | 4 | Chuẩn IEEE 754, "full precision" |
| FP16 (half) | 1-5-10 | 2 | Ít bit mũ, giá trị lớn nhất 65504, dễ tràn số |
| BF16 (bfloat16, "Brain floating point") | 1-8-7 | 2 | Cùng số bit mũ với FP32 nên cùng khoảng giá trị, nhưng kém mịn hơn FP16 |

**[Nguồn ngoài]** Bố cục S-E-M của FP16 và BF16 theo bảng dtype PyTorch https://docs.pytorch.org/docs/stable/tensor_attributes.html ; FP16 "1 sign bit, 5 exponent bits, and 10 fractional bits" theo NVIDIA https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html ; FP32 có 23 bit định trị, 8 bit mũ theo docs Unsloth (trang NVFP4); giá trị lớn nhất 65504 của FP16 theo https://docs.pytorch.org/docs/stable/amp.html .

**Vì sao BF16 phổ biến khi train.** Với FP16, gradient rất nhỏ bị làm tròn về 0 (underflow) nên phải dùng "gradient scaling": nhân loss với một hệ số trước lượt backward **[Nguồn ngoài]** https://docs.pytorch.org/docs/stable/amp.html . NVIDIA ví dụ mạng Multibox SSD có 31% giá trị gradient thành 0 khi chuyển sang FP16 **[Nguồn ngoài]** https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html . BF16 "has the same exponent range as fp32 so it almost never overflows". Hugging Face khuyên dùng BF16 trên GPU Ampere trở lên (A100, H100) và quay về FP16 trên phần cứng cũ như V100, T4 **[Nguồn ngoài]** https://huggingface.co/docs/transformers/mixed_precision_training . PyTorch cũng cảnh báo phần lớn model pretrain ở BF16 không chạy được trong khoảng giá trị FP16 (tối đa 65504) và sẽ tràn gradient **[Nguồn ngoài]** https://docs.pytorch.org/docs/stable/amp.html .

**Mixed precision** (độ chính xác hỗn hợp): tính forward/backward bằng FP16/BF16 nhưng giữ một bản trọng số FP32 để cập nhật **[Nguồn ngoài]** https://huggingface.co/docs/transformers/mixed_precision_training .

**Ảnh hưởng khi dùng Unsloth.** Tham số `dtype = None` là mặc định (tự chọn); docs ghi dùng `torch.float16` hoặc `torch.bfloat16` "for newer GPUs". **[Nhận định]** Trên GPU T4 (Colab miễn phí) nên để `None` hoặc FP16, vì theo Hugging Face T4 thuộc nhóm nên dùng FP16. Model "không hậu tố" trên Hugging Face của Unsloth là bản 16-bit gốc.

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning), [Cài đặt & phần cứng](/cai-dat).

**Nguồn:** https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide ; **[Nguồn ngoài]** các URL PyTorch, NVIDIA, Hugging Face ghi trong mục.

## FP8: E4M3 và E5M2

**Khái niệm.** FP8 là số dấu phẩy động 8 bit, có hai dạng:
- **E4M3** (4 bit mũ, 3 bit định trị): mịn hơn, khoảng giá trị ±448. Mở rộng khoảng giá trị bằng cách không biểu diễn vô cực và chỉ dành một mẫu bit cho NaN.
- **E5M2** (5 bit mũ, 2 bit định trị): khoảng giá trị rộng hơn, ±57344, theo quy ước IEEE 754 cho giá trị đặc biệt.

NVIDIA Transformer Engine giải thích: activation và trọng số ở lượt forward cần độ mịn nên hợp E4M3; gradient ở lượt backward cần khoảng giá trị rộng nên hợp E5M2. PyTorch có hai dtype tương ứng `torch.float8_e4m3fn` và `torch.float8_e5m2`.

Vì 8 bit biểu diễn được rất ít giá trị, FP8 luôn đi kèm **scale** (hệ số co giãn) ở độ chính xác cao hơn. Docs Unsloth viết: `original_weight ≈ quantized_weight * weight_scale`. Có thể dùng một scale cho cả tensor (per-tensor), một scale mỗi hàng/cột (per-channel) hoặc mỗi khối (block-wise, vd 128×128).

**Ví dụ.** Unsloth đo trên Qwen3-8B (throughput, MMLU Pro, GPQA Diamond): BF16 baseline 11,367 / 62.04% / 28.79%; block-wise 12,041 / 62.37% / 29.29%; per-channel 12,963 / 61.89% / 31.82%; per-tensor 13,681 / 61.83% / 27.78%. Docs kết luận block-wise hoặc per-channel (`-FP8-Dynamic`) tốt nhất.

**Ảnh hưởng khi dùng Unsloth.** Bật FP8 RL bằng `load_in_fp8 = True` trong `FastLanguageModel.from_pretrained`. Docs Unsloth: T4 trên Colab miễn phí "don't support FP8"; FP8 RL chạy trên H100, L4, RTX 40x, RTX 50x, H200, B200 và GPU NVIDIA ra sau RTX 4090. Unsloth lưu trọng số LoRA bị đóng băng ở FP8, adapter train được ở BF16, và tính gradient ở BF16.

**Gặp ở đâu trong Unsloth.** [Reinforcement Learning](/reinforcement-learning).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning ; **[Nguồn ngoài]** https://arxiv.org/abs/2209.05433 , https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html , https://docs.pytorch.org/docs/stable/tensor_attributes.html

## FP4: NVFP4 và MXFP4

**Khái niệm.** FP4 dùng dạng E2M1 (1 bit dấu, 2 bit mũ, 1 bit định trị), chỉ biểu diễn được vài giá trị trong khoảng xấp xỉ -6 đến 6 (vd 0, 0.5, 1, 1.5, 2, 3, 4, 6) **[Nguồn ngoài]** https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ . Để dùng được, FP4 dựa vào **block scaling**: chia tensor thành các khối nhỏ, mỗi khối có scale riêng. Paper Microscaling (MX) định nghĩa họ định dạng kết hợp "a per-block scaling factor with narrow floating-point and integer types" **[Nguồn ngoài]** https://arxiv.org/abs/2310.10537 .

| | MXFP4 | NVFP4 |
|---|---|---|
| Phần tử | E2M1 | E2M1 |
| Kích thước khối | 32 | 16 |
| Scale mỗi khối | E8M0 (chỉ lũy thừa của 2) | E4M3 (FP8) |
| Scale cấp 2 | không | một scale FP32 cho cả tensor |

**[Nguồn ngoài]** Bảng theo NVIDIA Technical Blog (URL trên) và Transformer Engine primer https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html . PyTorch có dtype `torch.float4_e2m1fn_x2` (hai giá trị 4-bit gói trong 1 byte) và `torch.float8_e8m0fnu` theo đặc tả OCP MX.

Docs Unsloth nêu hai lý do NVFP4 chính xác hơn MXFP4: khối 16 cô lập giá trị ngoại lai (outlier) tốt hơn khối 32, và scale E4M3 tốt hơn scale lũy thừa 2.

**Ví dụ.**
- NVIDIA: NVFP4 giảm bộ nhớ khoảng 3.5 lần so với FP16 và 1.8 lần so với FP8. **[Ước tính]** Tính cả scale E4M3 cho mỗi 16 phần tử: `4 + 8/16 = 4.5 bit ≈ 0.56 byte mỗi tham số`; `16 / 4.5 ≈ 3.6` và `8 / 4.5 ≈ 1.8`, khớp với con số của NVIDIA.
- Docs Unsloth (Qwen3.5 GGUF Benchmarks): MXFP4 dùng 4.25 bit/trọng số, Q4_K dùng 4.5; khi phải chọn thì nên dùng Q4_K. Unsloth đã bỏ MXFP4 khỏi các quant `Q2_K_XL`, `Q3_K_XL`, `Q4_K_XL` của Qwen3.5, chỉ giữ ở bản `MXFP4_MOE`.

**Ảnh hưởng khi dùng Unsloth.** Unsloth Dynamic NVFP4 giữ lớp quan trọng ở FP8 (W8A8) hoặc BF16, phần còn lại ở W4A4. Docs ghi NVFP4 cần GPU Blackwell (RTX 50X, DGX Spark, B200, B300); GPU cũ hơn thì dùng GGUF.

::: warning Docs chưa thống nhất
Về engine chạy NVFP4, trang Qwen3.8 ghi "You can run NVFP4 quants in vLLM only for now (SGLang is not supported)", nhưng cùng trang đó lại hướng dẫn `sglang serve unsloth/Qwen3.8-27B-NVFP4` và ghi "v0.5.19 works now!". Trang NVFP4 cũng có hướng dẫn SGLang. Xem https://unsloth.ai/docs/models/qwen3.8 và https://unsloth.ai/docs/basics/nvfp4 .
:::

**Gặp ở đâu trong Unsloth.** [Inference](/inference), [Export & deploy](/export-deploy).

**Nguồn:** https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks ; **[Nguồn ngoài]** https://arxiv.org/abs/2310.10537 , https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ , https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html , https://docs.pytorch.org/docs/stable/tensor_attributes.html

## Số nguyên: INT8 và INT4

**Khái niệm.** Thay vì số dấu phẩy động, lưu trọng số thành số nguyên nhỏ cộng một scale. Docs Unsloth mô tả lượng tử hóa INT8 ngây thơ:
1. Tìm `max(abs(W))`.
2. Tính `a = 127/max(abs(W))` (127 là giá trị lớn nhất của int8).
3. Lượng tử hóa `qW = int8(round(W * a))`; giải lượng tử bằng `float16(qW) / a`.

INT4 theo cùng ý tưởng nhưng chỉ có 16 mức giá trị.

**Ví dụ.** LLM.int8() (paper 2208.07339) dùng scale riêng cho từng vector và tách các chiều "outlier" (giá trị ngoại lai rất lớn) sang phép nhân 16-bit, trong khi hơn 99.9% giá trị vẫn nhân ở 8-bit; nhờ đó giảm một nửa bộ nhớ inference mà giữ chất lượng **[Nguồn ngoài]** https://arxiv.org/abs/2208.07339 . Hugging Face: lượng tử 8-bit bằng bitsandbytes "halves the memory-usage", 4-bit giảm 4 lần **[Nguồn ngoài]** https://huggingface.co/docs/transformers/quantization/bitsandbytes .

**Ảnh hưởng khi dùng Unsloth.** `load_in_8bit = True` bật fine-tune 8-bit. Với QAT, Unsloth hỗ trợ `qat_scheme` gồm `fp8-int4`, `fp8-fp8`, `int8-int4`, `int4`.

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning), [Export & deploy](/export-deploy).

**Nguồn:** https://unsloth.ai/docs/blog/quantization-aware-training-qat, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide ; **[Nguồn ngoài]** https://arxiv.org/abs/2208.07339 , https://huggingface.co/docs/transformers/quantization/bitsandbytes

## Bảng tổng hợp định dạng

| Định dạng | Số bit (S-E-M) | Byte/tham số | Dùng cho | Nguồn |
|---|---|---|---|---|
| FP32 | 32 (1-8-23) | 4 | Bản trọng số chính và optimizer state khi train mixed precision | PyTorch; Unsloth NVFP4; HF model_memory_anatomy |
| BF16 | 16 (1-8-7) | 2 | Train và chạy trên GPU Ampere trở lên | PyTorch; HF mixed_precision_training |
| FP16 | 16 (1-5-10) | 2 | GPU cũ (V100, T4), cần gradient scaling khi train | PyTorch; NVIDIA mixed precision; PyTorch AMP |
| FP8 E4M3 | 8 (1-4-3) | 1 + scale | Trọng số, activation (forward) | arXiv 2209.05433; NVIDIA TE |
| FP8 E5M2 | 8 (1-5-2) | 1 + scale | Gradient (backward) | arXiv 2209.05433; NVIDIA TE |
| NVFP4 | 4 (1-2-1), scale E4M3/16 phần tử + FP32/tensor | ≈ 0.56 **[Ước tính]** | Inference trên Blackwell | NVIDIA blog; NVIDIA TE |
| MXFP4 | 4 (1-2-1), scale E8M0/32 phần tử | ≈ 0.53 (4.25 bit theo Unsloth) | GGUF `MXFP4_MOE` | arXiv 2310.10537; Unsloth Qwen3.5 GGUF Benchmarks |
| INT8 | 8 | 1 + scale | LLM.int8(), W8A8 | arXiv 2208.07339; HF bitsandbytes |
| INT4 / NF4 | 4 | 0.5 + scale | QLoRA, `Int4WeightOnlyConfig` | arXiv 2305.14314; Unsloth QAT |

**[Ước tính]** Cột byte của MXFP4: `4.25 ÷ 8 ≈ 0.53`.

**Nguồn:** như cột Nguồn; URL đầy đủ ở các mục trên.

## Quantization là gì

**Khái niệm.** Quantization (lượng tử hóa) là lưu trọng số ở độ chính xác thấp hơn để giảm bộ nhớ, cố giữ độ chính xác càng nhiều càng tốt. Một số phương pháp cần dữ liệu calibration (hiệu chỉnh) để đạt độ chính xác cao ở mức 1-2 bit, số khác lượng tử hóa ngay khi nạp model (on-the-fly) **[Nguồn ngoài]** https://huggingface.co/docs/transformers/quantization/overview . Mức mất chất lượng thường đo bằng perplexity (PPL) hoặc KL divergence (KLD) so với model gốc **[Nguồn ngoài]** https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md .

**Nguồn:** **[Nguồn ngoài]** hai URL trong đoạn.

### PTQ và QAT

**Khái niệm.**
- **PTQ** (post-training quantization, lượng tử hóa sau huấn luyện): lấy model đã train rồi nén, như công thức INT8 ở trên. Rẻ và nhanh, nhưng docs Unsloth ghi thường làm giảm độ chính xác, "especially at 4-bit or lower".
- **QAT** (quantization-aware training, huấn luyện có tính đến lượng tử hóa): trong lúc train, "fake quantize" trọng số (làm tròn về giá trị lượng tử rồi giải lượng tử ngay, vẫn ở BF16) để model học cách chịu sai số. Sau train mới chuyển sang lượng tử thật.

**Ví dụ.** Docs Unsloth: QAT phục hồi tới 70% độ chính xác bị mất và cải thiện 1–3% trên GPQA, MMLU Pro; Gemma3-4B phục hồi 66.9% trên GPQA (+1.0% tuyệt đối); Gemma3-12B phục hồi 45.5% trên BBH (+2.1%). QAT không tốn thêm chi phí khi inference. Ngược lại, Unsloth Dynamic 3.0 GGUF là "pure PTQ", không dùng QAT hay QAD (quantization-aware distillation).

**Ảnh hưởng khi dùng Unsloth.** Bật QAT khi fine-tune LoRA bằng `qat_scheme` (code nguyên văn từ docs):

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    
    # We support fp8-int4, fp8-fp8, int8-int4, int4
    qat_scheme = "int4",
)
```

Sau train, xuất bằng `model.save_pretrained_torchao`. Docs Unsloth cũng khuyên "training and serving in the same precision": định chạy 4-bit thì train 4-bit.

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning), [Export & deploy](/export-deploy).

**Nguồn:** https://unsloth.ai/docs/blog/quantization-aware-training-qat, https://unsloth.ai/docs/basics/dynamic-3.0-ggufs, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

### Weight-only và weight + activation

**Khái niệm.** Ký hiệu `WxAy`: trọng số (W, weight) ở x bit, activation (A) ở y bit.
- **Weight-only** (vd W4A16): chỉ nén trọng số; lúc tính, trọng số được giải lượng tử về 16-bit. Tiết kiệm bộ nhớ nhưng phép nhân vẫn ở 16-bit.
- **Weight + activation** (vd W8A8, W4A4): cả hai vào phép nhân ở độ chính xác thấp, tận dụng được tensor core FP8/FP4.

**Ví dụ.** Docs Unsloth: NVFP4 của NVIDIA dùng W4A16, còn NVFP4 của Unsloth dùng W4A4 "so they actually use the FP4 tensor cores". Bảng đo Qwen3.6-27B trên backend `cute-DSL`: Unsloth W4A4 đạt 6,863 tok/s throughput so với 2,403 tok/s của bản NVIDIA W4A16 qua Marlin. Trong TorchAO, `Int4WeightOnlyConfig` là weight-only, `Int8DynamicActivationInt8WeightConfig` là W8A8. **[Nhận định]** QLoRA thuộc loại weight-only: trọng số lưu NF4 nhưng tính toán ở BF16 **[Nguồn ngoài]** https://arxiv.org/abs/2305.14314 .

**Ảnh hưởng khi dùng Unsloth.** Docs NVFP4: không tự chọn backend MoE trong vLLM (Marlin chậm 2.5 lần với W4A4), để vLLM tự chọn.

**Gặp ở đâu trong Unsloth.** [Inference](/inference), [Export & deploy](/export-deploy).

**Nguồn:** https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/blog/quantization-aware-training-qat ; **[Nguồn ngoài]** https://arxiv.org/abs/2305.14314

## Các mức quant GGUF

**Khái niệm.** GGUF là định dạng file một-tệp của ggml/llama.cpp, chứa cả tensor và metadata **[Nguồn ngoài]** https://github.com/ggml-org/ggml/blob/master/docs/gguf.md . Tên mức quant cho biết số bit và cách chia khối:
- **Legacy** `Q8_0`, `Q4_0`…: khối 32 trọng số, `w = q * block_scale`.
- **K-quant** `Q2_K` … `Q6_K`: super-block gồm nhiều khối nhỏ, scale của khối cũng được lượng tử hóa. Theo bảng của Hugging Face Hub: `Q6_K` 6.5625, `Q5_K` 5.5, `Q4_K` 4.5, `Q3_K` 3.4375, `Q2_K` 2.625 bit/trọng số.
- **I-quant** `IQ4_XS`, `IQ3_XXS`, `IQ2_XXS`, `IQ1_S`…: super-block 256 trọng số, dùng importance matrix; vd `IQ2_XXS` 2.06, `IQ1_S` 1.56 bit/trọng số.
- Hậu tố `_S`, `_M`, `_L`: phối nhiều kiểu cho các tensor khác nhau. Danh sách tùy chọn của Unsloth mô tả `q4_k_m` "Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K", `q3_k_s` "Uses Q3_K for all tensors".

**[Nguồn ngoài]** Số bit/trọng số của từng kiểu theo https://huggingface.co/docs/hub/gguf .

**imatrix** (importance matrix, ma trận độ quan trọng): dữ liệu calibration cho llama.cpp biết trọng số nào quan trọng để lượng tử hóa cẩn thận hơn; bật bằng `--imatrix file_name` khi chạy `llama-quantize` **[Nguồn ngoài]** https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md .

**Ví dụ.** Bảng trong README `llama-quantize` cho Llama-3.1-8B (cả file, vì các tensor được phối nhiều kiểu nên bit/trọng số cao hơn kiểu gốc) **[Nguồn ngoài]** https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md :

| Kiểu | Bit/trọng số | Kích thước (GiB) | Sinh chữ t/s @ 128 |
|---|---|---|---|
| F16 | 16.0005 | 14.96 | 29.17 |
| Q8_0 | 8.5008 | 7.95 | 50.93 |
| Q6_K | 6.5633 | 6.14 | 58.67 |
| Q5_K_M | 5.7036 | 5.33 | 67.23 |
| Q4_K_M | 4.8944 | 4.58 | 71.93 |
| IQ4_XS | 4.4597 | 4.17 | 77.51 |
| Q3_K_M | 3.9960 | 3.74 | 71.68 |
| IQ3_XXS | 3.2548 | 3.04 | 73.95 |
| Q2_K | 3.1593 | 2.95 | 79.85 |
| IQ2_XXS | 2.3824 | 2.23 | 79.86 |
| IQ1_S | 2.0042 | 1.87 | 79.73 |

README không ghi phần cứng đo tốc độ.

**Ảnh hưởng khi dùng Unsloth.**
- Xuất GGUF từ model đã fine-tune (code nguyên văn từ docs):

```python
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "f16")
```

  Docs đánh dấu `q4_k_m`, `q5_k_m` là "Recommended"; `q8_0` "High resource use, but generally acceptable"; `f16` "retains 100% accuracy" nhưng chậm và tốn bộ nhớ.
- Docs Unsloth (Qwen3.5 GGUF Benchmarks): imatrix giảm KLD và PPL rõ rệt, nhất là ở bit thấp; I-quant như `iq3_xxs`, `iq2_s` hiệu quả hơn về dung lượng nhưng inference chậm hơn 5-10%.
- Unsloth khuyên dùng tối thiểu `UD-Q2_K_XL` (2-bit dynamic) để cân bằng kích thước và độ chính xác.

**Gặp ở đâu trong Unsloth.** [Export & deploy](/export-deploy), [Inference](/inference), [Model catalog](/model-catalog).

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks, https://unsloth.ai/docs/models/qwen3.5 ; **[Nguồn ngoài]** https://huggingface.co/docs/hub/gguf , https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md , https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Unsloth Dynamic quants

**Khái niệm.** Thay vì dùng cùng một mức quant cho mọi layer, Unsloth Dynamic chọn kiểu quant riêng cho từng layer của từng model, dựa trên dữ liệu calibration riêng. Các file có tiền tố `UD-` (xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) về cách đọc tên).
- **Dynamic 2.0**: điều chỉnh kiểu quant của "every possible layer", mỗi model một sơ đồ riêng; dữ liệu calibration hơn 1.5M token; áp dụng cho cả MoE và dense. Ở Qwen3.5, "4-bit has important layers upcasted to 8 or 16-bit".
- **Dynamic 3.0**: dữ liệu imatrix mới (agentic coding, chat, đa ngôn ngữ), chọn layer tốt hơn, thêm kỹ thuật quant; vẫn là PTQ thuần. Unsloth vẫn dùng UD-2 cho các quant lớn vì UD-3 cải thiện ít ở đó.

Unsloth dùng KL divergence làm thước đo chính vì theo paper "Accuracy is Not All You Need", perplexity có thể che sai số (các token lệch triệt tiêu nhau), còn KLD tương quan cao với số câu trả lời bị "lật" đúng/sai.

**Ví dụ (kèm điều kiện đo).**
- Bảng KLD trong mục Gemma 3 của trang Dynamic (trang không ghi rõ kích thước model và baseline; dung lượng khớp bản 27B). Docs ghi khi đo KLD, Unsloth dùng bộ Wikipedia chuẩn thay vì dữ liệu calibration riêng: `Q4_K_XL` baseline 0.024916 (15.41 GB) → mới 0.023701 (15.64 GB); `Q2_K_XL` 0.229671 (9.78 GB) → 0.220937 (9.95 GB).
- Gemma 3 12B: bản QAT `Q4_0` của Google đạt 67.07% MMLU 5-shot, BF16 đạt 67.15%. Với 27B, Unsloth dynamic `Q4_K_XL` đạt 71.47% (15.64 GB), bản Google QAT 70.64% (17.2 GB).
- Qwen3.8-27B, Dynamic 3.0: Unsloth tuyên bố top-1 accuracy cao hơn >10% so với các nhà cung cấp khác ở cùng dung lượng. `UD-IQ1_S` 6.2 GB giữ khoảng 72% top-1 và nhỏ hơn 89%. Theo chỉ số Divergence-300 @32 (300 prompt không có trong calibration, sinh greedy 32 token rồi so với BF16), độ khớp rơi từ khoảng 25% ở `UD-Q2_K_XL` xuống dưới 8-10% ở `UD-IQ2_S`.
- Qwen3.5 GGUF Benchmarks (trang không ghi rõ model của bảng đầy đủ; cần kiểm tra lại): Unsloth `Q4_K_XL` 19.17 GB, mean KLD 0.0137; `Q4_K_M` 18.49 GB, 0.0192; `Q8_K_XL` 36.04 GB, 0.0026; `IQ2_XXS` 9.09 GB, 0.1846. Các tensor `attn_*` và `ssm_out` rất nhạy khi lượng tử hóa; `ffn_up_exps`, `ffn_gate_exps` chịu được 3-bit.

**Ảnh hưởng khi dùng Unsloth.**
- Không dùng quant 1-bit cho tác vụ agentic/tool calling; dưới `UD-Q2_K_XL` thì đặt `presence_penalty = 1.5` trở lên để giảm lặp, và luôn bật thinking với quant 1-bit.
- Docs cảnh báo PPL và KLD có thể gây hiểu nhầm vì phụ thuộc dữ liệu calibration; imatrix của Unsloth dùng dữ liệu chat và tool-calling ngữ cảnh dài nên đôi khi PPL cao hơn quant khác.

**Gặp ở đâu trong Unsloth.** [Inference](/inference), [Model catalog](/model-catalog).

**Nguồn:** https://unsloth.ai/docs/basics/dynamic-3.0-ggufs, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/models/qwen3.5, https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks

## GPU nào hỗ trợ FP8 và FP4 ở phần cứng

**Khái niệm.** "Hỗ trợ phần cứng" nghĩa là GPU có tensor core nhân trực tiếp FP8/FP4, nhờ vậy nhanh hơn. **[Nhận định]** GPU không có tính năng này vẫn có thể chạy model đã nén (vd GGUF 4-bit) bằng cách giải lượng tử bằng phần mềm; docs Unsloth ghi "For older GPUs, our GGUFs work well".

Theo NVIDIA:
- "H100 GPU introduced support for a new datatype, FP8" và "Blackwell added support for NVFP4 and MXFP8 datatypes" **[Nguồn ngoài]** https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html .
- NVFP4 được hỗ trợ trên "NVIDIA Blackwell and Blackwell Ultra GPUs" **[Nguồn ngoài]** https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ .

Compute capability (CC, mã tính năng phần cứng) theo bảng của NVIDIA **[Nguồn ngoài]** https://developer.nvidia.com/cuda-gpus :

| Kiến trúc | CC | Ví dụ GPU trong bảng NVIDIA |
|---|---|---|
| Blackwell | 12.0 / 12.1 | GeForce RTX 5050–5090, RTX PRO 6000 Blackwell Server Edition (12.0); GB10 / DGX Spark (12.1) |
| Blackwell (data center) | 10.0 / 10.3 | B200, GB200 (10.0); B300, GB300 (10.3) |
| Hopper | 9.0 | H100, H200 |
| Ada Lovelace | 8.9 | L4, L40, L40S, GeForce RTX 4050–4090 |
| Ampere | 8.0 / 8.6 | A100, A30 (8.0); A40, A10 (8.6) |
| Turing | 7.5 | T4, GeForce RTX 2060–2080 Ti |

Tài liệu NVIDIA đã đọc không nêu Ada Lovelace hỗ trợ FP8 (cần kiểm tra lại với tài liệu NVIDIA khác). Docs Unsloth thì ghi FP8 RL chạy trên RTX 40x và L4 (đều là Ada), còn T4 không hỗ trợ FP8.

**Ảnh hưởng khi dùng Unsloth.** Unsloth Core yêu cầu tối thiểu CUDA Capability 7.0. Kịch bản DGX Spark trong docs NVFP4 kiểm tra `cap[0] == 12` trước khi chạy, nếu không vLLM sẽ rơi về Marlin W4A16 chậm hơn. Chọn NVFP4 chỉ khi có GPU Blackwell; FP8 RL cần GPU từ thế hệ RTX 40/L4/H100 trở lên theo docs Unsloth.

**Gặp ở đâu trong Unsloth.** [Cài đặt & phần cứng](/cai-dat), [Reinforcement Learning](/reinforcement-learning), [Inference](/inference).

**Nguồn:** https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements ; **[Nguồn ngoài]** các URL NVIDIA ghi trong mục.

## QLoRA và 4-bit

**Khái niệm.** QLoRA truyền gradient qua một model gốc đã đóng băng ở 4-bit vào các adapter LoRA (xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora)). Paper QLoRA đưa ra ba kỹ thuật **[Nguồn ngoài]** https://arxiv.org/abs/2305.14314 :
- **NF4** (4-bit NormalFloat): kiểu 4-bit "information theoretically optimal for normally distributed weights", tức hợp với trọng số phân bố chuẩn. Trọng số lưu ở NF4, tính toán ở BF16.
- **Double quantization**: lượng tử hóa luôn các hằng số scale, giảm phần phụ từ 0.5 xuống 0.127 bit mỗi tham số (khoảng 3 GB với model 65B). Hugging Face ghi tiết kiệm thêm khoảng 0.4 bit/tham số **[Nguồn ngoài]** https://huggingface.co/docs/transformers/quantization/bitsandbytes .
- **Paged optimizers**: dùng NVIDIA unified memory tự chuyển trang giữa CPU và GPU để tránh đỉnh bộ nhớ.

Kết quả: fine-tune model 65B trên một GPU 48GB.

**Ví dụ.** Trong bitsandbytes/Transformers, các tùy chọn tương ứng là `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.bfloat16`. Theo Hugging Face, NF4/FP4 của bitsandbytes chạy trên GPU NVIDIA Pascal trở lên, LLM.int8() cần Turing trở lên **[Nguồn ngoài]** https://huggingface.co/docs/transformers/quantization/bitsandbytes .

**Ảnh hưởng khi dùng Unsloth.**
- `load_in_4bit = True` bật QLoRA. Docs Unsloth khuyên bắt đầu bằng QLoRA.
- Model `-unsloth-bnb-4bit` là Unsloth dynamic 4-bit: "consume slightly more VRAM than standard BitsAndBytes 4-bit models but offer significantly higher accuracy". Model `-bnb-4bit` là BitsAndBytes 4-bit tiêu chuẩn.

::: warning Docs chưa thống nhất
Các trang docs Unsloth mô tả mức mất độ chính xác của QLoRA 4-bit so với LoRA 16-bit khác nhau:
- "there is a 1-2% accuracy degradation" — https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
- "the accuracy loss compared to standard 16-bit LoRA fine-tuning is now negligible" — https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me
- "the accuracy loss for QLoRA compared to LoRA is now largely recovered" — https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- "Slightly slower and marginally less accurate" — https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
:::

**Gặp ở đâu trong Unsloth.** [Fine-tuning](/fine-tuning).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide ; **[Nguồn ngoài]** https://arxiv.org/abs/2305.14314 , https://huggingface.co/docs/transformers/quantization/bitsandbytes

## Đánh đổi độ chính xác – bộ nhớ – tốc độ

**Khái niệm.** Giảm bit thì giảm bộ nhớ, thường tăng tốc (ít dữ liệu phải di chuyển hơn) nhưng mất độ chính xác. Hugging Face: "Shrinking your memory footprint makes a model 'faster' because there is less data to move around" **[Nguồn ngoài]** https://huggingface.co/docs/transformers/perf_train_gpu_one . Mức tăng tốc lớn nhất khi GPU có tensor core cho đúng kiểu số đó.

| Lựa chọn | Bộ nhớ trọng số | Độ chính xác (theo tài liệu, kèm điều kiện) | Tốc độ (theo tài liệu, kèm điều kiện) |
|---|---|---|---|
| BF16 | 2 byte/tham số | Mốc so sánh | Mốc so sánh |
| FP8 (Unsloth FP8 RL) | 1 byte + scale | Đường loss SFT của FP8 và BF16 bám sát nhau (Qwen3, Llama 3.2…) | RL inference qua vLLM nhanh khoảng 1.4 lần BF16; FP8 so với BF16 cho throughput 1.6 lần trên H100 |
| Unsloth Dynamic NVFP4 | ≈ 0.56 byte **[Ước tính]** | 92–97% top-1 so với BF16 (trang Qwen3.8, bảng không ghi rõ model); Qwen3.6-27B MMLU-Pro 86.25 so với BF16 85.96 | Qwen3.8-27B nhanh khoảng 1.41–1.49 lần BF16 (batch 1–64; trang không ghi GPU của bảng này) |
| GGUF Q8_0 | 8.5 bit | Gemma 3 27B MMLU 5-shot 71.60% | Llama-3.1-8B: 50.93 t/s so với F16 29.17 t/s (README llama.cpp) |
| GGUF Q4_K_M / UD-Q4_K_XL | 4.9 bit (Q4_K_M) | Gemma 3 27B: `Q4_K_M` 71.23%, `Q4_K_XL` 71.47%; KLD `Q4_K_XL` 0.0137 (Qwen3.5 benchmark) | Llama-3.1-8B `Q4_K_M` 71.93 t/s |
| GGUF 2-bit (`UD-Q2_K_XL`) | ≈ 2–3 bit | Mức tối thiểu Unsloth khuyên dùng | I-quant chậm hơn 5-10% so với K-quant |
| GGUF 1-bit (`UD-IQ1_S`) | ≈ 1.5–2 bit | Khoảng 72% top-1 (Qwen3.8-27B); không dùng cho agentic | — |
| QLoRA 4-bit (train) | 0.5 byte + scale cho model gốc | Xem hộp "Docs chưa thống nhất" ở trên | "Slightly slower" so với LoRA 16-bit, dùng khoảng 4× ít VRAM hơn |

**[Nhận định]** Cách chọn nhanh: có GPU Blackwell và phục vụ nhiều người thì xem NVFP4; chạy local một người trên GPU thường, CPU hoặc Mac thì dùng GGUF `UD-Q4_K_XL` rồi hạ dần nếu không vừa bộ nhớ; fine-tune thì bắt đầu QLoRA theo khuyến nghị Unsloth, đủ VRAM thì dùng LoRA 16-bit.

**Gặp ở đâu trong Unsloth.** [Inference](/inference), [Fine-tuning](/fine-tuning), [Export & deploy](/export-deploy), [Reinforcement Learning](/reinforcement-learning).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning, https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/models/qwen3.8, https://unsloth.ai/docs/basics/dynamic-3.0-ggufs, https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide ; **[Nguồn ngoài]** https://huggingface.co/docs/transformers/perf_train_gpu_one , https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

## Gặp ở đâu trong Unsloth

| Khái niệm | Trang Unsloth trên website | Docs gốc |
|---|---|---|
| BF16/FP16, `dtype` | [/fine-tuning](/fine-tuning) | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide |
| FP8, `load_in_fp8` | [/reinforcement-learning](/reinforcement-learning) | https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning |
| NVFP4, MXFP4, W4A4/W4A16 | [/inference](/inference), [/export-deploy](/export-deploy) | https://unsloth.ai/docs/basics/nvfp4 |
| PTQ, QAT, `qat_scheme` | [/fine-tuning](/fine-tuning), [/export-deploy](/export-deploy) | https://unsloth.ai/docs/blog/quantization-aware-training-qat |
| Mức quant GGUF, `quantization_method`, imatrix | [/export-deploy](/export-deploy), [/inference](/inference) | https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf |
| Unsloth Dynamic 2.0/3.0, KLD | [/inference](/inference), [/model-catalog](/model-catalog) | https://unsloth.ai/docs/basics/dynamic-3.0-ggufs |
| GPU hỗ trợ FP8/FP4 | [/cai-dat](/cai-dat) | https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements |
| QLoRA, NF4, `unsloth-bnb-4bit` | [/fine-tuning](/fine-tuning) | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide |
