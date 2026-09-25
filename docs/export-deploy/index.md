---
title: Export và deploy
description: "Unsloth hỗ trợ xuất model sau fine-tune sang những định dạng nào (LoRA, merged 16-bit, GGUF, NVFP4…) và cách đưa lên Ollama, llama-server, vLLM, LM Studio, LAN, Cloudflare."
---

# Export và deploy

Sau khi fine-tune (tinh chỉnh) xong, bạn có một checkpoint (bản lưu model trong lúc train) gồm model gốc cộng với LoRA adapter (phần trọng số nhỏ học thêm). Việc tiếp theo là xuất (export) model ra một định dạng, rồi đưa nó lên một engine (phần mềm chạy model) để dùng.

::: tip Tóm tắt
- **Dùng khi:** bạn vừa fine-tune xong và muốn đem model ra chạy ở nơi khác (Ollama, llama-server, vLLM, LM Studio...).
- **Kết quả:** trả lời được ba câu hỏi: **Unsloth xuất được ra định dạng gì, bằng lệnh nào, và định dạng đó chạy trên engine nào**.
- **Nên biết trước:** [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) (FP16/BF16, FP8, FP4/NVFP4, GGUF quant như Q4_K_M, Q8_0, Dynamic quants là gì — phần này không giải thích lại), [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) (LoRA adapter và merge), [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) (chat template).
:::

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [Xuất file GGUF](/export-deploy/gguf) | Lưu GGUF bằng Core API, chọn `quantization_method`, convert thủ công bằng llama.cpp, và hiểu các file GGUF `UD-` do Unsloth làm sẵn | Khi muốn chạy model bằng llama.cpp, Ollama hoặc LM Studio |
| [Định dạng trọng số NVFP4 và FP8](/export-deploy/nvfp4-fp8) | Hiểu NVFP4 (4-bit cho GPU Blackwell, chạy bằng vLLM hoặc SGLang) và vì sao với Unsloth, FP8 là chế độ lúc train chứ không phải một bước export | Khi có GPU Blackwell hoặc gặp các bản quant NVFP4/FP8 trên Hugging Face |
| [Chạy model đã xuất](/export-deploy/chay-model) | Đưa model lên Ollama (Unsloth tự tạo `Modelfile`), llama-server (endpoint tương thích OpenAI), vLLM (model merged 16-bit) hoặc LM Studio | Khi đã có file export và cần chạy nó |
| [Mở cho máy khác (LAN, Cloudflare)](/export-deploy/lan-remote) | Mở server cho máy khác trong mạng nội bộ, hoặc ra ngoài qua Cloudflare tunnel | Khi muốn máy hoặc người khác gọi được model của bạn |
| [Lỗi thường gặp](/export-deploy/loi-thuong-gap) | Tra các lỗi hay gặp sau khi export | Khi model xuất ra chạy sai hoặc không chạy |

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

**[Nhận định]** Sơ đồ được gom từ nhiều trang docs. Nhánh NVFP4 được vẽ tách riêng vì một lý do: nguồn chỉ hướng dẫn **chạy** các quant NVFP4 mà Unsloth đã upload sẵn. Nguồn không có hướng dẫn tự xuất NVFP4 từ model bạn fine-tune.

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/nvfp4

## Bảng định dạng export

Bảng này đặt các định dạng export cạnh nhau. Với mỗi định dạng, bạn thấy engine nào chạy được nó, cách xuất trong Studio và trong Core API (thư viện Python của Unsloth), cùng lưu ý chính.

| Định dạng | Dùng cho engine nào | Cách xuất | Ghi chú |
| --- | --- | --- | --- |
| LoRA adapter | vLLM (docs có trang LoRA Hot Swapping), Unsloth | Studio: **LoRA Only**. Core: `model.save_pretrained(...)` + `tokenizer.save_pretrained(...)`, hoặc `save_pretrained_merged(..., save_method = "lora")`, `push_to_hub_merged(..., save_method = "lora")` | Chỉ chứa trọng số adapter (~100MB theo trang Ollama). Khi chạy vẫn cần model gốc |
| Merged 16-bit (safetensors) | vLLM, SGLang, transformers. Cũng là đầu vào khi bạn tự convert sang GGUF | Studio: **Merged Model**. Core: `save_pretrained_merged(..., save_method = "merged_16bit")`, `push_to_hub_merged(..., save_method = "merged_16bit")` | LoRA đã được gộp vào trọng số gốc |
| Merged 4-bit | Hugging Face (inference online), DPO training | Core: `save_method = "merged_4bit"`, sau đó `merged_4bit_forced` nếu chắc chắn | Docs **không khuyến khích**, trừ khi bạn biết rõ mục đích |
| GGUF | llama.cpp, llama-server, Ollama, LM Studio, Unsloth Studio | Studio: **GGUF / llama.cpp**. Core: `save_pretrained_gguf(...)`, `push_to_hub_gguf(...)`. Hoặc làm thủ công qua `convert_hf_to_gguf.py` | Cần chọn `quantization_method` (xem [GGUF](/export-deploy/gguf)) |
| NVFP4 (Unsloth Dynamic NVFP4) | vLLM, SGLang trên GPU Blackwell | Không thấy hàm export NVFP4 trong nguồn — cần kiểm tra lại | Nguồn chỉ có các quant NVFP4 do Unsloth upload sẵn |
| FP8 | vLLM, SGLang (theo trang FP8 RL) | Không có trang export FP8 riêng — xem [FP8](/export-deploy/nvfp4-fp8#fp8) | Unsloth có upload sẵn bản FP8 Dynamic và bản FP8 Block |

::: info safetensors vs .bin trong Colab
Trang Troubleshooting ghi: trong Colab, Unsloth lưu file `.bin` vì cách này nhanh hơn ~4 lần. Muốn ép lưu `.safetensors`, bạn đặt `safe_serialization = None`. Ví dụ: `model.save_pretrained(..., safe_serialization = None)` hoặc `model.push_to_hub(..., safe_serialization = None)`.
:::

**Nguồn:** https://unsloth.ai/docs/new/studio/export, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama, https://unsloth.ai/docs/basics/nvfp4, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning, https://unsloth.ai/docs/basics/inference-and-deployment/troubleshooting-inference

## Xuất bằng Unsloth Studio {#xuat-bang-studio}

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

## Đọc tiếp

- [Xuất file GGUF](/export-deploy/gguf) — trang kế tiếp: định dạng export phổ biến nhất để chạy local.
- [Chạy model đã xuất](/export-deploy/chay-model) — đưa file vừa xuất lên Ollama, llama-server, vLLM hoặc LM Studio.
- [Chạy model và gọi API](/inference/) — các cách chạy model và gọi nó qua API.
