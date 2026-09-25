---
title: Host model trên máy chủ
description: "Chạy model đã fine-tune trên máy chủ 2×L40S cho cả đơn vị dùng: so sánh vLLM, llama.cpp, Ollama, cách chia 2 GPU và ước lượng số người dùng."
---

# Host model trên máy chủ

Trang này trả lời câu hỏi "có mấy cách host model local" cho máy 2×L40S. Có ba engine chính: vLLM, llama.cpp (llama-server) và Ollama. Đọc xong bạn biết engine nào hợp giai đoạn nào, cách bắt nó dùng cả hai card, và mỗi cỡ model phục vụ được khoảng bao nhiêu người.

::: tip Tóm tắt
- **Dùng khi:** đã có model (tải sẵn hoặc vừa fine-tune) và cần mở cho nhiều người trong đơn vị dùng qua API.
- **Kết quả:** chọn được engine, có lệnh chạy trên 2 GPU, biết model cỡ nào vừa 96 GB kèm bộ nhớ cho người dùng đồng thời.
- **Nên biết trước:** [Chạy model đã xuất](/export-deploy/chay-model) (lệnh cơ bản của từng engine theo docs Unsloth), [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) (KV cache), [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) (FP8, GGUF).
:::

## Ba engine, chọn cái nào

Quy tắc ngắn: **vLLM cho production nhiều người dùng, Ollama hoặc llama-server để thử nghiệm**. Cả ba đều có API tương thích OpenAI, nên ứng dụng viết một lần là gọi được engine nào cũng được.

| | vLLM | llama.cpp (llama-server) | Ollama |
| --- | --- | --- | --- |
| Định dạng model | safetensors BF16, FP8, AWQ, GPTQ (GGUF chỉ thử nghiệm) | GGUF | GGUF, import được safetensors |
| Dùng 2 GPU | `--tensor-parallel-size 2` hoặc `--pipeline-parallel-size 2` | `--split-mode layer/row/tensor`, `--tensor-split` | Tự chia khi model không vừa 1 card; `OLLAMA_SCHED_SPREAD=1` để luôn chia |
| Nhiều người cùng lúc | PagedAttention và continuous batching | `--parallel` và continuous batching | `OLLAMA_NUM_PARALLEL`, mặc định **1** |
| FP8 trên L40S | Có | Không, dùng quant GGUF | Như llama.cpp |
| Hợp khi | Phục vụ vài chục người, cần throughput | Thử model GGUF, cần chỉnh chia GPU, model lớn hơn VRAM (offload CPU) | Demo nhanh, người ít kỹ thuật, đổi model thường xuyên |
| Lệnh theo docs Unsloth | [vLLM](/export-deploy/chay-model#vllm) | [llama-server](/export-deploy/chay-model#llama-server) | [Ollama](/export-deploy/chay-model#ollama) |

<div class="dg">
  <div class="dg-title">Model đã fine-tune đi tới engine nào</div>
  <div class="dg-flow">
    <div class="dg-node is-main">Model sau fine-tune</div>
    <div class="dg-node"><div>safetensors 16-bit</div><small>merged_16bit</small></div>
    <div class="dg-node is-end">vLLM (production)</div>
  </div>
  <div class="dg-flow" style="margin-top: 14px">
    <div class="dg-node is-main">Model sau fine-tune</div>
    <div class="dg-node"><div>GGUF</div><small>q4_k_m, q8_0…</small></div>
    <div class="dg-node is-end">llama-server / Ollama (thử nghiệm)</div>
  </div>
  <div class="dg-cap">Dòng trên: lưu bằng save_pretrained_merged cho vLLM, có thể nén tiếp sang FP8 khi serve. Dòng dưới: GGUF cho các engine họ llama.cpp.</div>
</div>

**[Nguồn ngoài]** Tính năng và định dạng của vLLM lấy từ docs vLLM. Cờ chia GPU của llama-server lấy từ README llama.cpp. Hành vi chia GPU và giá trị mặc định của Ollama lấy từ FAQ Ollama.

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama, https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint, https://docs.vllm.ai/en/latest/, https://docs.vllm.ai/en/latest/features/quantization/gguf/, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md, https://docs.ollama.com/faq, https://docs.ollama.com/api/openai-compatibility

## Điều cần biết về L40S

L40S không có NVLink, nên hai card chỉ nói chuyện với nhau qua PCIe. Điều này quyết định cách chia model hợp lý.

**[Nguồn ngoài]** Theo trang sản phẩm NVIDIA:

| Mục | L40S |
| --- | --- |
| Kiến trúc | Ada Lovelace (compute capability 8.9) |
| VRAM | 48 GB GDDR6 có ECC |
| Băng thông VRAM | 864 GB/s |
| Kết nối | PCIe Gen4 x16, 64 GB/s hai chiều |
| NVLink | Không |
| MIG (chia 1 card thành nhiều GPU ảo) | Không |
| FP8 | Có (Transformer Engine) |
| Công suất | 350 W |

Hệ quả thực tế:

- **FP8 chạy được.** vLLM hỗ trợ FP8 trên GPU có compute capability từ 8.9, gồm Ada. Model 32B ở FP8 chiếm khoảng 32 GB, vừa một card.
- **NVFP4 không chạy.** Docs Unsloth ghi NVFP4 cần GPU Blackwell (RTX 50, B200…). Với GPU đời cũ, docs khuyên dùng GGUF. Xem [Định dạng trọng số NVFP4 và FP8](/export-deploy/nvfp4-fp8).
- **Đường nối giữa hai card chậm hơn nhiều so với bộ nhớ trong card** (64 GB/s so với 864 GB/s). Cách chia nào cần hai card trao đổi liên tục thì chịu chi phí này.

**Nguồn:** https://www.nvidia.com/en-us/data-center/l40s/, https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/, https://docs.ollama.com/gpu, https://unsloth.ai/docs/basics/nvfp4

## Chia model qua hai card

Nếu model vừa một card 48 GB, **chạy hai bản độc lập (mỗi card một bản) thường đơn giản và nhanh hơn chia một model qua hai card**. Chỉ khi model lớn hơn 48 GB mới phải chia.

<div class="dg">
  <div class="dg-title">Ba cách dùng 2 card</div>
  <div class="dg-grid" style="--cols: 3">
    <div class="dg-group">
      <div class="dg-glabel">Hai bản sao</div>
      <div class="dg-node">Card 0: model A</div>
      <div class="dg-node">Card 1: model A</div>
      <div class="dg-cap">Model phải vừa 48 GB. Không trao đổi qua PCIe. Cần load balancer phía trước.</div>
    </div>
    <div class="dg-group">
      <div class="dg-glabel">Tensor parallel (TP)</div>
      <div class="dg-node">Mỗi layer chia đôi cho 2 card</div>
      <div class="dg-cap">Hai card cùng tính rồi trao đổi kết quả sau mỗi layer, nên tốn đường PCIe.</div>
    </div>
    <div class="dg-group">
      <div class="dg-glabel">Pipeline parallel (PP)</div>
      <div class="dg-node">Card 0: nửa đầu layer</div>
      <div class="dg-node">Card 1: nửa sau layer</div>
      <div class="dg-cap">Ít trao đổi hơn, nhưng mỗi request đi qua hai card lần lượt.</div>
    </div>
  </div>
</div>

### vLLM: TP hay PP

::: warning Docs chưa thống nhất: TP hay PP cho L40S
| Nguồn | Nói gì |
| --- | --- |
| [Docs vLLM, Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html) | Khi GPU không có NVLink, "ví dụ L40S", nên dùng pipeline parallel thay tensor parallel để throughput cao hơn |
| [Thread diễn đàn vLLM](https://discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848) | Đo trên đúng 2×L40S PCIe 4.0, model Qwen3.6-35B-A3B-FP8: TP=2 đạt 0.95 req/s, PP=2 đạt 0.72 req/s |
:::

**[Nhận định]** Thread diễn đàn chỉ là một lần đo, trên một model MoE và một kiểu tải. Hãy chạy thử cả hai với dữ liệu thật của đơn vị rồi chọn:

```bash
# Tensor parallel
vllm serve ./model-merged --tensor-parallel-size 2
# Pipeline parallel
vllm serve ./model-merged --pipeline-parallel-size 2
```

### llama-server và Ollama

- **llama-server:** `--split-mode layer` (mặc định) chia các layer và KV cache cho hai card, chạy nối tiếp. `row` chia trọng số theo hàng để hai card tính song song. `tensor` còn đang thử nghiệm. `--tensor-split 1,1` đặt tỉ lệ chia.
- **Ollama:** nếu model vừa một card thì Ollama chỉ load lên một card. Không vừa thì tự chia cho mọi card. Đặt `OLLAMA_SCHED_SPREAD=1` để luôn chia. Chọn card bằng `CUDA_VISIBLE_DEVICES`, nên dùng UUID lấy từ `nvidia-smi -L`.

**Nguồn:** https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html, https://discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md, https://docs.ollama.com/faq, https://github.com/ollama/ollama/blob/main/envconfig/config.go, https://docs.ollama.com/gpu

## Model cỡ nào vừa 96 GB

VRAM phải chứa hai thứ: trọng số model, và KV cache (bộ nhớ lưu ngữ cảnh của từng người đang hỏi). Model càng nhỏ thì càng còn chỗ cho nhiều người dùng cùng lúc.

**[Ước tính]** Cách tính:

- Trọng số ≈ số tham số × số byte mỗi tham số. BF16 = 2 byte, FP8 = 1 byte, 4-bit ≈ 0.5–0.6 byte.
- KV cache mỗi token = 2 × số layer attention × số KV head × head_dim × số byte. Các thông số này lấy từ `config.json` của model trên Hugging Face.
- vLLM mặc định dùng 92% VRAM (`--gpu-memory-utilization 0.92`), tức khoảng 88 GB trên 96 GB. Trừ thêm khoảng 4 GB overhead (giả định, chưa đo).

| Model | Độ chính xác | Trọng số | Còn cho KV cache | Ví dụ 30 người dùng cùng lúc |
| --- | --- | --- | --- | --- |
| Dense 70B (Llama-3.3-70B) | BF16 | ~140 GB | Không vừa | — |
| Dense 70B | FP8 | ~70 GB | ~14 GB | Mỗi người ~1.5K token ngữ cảnh, chật |
| Dense 70B | 4-bit (AWQ/GPTQ) | ~40 GB | ~44 GB | Mỗi người ~4.5K token |
| Dense 32B (Qwen3-32B) | BF16 | ~64 GB | ~20 GB | Mỗi người ~2.7K token |
| Dense 32B | FP8 | ~32 GB | ~52 GB | Mỗi người ~7K token (KV FP8: ~14K) |

- KV cache FP8 (`--kv-cache-dtype fp8`) giảm một nửa bộ nhớ KV, tức gấp đôi số token trong bảng.
- Model dùng attention lai, như Qwen3.5 (chỉ 1/4 số layer là full attention), tốn KV ít hơn nhiều. Con số chính xác chưa xác minh.
- Các số trên là tổng token đang hoạt động cùng lúc. PagedAttention cấp phát theo nhu cầu, không phải ai cũng dùng hết context.

**[Nhận định]** Điểm cân bằng trên 96 GB là **model 27–35B ở FP8** (vừa một card, có thể chạy hai bản sao) hoặc **70B ở 4-bit**. Khi vLLM khởi động, dòng log "KV cache size / maximum concurrency" cho biết con số thật. Hãy đọc dòng đó thay vì tin bảng này.

**Nguồn:** https://docs.vllm.ai/en/latest/configuration/engine_args/, https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/, https://huggingface.co/Qwen/Qwen3-32B/resolve/main/config.json, https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/resolve/main/config.json, https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/config.json

## Kiến trúc đề xuất cho đơn vị

**[Nhận định]** Toàn bộ mục này là đề xuất của người viết, ghép từ các dữ kiện ở trên.

<div class="dg">
  <div class="dg-title">Một cách bố trí 2×L40S</div>
  <div class="dg-flow">
    <div class="dg-node is-ghost">Người dùng, ứng dụng nội bộ</div>
    <div class="dg-node"><div>Gateway / reverse proxy</div><small>API OpenAI</small></div>
    <div class="dg-node is-main"><div>Card 1: vLLM</div><small>model 27–35B, FP8</small></div>
  </div>
  <div class="dg-flow" style="margin-top: 14px">
    <div class="dg-node is-ghost">Dataset mới của đơn vị</div>
    <div class="dg-node"><div>Card 0: Unsloth fine-tune</div><small>ngoài giờ train: chạy vLLM bản 2</small></div>
    <div class="dg-node is-end"><div>Model mới</div><small>nạp lại vào vLLM</small></div>
  </div>
  <div class="dg-cap">Dòng trên phục vụ người dùng, dòng dưới train. Tách theo card hoặc theo giờ, vì vLLM chiếm trước khoảng 92% VRAM của card nó dùng.</div>
</div>

1. **Tách train và serve.** vLLM chiếm trước hầu hết VRAM nên không train trên cùng card được. Hoặc chia theo giờ (train ban đêm), hoặc chia theo card (card 0 train, card 1 serve).
2. **Production dùng vLLM.**
   - Lưu model fine-tune bằng `save_method="merged_16bit"`. Docs Unsloth khuyên dùng cách này cho vLLM.
   - Nén sang FP8 khi serve (trang FP8 của docs vLLM hiện ghi cờ `--quantization fp8_per_tensor`) và bật `--kv-cache-dtype fp8`.
   - Nếu có nhiều bản LoRA (mỗi phòng ban một bản), bật `--enable-lora` để phục vụ chung một base model. Docs Unsloth có trang [LoRA Hot Swapping](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide).
3. **Thử nghiệm dùng Ollama hoặc llama-server** với GGUF. Nếu nhiều người thử cùng lúc thì tăng `OLLAMA_NUM_PARALLEL`.
4. **Đặt một gateway phía trước.** Cả ba engine đều nói chuẩn OpenAI, nên đổi engine không phải sửa ứng dụng.
5. **Không đầu tư vào NVFP4** trên máy này.
6. **Mở ra mạng nội bộ có kiểm soát.** Xem [Mở cho máy khác (LAN, Cloudflare)](/export-deploy/lan-remote). Với dữ liệu cơ quan nhà nước, không nên mở ra Internet.

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide, https://docs.vllm.ai/en/latest/configuration/engine_args/, https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/

## Đọc tiếp

- [Đề bài và đáp án nhanh](/bai-toan/) — quay lại bản tóm tắt cả bài toán.
- [Chạy model đã xuất](/export-deploy/chay-model) — lệnh cài và chạy chi tiết của từng engine theo docs Unsloth.
- [Mở cho máy khác (LAN, Cloudflare)](/export-deploy/lan-remote) — cho máy khác trong đơn vị gọi vào server.
- [Fine-tune bằng Unsloth](/bai-toan/fine-tune) — tạo ra model để đem đi host.
