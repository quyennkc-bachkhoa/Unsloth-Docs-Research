---
title: Chọn hyperparameter
description: "Giá trị khuyến nghị cho learning rate, epoch, rank, alpha, batch... và bảng so sánh mặc định giữa Studio, tutorial và hướng dẫn hyperparameter."
---

# Chọn hyperparameter

Hyperparameter là các con số bạn đặt trước khi train, ví dụ learning rate hay số epoch. Docs khuyên **giữ mặc định của Unsloth**, trừ khi bạn cần train lâu hơn hoặc batch lớn hơn.

::: tip Tóm tắt
- **Dùng khi:** bạn cấu hình một lần train và cần biết nên đặt learning rate, epoch, rank, alpha, batch… bao nhiêu.
- **Kết quả:** có bảng giá trị khuyến nghị kèm hậu quả khi đặt sai, và thấy rõ mặc định khác nhau giữa Studio, tutorial và hướng dẫn hyperparameter.
- **Nên biết trước:** [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) (`q_proj` … `down_proj` là lớp nào), [Token & context](/kien-thuc-nen/token-va-context) (`max_seq_length`), [Dense & MoE](/kien-thuc-nen/dense-va-moe) (model như `30B-A3B`).
:::

## Giá trị khuyến nghị cho từng tham số

Vấn đề là "mặc định" không giống nhau giữa ba chỗ: Studio, tutorial hoặc notebook, và hướng dẫn hyperparameter. Nhiều tham số **không trùng nhau**; bảng so sánh nằm ngay sau bảng dưới đây. Cách đọc bảng:

- Cột "Đặt sai thì sao" diễn giải từ chính docs.
- Tham số có dấu ⚠ là tham số docs ghi nhiều giá trị khác nhau. Chi tiết nằm trong hộp "Docs chưa thống nhất" bên dưới.

| Tham số | Giá trị khuyến nghị (docs) | Ý nghĩa ngắn | Đặt sai thì sao |
| --- | --- | --- | --- |
| `learning_rate` | `2e-4` cho LoRA và QLoRA; `5e-6` cho RL (DPO, GRPO...); FFT dùng thấp hơn. Khoảng thường gặp `2e-4` → `5e-6`. Tutorial gợi ý thử `2e-4`, `1e-4`, `5e-5`, `2e-5` | Mức điều chỉnh trọng số mỗi bước | Quá cao: train bất ổn, dễ overfit trong run ngắn. Quá thấp: cần nhiều epoch hơn, có thể overfit hoặc không học được |
| Epochs (`num_train_epochs`) ⚠ | Hướng dẫn hyperparameter: 1–3. Tutorial: `max_steps = 60` để chạy thử, chạy thật `num_train_epochs = 1`. Studio mặc định 3 | Số lần model đi qua toàn bộ dataset | Hơn 3 epoch với dataset instruction: lợi ích giảm, dễ học thuộc. Quá ít: train chưa đủ |
| `r` (LoRA rank) ⚠ | Bảng khuyến nghị: 8, 16, 32, 64, 128; nên chọn 16 hoặc 32. Mục underfitting cùng trang: "usually is between 4 and 64". Studio: slider 4–128 | Số tham số train được trong adapter | Lớn: tốn bộ nhớ, chậm hơn; rank quá lớn có thể overfit. Nhỏ: có thể underfit |
| `lora_alpha` ⚠ | Bằng `r`, hoặc `r * 2`; giữ alpha/rank ≥ 1. Snippet mẫu dùng `16` (= r); Studio mặc định `32` | Hệ số scale độ mạnh của cập nhật LoRA | Lớn: học mạnh hơn nhưng dễ overfit |
| `lora_dropout` ⚠ | Code mẫu: 0 (được tối ưu), khoảng 0 – 0.1. Studio mặc định `0.05` | Regularization (điều chuẩn): ngẫu nhiên tắt một phần activation | Docs: không hữu ích lắm với run ngắn; dùng giá trị khác 0 nếu nghi overfit |
| `weight_decay` | 0.01 (khuyến nghị) – 0.1; Studio mặc định 0.01 | Phạt trọng số lớn để giảm overfit | Docs cảnh báo không dùng giá trị quá lớn |
| Warmup ⚠ | Hướng dẫn hyperparameter: 5–10% tổng số bước. Studio mặc định Warmup Steps = 5 | Tăng dần learning rate lúc đầu | — (docs không nêu) |
| Scheduler | `linear` hoặc `cosine`; Studio mặc định `linear` | Điều chỉnh learning rate theo thời gian | — (docs không nêu) |
| `random_state` (seed) | Số nguyên bất kỳ, ví dụ `42`, `3407` | Cố định yếu tố ngẫu nhiên để tái lập kết quả | Không cố định: khó so sánh các lần chạy |
| `target_modules` | Tất cả lớp linear chính: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Các lớp được gắn LoRA (attention + MLP) | Bỏ bớt module: tiết kiệm bộ nhớ rất ít nhưng giảm chất lượng; docs "strongly advise against" |
| `bias` ⚠ | Code: `"none"`; lời giải thích trong tutorial lại ghi "Leave this as 0" | Có train bias hay không | Train bias: thêm tham số, gần như không lợi |
| `use_gradient_checkpointing` | `"unsloth"` | Tiết kiệm bộ nhớ khi train | `"unsloth"` giảm thêm 30% bộ nhớ và hỗ trợ context rất dài so với `True` hoặc `False` |
| `per_device_train_batch_size` ⚠ | Tutorial và hướng dẫn hyperparameter: 2. Studio mặc định 4 | Số mẫu mỗi lượt forward/backward trên 1 GPU — **yếu tố chính quyết định VRAM** | Quá lớn: OOM (hết bộ nhớ); tăng batch còn có thể chậm hơn do padding |
| `gradient_accumulation_steps` ⚠ | Hướng dẫn hyperparameter: 8. Tutorial và Fine-tuning LLMs Guide: 4. Studio mặc định 8 | Số micro-batch trước mỗi lần cập nhật — **yếu tố chính quyết định thời gian train** | Cao: mỗi epoch lâu hơn |
| Effective batch size ⚠ | Hướng dẫn hyperparameter: 4–16, khuyến nghị 16 (= 2 × 8). Mặc định tutorial cho ra 8 (= 2 × 4); mặc định Studio cho ra 32 (= 4 × 8) | `batch_size * gradient_accumulation_steps` | Nhỏ: nhiều nhiễu hơn; lớn: ổn định hơn |
| `use_rslora` ⚠ | `False` (tính năng nâng cao). Hai trang mô tả tác dụng khác nhau | Rank-Stabilized LoRA | Theo hướng dẫn hyperparameter: có thể ổn định hơn với rank cao |
| `loftq_config` | `None` | Khởi tạo LoRA từ singular vectors của trọng số gốc | Có thể tăng độ chính xác nhưng tăng vọt bộ nhớ lúc bắt đầu |
| `max_seq_length` | 2048 để thử nghiệm; Studio mặc định 2048 (tùy chọn 512 → 32768) | Độ dài context khi train | — |

**Batch size và gradient accumulation.** Hai tham số này cùng quyết định effective batch, nhưng tốn VRAM rất khác nhau. Ví dụ, muốn effective batch = 32, các cấu hình `32×1`, `16×2`, `8×4`, `4×8`, `2×16`, `1×32` cho cùng kết quả cập nhật trọng số, nhưng khác xa về VRAM. Docs khuyên đặt `batch_size` nhỏ rồi tăng `gradient_accumulation_steps`. Unsloth đã sửa lỗi gradient accumulation, nên trong Unsloth hai cách này **tương đương hoàn toàn**.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/new/studio/start

## So sánh mặc định: Studio vs Core

Bảng này đặt ba bộ "mặc định" cạnh nhau. "Core tutorial" là tutorial Llama-3 + Ollama, lấy giá trị trong notebook. Batch, gradient accumulation, max_steps và learning rate ở đây trùng với phần Training của Fine-tuning LLMs Guide. Dấu "—" nghĩa là trang đó không nêu giá trị.

| Tham số | [Studio](https://unsloth.ai/docs/new/studio/start) | [Core tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) | [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) |
| --- | --- | --- | --- |
| Batch size | 4 | `per_device_train_batch_size = 2` | 2 |
| Gradient accumulation | 8 | `gradient_accumulation_steps = 4` | 8 |
| Effective batch (tính ra) | 32 | 8 | 16 (khuyến nghị 4–16) |
| Epochs / steps | Epochs 3; Max Steps `0` (= dùng Epochs) | `max_steps = 60` để chạy thử; chạy thật `num_train_epochs = 1`; gợi ý 1–3 lượt | 1–3 epoch |
| Learning rate | `2e-4` | `2e-4` | `2e-4` (LoRA/QLoRA) |
| Context length | 2048 (512 → 32768) | `max_seq_length = 2048` | — |
| Rank `r` | 16 (slider 4–128) | `r = 16` | 16 hoặc 32 (khoảng 8–128; mục underfitting: 4–64) |
| `lora_alpha` | 32 | `lora_alpha = 16` | `r` hoặc `r * 2`; snippet `lora_alpha = 16` |
| `lora_dropout` | 0.05 | `lora_dropout = 0` | 0 (mặc định) – 0.1 |
| `bias` | — | `bias = "none"` (lời giải thích ghi "Leave this as 0") | `"none"` |
| Weight decay | 0.01 | — | 0.01 – 0.1 |
| Warmup | 5 steps | — | 5–10% tổng số bước |
| LR scheduler | linear | — | `linear` hoặc `cosine` |
| Optimizer | AdamW 8-bit | — | — |
| Gradient checkpointing | `unsloth` | `"unsloth"` | `"unsloth"` |
| Seed | 3407 | `random_state = 3407` | Số nguyên bất kỳ (ví dụ 42, 3407) |
| Target modules | Bật hết 7 module | Cả 7 module | Cả 7 module |
| LoRA variant | `LoRA` / `RS-LoRA` / `LoftQ` (mặc định LoRA) | `use_rslora = False`, `loftq_config = None` | `use_rslora = False`, `loftq_config = None` |
| Train on completions | false | — | Khuyên dùng (tăng độ chính xác, nhất là hội thoại nhiều lượt) |
| Packing | false | — | — |
| Save / Eval steps | 0 / 0 | — | — |

::: warning Docs chưa thống nhất: giá trị mặc định/khuyến nghị
Các trang docs không đưa cùng một bộ "mặc định". Trang này liệt kê đủ các giá trị và không chọn thay bạn:
- **Batch size:** 4 — [Studio](https://unsloth.ai/docs/new/studio/start); 2 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).
- **Gradient accumulation:** 8 — [Studio](https://unsloth.ai/docs/new/studio/start), [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); 4 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide).
- **Epochs:** 3 — [Studio](https://unsloth.ai/docs/new/studio/start); `max_steps = 60` (chạy thử) hoặc `num_train_epochs = 1` (chạy thật) — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide); 1–3 — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **`lora_alpha`:** 32 — [Studio](https://unsloth.ai/docs/new/studio/start); 16 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama) và snippet của [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); "`r` hoặc `r * 2`" — bảng của [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **`lora_dropout`:** 0.05 — [Studio](https://unsloth.ai/docs/new/studio/start); 0 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) (khoảng 0–0.1).
- **Warmup:** 5 steps — [Studio](https://unsloth.ai/docs/new/studio/start); 5–10% tổng số bước — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **Train on completions:** mặc định false — [Studio](https://unsloth.ai/docs/new/studio/start); khuyên dùng — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **Khoảng rank:** 8, 16, 32, 64, 128 (chọn 16 hoặc 32) — bảng khuyến nghị của [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); "usually is between 4 and 64" — mục Underfitting cùng trang [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); slider 4–128 — [Studio](https://unsloth.ai/docs/new/studio/start).
- **`bias`:** code `bias = "none"` nhưng lời giải thích ghi "Leave this as 0" — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama); "Leave this as `"none"`" — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- **`use_rslora`:** "Advanced feature to set the `lora_alpha = 16` automatically" — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama); nếu `True` thì scaling thành `lora_alpha / sqrt(r)` thay vì `lora_alpha / r` — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/new/studio/start

## Đọc tiếp

- [Đánh giá và overfitting](/fine-tuning/danh-gia) — theo dõi loss để biết bộ tham số vừa chọn có làm model học thuộc hay không.
- [Quy trình từng bước](/fine-tuning/quy-trinh) — xem các tham số này nằm ở đâu trong code Core và trong Studio.
- [Lỗi thường gặp](/fine-tuning/loi-thuong-gap) — các lỗi hay gặp liên quan đến epoch và batch size.
