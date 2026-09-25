---
title: Quy trình từng bước
description: "Các bước chọn model, chuẩn bị dữ liệu, train, đánh giá, export; so sánh Studio và Core, kèm code Core chép nguyên văn từ docs."
---

# Quy trình từng bước

Một lượt fine-tune đi qua các bước: chọn model, chuẩn bị dữ liệu, đặt tham số, train, đánh giá, rồi lưu và export. Nếu đánh giá chưa đạt, bạn quay lại bước dữ liệu.

::: tip Tóm tắt
- **Dùng khi:** bạn fine-tune lần đầu và muốn biết cần làm gì ở từng bước, trong Studio hoặc bằng code Python.
- **Kết quả:** nắm được các bước từ chọn model tới export, biết mỗi bước làm ở đâu trong Studio và gọi hàm nào trong Core, có code mẫu chép từ docs.
- **Nên biết trước:** đã chọn cách train và model ở [Chọn cách train và model](/fine-tuning/chon-cach-train).
:::

## Các bước trong Studio và Core

Mỗi bước dưới đây đều làm được bằng giao diện Studio hoặc bằng code Core; sơ đồ cho thấy thứ tự và vòng quay lại khi chưa đạt.

<div class="dg">
<div class="dg-flow">
<div class="dg-node"><div><span class="dg-n">1</span>Chọn model</div><small>+ phương pháp</small></div>
<div class="dg-node"><div><span class="dg-n">2</span>Dataset</div><small>+ chat template</small></div>
<div class="dg-node"><div><span class="dg-n">3</span>Hyperparameter</div><small>và cấu hình LoRA</small></div>
<div class="dg-node"><div><span class="dg-n">4</span>Train</div><small>theo dõi loss</small></div>
<div class="dg-node is-main"><div><span class="dg-n">5</span>Đánh giá</div><small>eval loss + chat thử</small></div>
<div class="dg-node" data-e="Đạt"><div><span class="dg-n">6</span>Lưu / Export</div><small>LoRA, GGUF, safetensors</small></div>
<div class="dg-node is-end"><div><span class="dg-n">7</span>Chạy</div><small>Unsloth, Ollama, llama.cpp, vLLM</small></div>
<div class="dg-back" style="grid-column: 2 / 6"><span>Chưa đạt: quay lại bước 2</span></div>
</div>
</div>

Unsloth có hai cách làm. **Studio** là giao diện web chạy local, không cần code. **Core** là thư viện Python, dùng qua notebook hoặc script. Bảng sau đặt hai cách cạnh nhau theo từng bước.

| Bước | Studio (UI) | Core (code) |
| --- | --- | --- |
| 1. Model + phương pháp | Chọn Model Type (Text, Vision, Audio hoặc Embeddings). Chọn QLoRA, LoRA hoặc Full Fine-tuning. Gõ tên model Hugging Face hoặc chọn model local. Studio tự điền hyperparameter mặc định. Model GGUF không train được, chỉ để inference. Model gated (Llama, Gemma) cần dán Hugging Face token. | `FastLanguageModel.from_pretrained(...)` với `max_seq_length`, `dtype`, `load_in_4bit` |
| 2. Dataset | Tab HuggingFace Hub hoặc Local (upload `PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet`). Chọn format `auto`, `alpaca`, `chatml` hoặc `sharegpt`. Chọn Train split và Eval split. Dùng Column Mapping nếu Studio không tự nhận cột. | `load_dataset(...)`, `get_chat_template`, `standardize_sharegpt`, `dataset.map(...)` — chi tiết ở trang [Dữ liệu](/du-lieu/) |
| 3. Hyperparameter | Các nhóm tham số thu gọn được. Lưu và tải cấu hình dạng YAML. | `FastLanguageModel.get_peft_model(...)` + tham số của trainer |
| 4. Train | Bấm **Start Training**. Theo dõi Loss, LR, Grad Norm và GPU (utilization, nhiệt độ, VRAM, công suất). Có biểu đồ Training Loss, Learning Rate, Gradient Norm, Eval Loss. Nút **Stop & Save** lưu checkpoint trước khi dừng. | `SFTTrainer(...)` rồi `trainer.train()` |
| 5. Đánh giá | Biểu đồ Eval Loss, chỉ hiện khi đã chọn eval split. Chat thử model trong Studio Chat. | `eval_dataset`, `eval_strategy`, `EarlyStoppingCallback`; `FastLanguageModel.for_inference(model)` rồi chat thử |
| 6. Export | Trang **Export**: GGUF, Safetensors hoặc LoRA, dùng cho Unsloth, llama.cpp, Ollama, vLLM... | Lưu LoRA adapter (~100MB) hoặc push lên Hugging Face; export GGUF cho Ollama — xem [Export và deploy](/export-deploy/) |

**Nguồn:** https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama

## Code Core tiêu biểu

Trong các trang nguồn, code Core chỉ xuất hiện theo từng mảnh. Bản đầy đủ chạy được nằm trong các notebook (xem trang [Notebook chạy sẵn](/fine-tuning/notebooks)). Các mảnh dưới đây được chép nguyên văn.

**Nạp model + gắn LoRA.** Trong các trang nguồn, chỉ có một khối code chứa đủ cả `from_pretrained` và `get_peft_model`: ví dụ ở trang QAT. Vì vậy khối đó được giữ nguyên văn. Dòng `qat_scheme` chỉ dùng khi làm QAT (xem mục QAT ở trang [Kỹ thuật nâng cao](/fine-tuning/mo-rong)). Khi fine-tune LoRA thường, bạn bỏ dòng này.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = 2048,
    load_in_16bit = True,
)
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

Các tùy chọn nạp model mà tutorial giải thích từng dòng:

```python
max_seq_length = 2048
```

```python
dtype = None
```

```python
load_in_4bit = True
```

Các tham số của `get_peft_model` theo hướng dẫn hyperparameter. Trong nguồn, mỗi dòng là một snippet riêng:

```python
r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
```

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",],
```

```python
lora_alpha = 16,
```

```python
lora_dropout = 0, # Supports any, but = 0 is optimized
```

```python
bias = "none",    # Supports any, but = "none" is optimized
```

```python
use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
```

```python
random_state = 3407,
```

**Trainer.** Unsloth dùng `SFTTrainer` của thư viện TRL. Mẫu dưới đây lấy từ trang checkpoint; `....` là phần docs lược bỏ.

```python
trainer = SFTTrainer(
    ....
    args = TrainingArguments(
        ....
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 50,
    ),
)
```

Các tham số train mặc định trong tutorial:

```python
per_device_train_batch_size = 2,
```

```python
gradient_accumulation_steps = 4,
```

```python
max_steps = 60, # num_train_epochs = 1,
```

```python
learning_rate = 2e-4,
```

::: warning Docs chưa thống nhất: các giá trị mẫu ở trên
Các giá trị mẫu vừa nêu không giống nhau giữa các trang docs:

- `gradient_accumulation_steps`: 4 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide); 8 — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Studio](https://unsloth.ai/docs/new/studio/start).
- `per_device_train_batch_size`: 2 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); 4 — [Studio](https://unsloth.ai/docs/new/studio/start).
- `lora_alpha`: 16 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); 32 — [Studio](https://unsloth.ai/docs/new/studio/start), ví dụ [QAT](https://unsloth.ai/docs/blog/quantization-aware-training-qat) ở trên.
- `lora_dropout`: 0 — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide); 0.05 — [Studio](https://unsloth.ai/docs/new/studio/start).
- Số epoch: `max_steps = 60` / `num_train_epochs = 1` — [Tutorial](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama); 3 — [Studio](https://unsloth.ai/docs/new/studio/start); 1–3 — [Hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).

Bảng đầy đủ ở mục "So sánh mặc định: Studio vs Core" của trang [Chọn hyperparameter](/fine-tuning/hyperparameter).
:::

**Chỉ train trên câu trả lời** (train on completions). Cách này che phần input của user và chỉ tính loss trên phần assistant trả lời. Docs dẫn bài báo QLoRA: cách này tăng độ chính xác, nhất là với hội thoại nhiều lượt. Với Llama 3, 3.1, 3.2, 3.3 và 4:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

Với Gemma 2, 3, 3n:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```

**Sau khi train:**

- Gọi `FastLanguageModel.for_inference(model)` để chat thử. Docs nói inference native nhanh gấp 2×.
- Muốn câu trả lời dài hơn, tăng `max_new_tokens = 128` lên 256 hoặc 1024.
- Lưu model dưới dạng LoRA adapter, khoảng 100MB. Hoặc push lên Hugging Face (cần [token](https://huggingface.co/settings/tokens)).
- Khi export GGUF trong notebook Ollama, chỉ đổi `False` thành `True` ở **một** dòng, thường là `Q8_0`. Đừng bật tất cả, nếu không bạn sẽ phải chờ rất lâu. Quá trình export mất 5–10 phút.
- Unsloth tự tạo `Modelfile` cho Ollama.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint, https://unsloth.ai/docs/blog/quantization-aware-training-qat, https://unsloth.ai/docs/new/studio/start

## Đọc tiếp

- [Chọn hyperparameter](/fine-tuning/hyperparameter) — chọn giá trị cho các tham số đã thấy trong code mẫu và biết đặt sai thì sao.
- [Đánh giá và overfitting](/fine-tuning/danh-gia) — đọc loss ở bước 4–5 để biết model học thật hay học thuộc.
- [Dữ liệu](/du-lieu/) — chuẩn bị dataset và chat template cho bước 2.
- [Export và deploy](/export-deploy/) — chi tiết bước lưu và export sau khi train.
