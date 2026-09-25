---
title: Kỹ thuật nâng cao
description: "Multi-GPU, vision fine-tuning, continued pretraining, resume từ checkpoint và QAT."
---

# Kỹ thuật nâng cao

Trang này dành cho lúc bạn đã fine-tune cơ bản xong và muốn đi tiếp: train trên nhiều GPU, train model nhận ảnh, dạy model ngôn ngữ hoặc lĩnh vực mới, train tiếp từ checkpoint, và train sẵn cho bản 4-bit (QAT).

::: tip Tóm tắt
- **Dùng khi:** bạn đã fine-tune cơ bản xong và cần nhiều GPU, model nhận ảnh, dạy ngôn ngữ hoặc lĩnh vực mới, train tiếp từ checkpoint, hoặc giữ chất lượng khi chạy 4-bit.
- **Kết quả:** biết cách bật từng kỹ thuật (DDP, `FastVisionModel`, CPT, `resume_from_checkpoint`, `qat_scheme`) kèm code mẫu từ docs.
- **Nên biết trước:** code nạp model và trainer ở [Quy trình từng bước](/fine-tuning/quy-trinh).
:::

## Multi-GPU

Unsloth hiện hỗ trợ nhiều GPU qua Accelerate và DeepSpeed. Nhờ vậy bạn dùng được FSDP và DDP. DDP (Distributed Data Parallel) là cách mỗi GPU giữ một bản model và dữ liệu được chia cho các GPU. Docs thừa nhận việc thiết lập còn thủ công và hứa "official multi-GPU support soon".

Cách bật DDP:

1. Viết script train `train.py`. Bạn có thể lấy từ [training scripts](https://github.com/unslothai/notebooks/tree/main/python_scripts), là các notebook đã chuyển thành script.
2. Chạy `accelerate launch train.py` hoặc `torchrun --nproc_per_node N_GPUS train.py`, với `N_GPUS` là số GPU.

Nếu một GPU không đủ VRAM để nạp model (ví dụ Llama 70B), dùng `device_map = "balanced"` để chia model qua các GPU:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    load_in_4bit = True,
    device_map = "balanced",
)
```

**Nguồn:** https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth

## Vision fine-tuning

VLM (vision-language model) là model nhận cả ảnh và chữ. Bạn fine-tune VLM cho các tác vụ như phân tích X-quang hay chuyển chữ viết tay sang LaTeX. Trong code, bạn dùng `FastVisionModel`. Bạn có thể chọn chỉ train lớp vision, lớp ngôn ngữ, attention hoặc MLP; mặc định bật hết:

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
```

Trainer dùng một data collator riêng cho vision:

```python
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = dataset,
    args = SFTConfig(...),
)
```

Lưu ý từ docs:

- Ảnh trong dataset nên cùng kích thước, khoảng 300–1000px, để train không quá lâu.
- Với multi-image, dùng list comprehension thay cho `ds.map(...)`.
- Muốn train-on-responses-only cho VLM, bật qua các tham số `train_on_responses_only`, `instruction_part`, `response_part` của `UnslothVisionDataCollator`.
- Định dạng dữ liệu vision: xem trang [Định dạng dữ liệu](/du-lieu/dinh-dang#vision-anh-chu).

**Nguồn:** https://unsloth.ai/docs/basics/vision-fine-tuning

## Continued pretraining (CPT)

CPT là huấn luyện tiếp trên văn bản thô. Nó giúp model học một ngôn ngữ mới, hoặc một lĩnh vực ngoài phân phối như luật, y khoa. Docs có [notebook text completion](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_\(7B\)-Text_Completion.ipynb) và [notebook CPT học ngôn ngữ khác](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-CPT.ipynb).

Khi CPT, bạn có thể train cả `lm_head` và `embed_tokens`. Trên Colab, nếu hết bộ nhớ với Llama-3 8b thì chỉ thêm `lm_head`. Phần embedding dùng learning rate riêng, nhỏ hơn 2–10 lần:

```python
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    ....
    args = UnslothTrainingArguments(
        ....
        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6, # 2-10x smaller than learning_rate
    ),
)
```

Bạn có thể nạp lại LoRA adapter đã lưu để train tiếp. Cách làm: truyền tên adapter vào `model_name` của `FastLanguageModel.from_pretrained`. Lưu ý optimizer state sẽ bị reset.

**Nguồn:** https://unsloth.ai/docs/basics/continued-pretraining

## Resume từ checkpoint

Resume giúp bạn train tiếp từ chỗ đã dừng thay vì chạy lại từ đầu. Trước hết, thêm `save_strategy` và `save_steps` vào `TrainingArguments`. Ví dụ ở trang [Quy trình từng bước](/fine-tuning/quy-trinh) lưu mỗi 50 bước vào thư mục `outputs` (xem khối `SFTTrainer`). Sau đó gọi:

```python
trainer_stats = trainer.train(resume_from_checkpoint = True)
```

Trainer sẽ tiếp tục từ checkpoint gần nhất. Docs cũng hướng dẫn tích hợp Weights & Biases (`report_to = "wandb"`) để lưu checkpoint dạng artifact rồi resume từ đó. Trong Studio, nút **Stop & Save** lưu checkpoint trước khi dừng.

**Nguồn:** https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint, https://unsloth.ai/docs/new/studio/start

## QAT (Quantization-Aware Training)

QAT (huấn luyện có mô phỏng lượng tử hóa) giúp model chạy 4-bit mà mất ít độ chính xác hơn. Trong lúc train, QAT "fake quantize" trọng số để model quen dần với sai số 4-bit. Cách thường gặp là lượng tử hóa sau khi train (PTQ); QAT làm việc này ngay trong lúc train. QAT làm cùng PyTorch và TorchAO.

Docs Unsloth ghi các con số sau:

- QAT khôi phục tới **70%** độ chính xác bị mất.
- Cải thiện **1–3%** trên benchmark như GPQA, MMLU Pro.
- Gemma3-4B trên GPQA khôi phục 66.9% (+1.0% độ chính xác thô). Gemma3-12B trên BBH khôi phục 45.5% (+2.1%).
- Không tốn thêm chi phí lúc inference.

Cách dùng:

- Bật qua tham số `qat_scheme` trong `get_peft_model` (xem khối code ở trang [Quy trình từng bước](/fine-tuning/quy-trinh)). Các giá trị hỗ trợ: `fp8-int4`, `fp8-fp8`, `int8-int4`, `int4`.
- Sau khi train: gọi `quantize_(model, QATConfig(step = "convert"))` rồi `model.save_pretrained_torchao(...)`.
- Cài đặt:

```bash
pip install --upgrade --no-cache-dir --force-reinstall unsloth unsloth_zoo
pip install torchao==0.14.0 fbgemm-gpu-genai==1.3.0
```

Thử với [notebook Qwen3 (4B) QAT](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(4B\)_Instruct-QAT.ipynb).

**Nguồn:** https://unsloth.ai/docs/blog/quantization-aware-training-qat

## Đọc tiếp

- [Lỗi thường gặp](/fine-tuning/loi-thuong-gap) — rà lại các lỗi hay gặp trước khi chạy một lần train lớn.
- [Định dạng dữ liệu](/du-lieu/dinh-dang) — chuẩn bị dữ liệu ảnh và chữ cho vision fine-tuning.
- [Export và deploy](/export-deploy/) — lưu và xuất model sau khi train, kể cả bản lượng tử hóa.
