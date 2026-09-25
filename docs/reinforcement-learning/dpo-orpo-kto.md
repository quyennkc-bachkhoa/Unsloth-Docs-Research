---
title: Train theo cặp tốt/xấu (DPO, ORPO, KTO)
description: "Nhóm phương pháp căn chỉnh model theo câu trả lời được ưa thích: danh sách notebook, code DPO từ docs và những chỗ docs còn thiếu."
---

# Train theo cặp tốt/xấu (DPO, ORPO, KTO)

DPO, ORPO và KTO dạy model bằng cách so sánh: với cùng một câu hỏi, câu trả lời nào được ưa thích hơn. Bạn không cần viết reward function như GRPO; trang này liệt kê các notebook, chép code DPO từ docs và chỉ ra những chỗ docs còn thiếu.

::: tip Tóm tắt
- **Dùng khi:** bạn có sẵn dữ liệu kiểu "câu này tốt hơn câu kia" và muốn căn chỉnh model theo đó thay vì viết reward function.
- **Kết quả:** biết Unsloth chạy được những phương pháp preference nào, notebook nào để bắt đầu, và đọc hiểu đoạn code DPO mẫu cùng các chỗ cần tự bổ sung.
- **Nên biết trước:** [Nên chọn SFT, DPO hay GRPO](/reinforcement-learning/#chon-phuong-phap), [Fine-tuning](/fine-tuning/) (code mẫu xuất phát từ model đã SFT).
:::

## Các phương pháp chạy được với Unsloth

Preference optimization (tối ưu theo sở thích) là nhóm phương pháp căn chỉnh model theo câu trả lời được ưa thích hơn. Theo docs, các phương pháp sau đều chạy được với Unsloth:

| Phương pháp | Tên đầy đủ | Notebook trong docs |
| --- | --- | --- |
| DPO | Direct Preference Optimization | [DPO Zephyr](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb) |
| ORPO | Odds Ratio Preference Optimization | [ORPO Llama 3 (8B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-ORPO.ipynb) |
| KTO | (docs không ghi tên đầy đủ) | [KTO](https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing) |
| SimPO | (docs không ghi tên đầy đủ) | [SimPO](https://colab.research.google.com/drive/1Hs5oQDovOay4mFA6Y9lQhVJ8TnbFLFh2?usp=sharing) |
| PPO, Reward Modelling | | Docs chỉ ghi "chạy được với Unsloth" |

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto

## Dữ liệu cho từng phương pháp


Trang docs Preference Optimization **không** mô tả định dạng dữ liệu cho từng phương pháp. Ví dụ, trang không nói DPO và ORPO cần cột prompt, chosen, rejected, hay KTO cần nhãn tốt và xấu. Điểm này **cần kiểm tra lại** trong notebook tương ứng hoặc tài liệu TRL. Docs có ghi Unsloth xuất hiện trong tài liệu chính thức của Hugging Face cho [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth).

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto

## Code DPO từ docs

Chỉ DPO có code; ORPO và KTO chỉ có notebook.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()
import torch
from trl import DPOTrainer, DPOConfig  # Changed from TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig( # Use DPOConfig
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs",
    ),
    beta = 0.1,
    train_dataset = YOUR_DATASET_HERE,
    # eval_dataset = YOUR_DATASET_HERE,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

dpo_trainer.train()
```

Các điểm đáng chú ý trong đoạn code:

- Bạn phải gọi `PatchDPOTrainer()` trước khi dùng `DPOTrainer` của TRL.
- Model xuất phát là `zephyr-sft-bnb-4bit`, tức một model **đã qua SFT**.
- `ref_model = None` nghĩa là không nạp riêng model tham chiếu.
- `beta = 0.1` là tham số riêng của DPO. Docs không giải thích ý nghĩa của nó.

::: warning Docs chưa thống nhất
Đoạn code DPO trên [trang Preference Optimization](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) không tự chạy được. Lý do: code dùng `max_seq_length` và `YOUR_DATASET_HERE` nhưng không định nghĩa chúng ở đâu. Docs cũng không nêu giá trị hay định dạng dataset cần truyền vào. Hai giá trị này cần kiểm tra lại (ví dụ trong [notebook DPO Zephyr](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb)).
:::

**[Nhận định]** Tham số `tokenizer = tokenizer` của `DPOTrainer` có thể không khớp với các phiên bản TRL mới. Bạn nên đối chiếu với phiên bản TRL mình cài.

**Nguồn:** https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto

## Đọc tiếp

- [Tiết kiệm VRAM khi chạy RL](/reinforcement-learning/memory-efficient) — trang kế tiếp: vì sao RL tốn bộ nhớ GPU và cách Unsloth giảm mức tốn đó.
- [Nên chọn SFT, DPO hay GRPO](/reinforcement-learning/#chon-phuong-phap) — đối chiếu lại DPO với SFT và GRPO trước khi bắt tay vào train.
- [Train bằng GRPO](/reinforcement-learning/grpo) — hướng còn lại khi bạn viết được reward function cho tác vụ.
