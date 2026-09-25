---
title: Đánh giá và overfitting
description: "Đọc training loss, cách đánh giá, early stopping, xử lý overfitting và underfitting."
---

# Đánh giá và overfitting

Sau khi train, bạn cần biết model đã học được điều bạn muốn hay chỉ học thuộc dữ liệu train. Học thuộc như vậy gọi là overfitting. Trang này nói cách đọc loss, cách đánh giá, và cách xử lý khi model học quá kỹ hoặc chưa đủ.

::: tip Tóm tắt
- **Dùng khi:** bạn đang train hoặc vừa train xong và muốn biết model học thật hay chỉ học thuộc.
- **Kết quả:** đọc được training loss và eval loss, bật được đánh giá và early stopping, biết cần chỉnh gì khi overfitting hoặc underfitting.
- **Nên biết trước:** các bước train ở [Quy trình từng bước](/fine-tuning/quy-trinh) và các tham số ở [Chọn hyperparameter](/fine-tuning/hyperparameter).
:::

## Đọc training loss

Nếu loss không giảm, bạn cần chỉnh cấu hình. Nếu loss quá thấp, đó có thể là overfitting. Tuy vậy, docs đưa ra các ngưỡng khác nhau:

::: warning Docs chưa thống nhất: ngưỡng loss
- "a loss around 0.5 to 1.0 is a good sign"; "If the loss goes to 0, that could mean overfitting" — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- "If your training loss drops below 0.2, your model is likely overfitting" — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

Hai trang đều nói ngưỡng phụ thuộc vào dataset và tác vụ. Cả hai đều khuyên kiểm tra thêm bằng validation loss hoặc eval loss.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

## Các cách đánh giá

Docs gợi ý các cách sau:

- Chat trực tiếp với model xem có đạt ý không. Đây là đánh giá thủ công.
- Tách khoảng 20% dữ liệu train làm tập test. Nếu đã dùng hết dữ liệu để train thì chỉ còn cách đánh giá thủ công.
- Bật evaluation trong Unsloth. Với dataset lớn, bước này có thể chậm. Khi đó bạn giảm kích thước tập eval, hoặc giãn số bước giữa các lần eval.
- Dùng công cụ eval tự động. Cách này dùng được, nhưng có thể không khớp tiêu chí của bạn.
- Trong Studio: chọn **Eval split** để có biểu đồ **Eval Loss**.

::: warning Docs chưa thống nhất: tên tham số số bước giữa các lần eval
Tham số "sau bao nhiêu bước thì eval một lần" mang tên khác nhau tùy trang:

- `evaluation_steps = 100` — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- `eval_steps = 10` (cùng `eval_strategy = "steps"`, trong `SFTConfig`) — [Finetuning from Last Checkpoint](https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint)
- "Eval Steps", mặc định `0` — [Studio](https://unsloth.ai/docs/new/studio/start)

Tên tham số nào đúng với phiên bản TRL/Transformers bạn đang dùng: cần kiểm tra lại.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint, https://unsloth.ai/docs/new/studio/start

## Early stopping theo eval loss

Early stopping là dừng train sớm khi `eval_loss` không giảm nữa sau vài lần eval. Code gồm hai phần: cấu hình trainer để lưu và eval định kỳ, rồi gắn callback dừng sớm.

```python
from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        output_dir = "training_checkpoints", # location of saved checkpoints for early stopping
        save_strategy = "steps",             # save model every N steps
        save_steps = 10,                     # how many steps until we save the model
        save_total_limit = 3,                # keep only 3 saved checkpoints to save disk space
        eval_strategy = "steps",             # evaluate every N steps
        eval_steps = 10,                     # how many steps until we do evaluation
        load_best_model_at_end = True,       # MUST USE for early stopping
        metric_for_best_model = "eval_loss", # metric we want to early stop on
        greater_is_better = False,           # the lower the eval loss, the better
    ),
    model = model,
    tokenizer = tokenizer,
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
)
```

```python
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 3,     # How many steps we will wait if the eval loss doesn't decrease
                                     # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold = 0.0,  # Can set higher - sets how much loss should decrease by until
                                     # we consider early stopping. For eg 0.01 means if loss was
                                     # 0.02 then 0.01, we consider to early stop the run.
)
trainer.add_callback(early_stopping_callback)
```

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint

## Xử lý overfitting và underfitting

Cách xử lý tùy vào việc model học quá kỹ (overfitting) hay chưa đủ (underfitting).

### Khi bị overfitting

Docs gợi ý các cách sau.

Chỉnh cách train:

- Chỉnh learning rate. LR cao dễ overfit trong run ngắn.
- Giảm epoch: dừng sau 1, 2 hoặc 3 epoch.
- Tăng `weight_decay`, bắt đầu từ `0.01` hoặc `0.1`. Tăng `lora_dropout`, ví dụ `0.1`.
- Tăng batch size hoặc gradient accumulation.
- Dùng early stopping theo eval loss.

Chỉnh dữ liệu:

- Mở rộng dataset bằng dataset mã nguồn mở chất lượng cao.

Làm "nhẹ" model sau khi train:

- LoRA alpha scaling: giảm alpha, ví dụ nhân 0.5, sau khi train hoặc lúc inference.
- Weight averaging: cộng trọng số model instruct gốc với bản fine-tune rồi chia 2. Cách này tương đương giảm alpha một nửa.

### Khi bị underfitting

Underfitting là khi model trả lời quá chung chung. Docs gợi ý:

- Chỉnh learning rate.
- Tăng epoch, đồng thời theo dõi validation loss.
- Tăng rank và alpha.
- Dùng dữ liệu sát tác vụ hơn.
- Giảm batch size về 1.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

## Kiểm tra LoRA đã thực sự cập nhật

Đừng dùng `np.allclose()`, vì hàm này có thể bỏ sót thay đổi nhỏ, nhất là ở ma trận LoRA A. Thay vào đó, dùng một trong các cách: checksum hoặc hash (MD5), tổng hiệu tuyệt đối giữa các tensor, thống kê tensor, hoặc `np.array_equal()`.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

## Đọc tiếp

- [Notebook chạy sẵn](/fine-tuning/notebooks) — thử ngay một lượt train trên Colab để quan sát loss thật.
- [Chọn hyperparameter](/fine-tuning/hyperparameter) — tra lại giá trị learning rate, epoch, dropout khi cần chỉnh.
- [Kỹ thuật nâng cao](/fine-tuning/mo-rong) — train tiếp từ checkpoint đã lưu trong lúc eval.
