---
title: Cạm bẫy thường gặp
description: "Các lỗi hay gặp khi fine-tune và cách tránh."
---

# Cạm bẫy thường gặp

Đây là các lỗi hay gặp khi fine-tune, gom từ nhiều trang docs. Mỗi dòng nêu lỗi và cách tránh.

::: warning Cạm bẫy
Về chọn phương pháp và model:
- **Nhảy thẳng vào full fine-tuning.** Hãy thử LoRA hoặc QLoRA trước. LoRA thất bại thì FFT cũng không cứu được.
- **Lệch độ chính xác giữa train và serve.** Muốn chạy 4-bit thì train 4-bit.
- **Dùng model GGUF để train.** Trong Studio, GGUF bị loại khỏi danh sách train vì chỉ dùng để inference.
- **Bật nhiều cờ độ chính xác cùng lúc** (`load_in_4bit`, `load_in_8bit`, `full_finetuning`...). Mỗi lần chỉ một cờ được `True`.
- **Model gated** (Llama, Gemma) cần Hugging Face token.

Về hyperparameter:
- **Train quá nhiều epoch.** Hơn 3 epoch dễ học thuộc. Loss về gần 0 (hoặc dưới 0.2) là dấu hiệu overfit; kiểm tra bằng tập validation.
- **Bỏ bớt `target_modules` để tiết kiệm bộ nhớ.** Tiết kiệm rất ít mà mất chất lượng.
- **Tăng `per_device_train_batch_size` để "nhanh hơn".** Dễ OOM và có thể chậm hơn do padding. Hãy tăng `gradient_accumulation_steps` thay thế.
- **Fine-tune chồng nhiều lần** lên một model. Việc này có thể làm hỏng những gì lần trước đã học. Docs khuyên gộp dataset và train một lần.

Về chạy và export:
- **Export GGUF bật `True` ở mọi dòng** trong notebook Ollama. Bạn sẽ chờ rất lâu; chỉ bật một định dạng.
- **Đo tốc độ khi `torch.compile` chưa warm-up** (~5 phút).
- **Studio:** đặt Eval Steps mà không chọn eval split, hoặc ghép model text-only với dataset vision. Studio sẽ báo lỗi validation và khóa nút Start Training.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/basics/unsloth-benchmarks, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide
