---
title: Lỗi thường gặp
description: "Các lỗi hay gặp khi fine-tune và cách tránh."
---

# Lỗi thường gặp

Đây là các lỗi hay gặp khi fine-tune, gom từ nhiều trang docs. Mỗi dòng nêu lỗi và cách tránh.

::: tip Tóm tắt
- **Dùng khi:** bạn sắp train, hoặc kết quả train không như ý và muốn rà nhanh các lỗi phổ biến.
- **Kết quả:** biết các lỗi hay gặp về chọn phương pháp, hyperparameter, chạy và export, cùng cách tránh từng lỗi.
- **Nên biết trước:** [Chọn cách train và model](/fine-tuning/chon-cach-train) và [Chọn hyperparameter](/fine-tuning/hyperparameter).
:::

## Chọn phương pháp và model

Phần lớn lỗi ở bước này đến từ chọn sai cách train hoặc sai loại model.

::: warning Lỗi thường gặp
- **Nhảy thẳng vào full fine-tuning.** Hãy thử LoRA hoặc QLoRA trước. LoRA thất bại thì FFT cũng không cứu được.
- **Lệch độ chính xác giữa train và serve.** Muốn chạy 4-bit thì train 4-bit.
- **Dùng model GGUF để train.** Trong Studio, GGUF bị loại khỏi danh sách train vì chỉ dùng để inference.
- **Bật nhiều cờ độ chính xác cùng lúc** (`load_in_4bit`, `load_in_8bit`, `full_finetuning`...). Mỗi lần chỉ một cờ được `True`.
- **Model gated** (Llama, Gemma) cần Hugging Face token.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/new/studio/start

## Hyperparameter

Các lỗi hyperparameter thường làm model học thuộc, tốn bộ nhớ vô ích hoặc mất chất lượng.

::: warning Lỗi thường gặp
- **Train quá nhiều epoch.** Hơn 3 epoch dễ học thuộc. Loss về gần 0 (hoặc dưới 0.2) là dấu hiệu overfit; kiểm tra bằng tập validation.
- **Bỏ bớt `target_modules` để tiết kiệm bộ nhớ.** Tiết kiệm rất ít mà mất chất lượng.
- **Tăng `per_device_train_batch_size` để "nhanh hơn".** Dễ OOM và có thể chậm hơn do padding. Hãy tăng `gradient_accumulation_steps` thay thế.
- **Fine-tune chồng nhiều lần** lên một model. Việc này có thể làm hỏng những gì lần trước đã học. Docs khuyên gộp dataset và train một lần.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Chạy và export

Các lỗi ở bước chạy và export chủ yếu làm mất thời gian chờ hoặc khiến Studio không cho bắt đầu train.

::: warning Lỗi thường gặp
- **Export GGUF bật `True` ở mọi dòng** trong notebook Ollama. Bạn sẽ chờ rất lâu; chỉ bật một định dạng.
- **Đo tốc độ khi `torch.compile` chưa warm-up** (~5 phút).
- **Studio:** đặt Eval Steps mà không chọn eval split, hoặc ghép model text-only với dataset vision. Studio sẽ báo lỗi validation và khóa nút Start Training.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/basics/unsloth-benchmarks, https://unsloth.ai/docs/new/studio/start

## Đọc tiếp

- [Reinforcement Learning](/reinforcement-learning/) — phần tiếp theo: dạy model bằng điểm thưởng khi SFT chưa đủ.
- [Đánh giá và overfitting](/fine-tuning/danh-gia) — cách xử lý chi tiết khi model học thuộc hoặc học chưa đủ.
- [Export và deploy](/export-deploy/) — export GGUF và chạy model đúng cách sau khi train.
