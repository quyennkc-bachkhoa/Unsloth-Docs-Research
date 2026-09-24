---
title: LoRA vs QLoRA vs full fine-tuning
description: "Ba cách fine-tune khác nhau ở phần model được train và lượng VRAM cần; cờ bật trong code và mâu thuẫn trong docs về mức tiết kiệm."
---

# LoRA vs QLoRA vs full fine-tuning

Ba cách này khác nhau ở chỗ bạn train bao nhiêu phần của model, và vì thế tốn bao nhiêu VRAM. Docs khuyên người mới bắt đầu từ QLoRA.

- **LoRA** (Low-Rank Adaptation) giữ nguyên trọng số model gốc. Nó chỉ train các ma trận adapter (bộ chuyển đổi) mỏng gắn thêm vào model. Docs nói cách này chỉ tối ưu khoảng 1% trọng số.
- **QLoRA** là LoRA chạy trên model gốc đã được lượng tử hóa xuống 4-bit.
- **Full fine-tuning** (FFT) cập nhật toàn bộ trọng số.

::: tip Kiến thức nền
Chưa rõ LoRA rank/alpha, QLoRA, gradient checkpointing là gì? Xem [LoRA và QLoRA](/kien-thuc-nen/lora-va-qlora). Về 4-bit/16-bit: [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

| Tiêu chí | QLoRA (4-bit) | LoRA (16-bit) | Full fine-tuning |
| --- | --- | --- | --- |
| Model gốc | Lượng tử hóa 4-bit + LoRA adapter | Độ chính xác đầy đủ (16-bit) + LoRA adapter | Train toàn bộ trọng số |
| VRAM (bộ nhớ card đồ họa) | Thấp nhất. Docs ghi mức giảm so với LoRA theo nhiều cách (xem hộp cảnh báo bên dưới) | Trung bình. Docs ghi gấp 4× QLoRA | Cao nhất |
| Tốc độ | Chậm hơn LoRA một chút | Nhanh hơn QLoRA một chút | Docs chỉ ghi "compute-heavy", cần nhiều tài nguyên hơn hẳn |
| Chất lượng | Kém LoRA một chút. Docs ghi mức chênh khác nhau (xem hộp cảnh báo bên dưới) | Chính xác hơn QLoRA một chút | Docs: LoRA làm đúng có thể ngang FFT |
| Ví dụ theo docs | Llama 70B vừa dưới 48GB VRAM với QLoRA trong Unsloth | — | — |
| Cờ trong code | `load_in_4bit = True` | `load_in_4bit = False` hoặc `load_in_16bit = True` | `full_finetuning = True` |
| Docs khuyên | **Bắt đầu từ đây** | Khi có môi trường 16-bit và cần độ chính xác tối đa | Thường không cần; thử LoRA hoặc QLoRA trước |

Ngoài ra còn chế độ 8-bit (`load_in_8bit = True`). Mỗi lần bạn chỉ được bật **một** phương pháp là `True`.

::: warning Docs chưa thống nhất: QLoRA tiết kiệm bao nhiêu VRAM và mất bao nhiêu độ chính xác
Các trang docs đưa ra những cách diễn đạt khác nhau cho hai câu hỏi này. Trang này liệt kê đủ, không chọn thay bạn.

**Mức tiết kiệm VRAM của QLoRA (4-bit) so với LoRA 16-bit:**
- "4× less" VRAM; LoRA "4× more than QLoRA" — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- "reducing VRAM usage by over 75%" — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- "quantizes to 4-bit to save 75% memory"; `load_in_4bit = True` "reducing memory use 4×" — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- "reduces memory usage by 4x, allowing us to actually do finetuning in a free 16GB memory GPU" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)

**Độ chính xác mất đi khi dùng 4-bit:**
- "1-2% accuracy degradation" (với `load_in_4bit = True`) — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- "Slightly slower and marginally less accurate" — [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- Với Unsloth dynamic 4-bit, phần mất mát "is now negligible" — [FAQ + Is Fine-tuning Right For Me?](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me)
- Với Unsloth dynamic 4-bit, phần mất mát "is now largely recovered" — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
:::

Docs có hai lời khuyên đáng nhớ:

- **Train và serve cùng độ chính xác.** Serve là lúc model phục vụ inference. Muốn chạy 4-bit thì train 4-bit, và ngược lại.
- **Đừng nhảy thẳng vào FFT.** Nếu LoRA hoặc QLoRA không chạy được, gần như chắc chắn FFT cũng không. FFT không "tự sửa" được lỗi đó.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/new/studio/start
