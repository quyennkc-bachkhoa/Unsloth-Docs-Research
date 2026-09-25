---
title: Chọn cách train và model
description: "Chọn cách train (LoRA, QLoRA hay full fine-tuning) và chọn model (base hay instruct) trước khi fine-tune."
---

# Chọn cách train và model

Trước khi fine-tune, bạn cần quyết định hai việc: train theo cách nào, và bắt đầu từ model nào. Trang này đi qua lần lượt hai quyết định đó.

::: tip Tóm tắt
- **Dùng khi:** bạn sắp fine-tune và chưa biết nên dùng LoRA, QLoRA hay full fine-tuning, hoặc nên bắt đầu từ model nào.
- **Kết quả:** chọn được cách train vừa với VRAM (người mới: QLoRA), chọn base hay instruct theo lượng dữ liệu, và đọc được hậu tố tên model trên Hugging Face.
- **Nên biết trước:** [LoRA và QLoRA](/kien-thuc-nen/lora-va-qlora) (rank/alpha, gradient checkpointing), [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) (4-bit/16-bit), [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) (base vs instruct), [Tham số và bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) (đọc "8B", ước tính VRAM).
:::

## LoRA, QLoRA hay full fine-tuning {#lora-qlora-full}

Ba cách này khác nhau ở chỗ bạn train bao nhiêu phần của model, và vì thế tốn bao nhiêu VRAM. Docs khuyên người mới bắt đầu từ QLoRA.

- **LoRA** (Low-Rank Adaptation) giữ nguyên trọng số model gốc. Nó chỉ train các ma trận adapter (bộ chuyển đổi) mỏng gắn thêm vào model. Docs nói cách này chỉ tối ưu khoảng 1% trọng số.
- **QLoRA** là LoRA chạy trên model gốc đã được lượng tử hóa xuống 4-bit.
- **Full fine-tuning** (FFT) cập nhật toàn bộ trọng số.

| Tiêu chí | QLoRA (4-bit) | LoRA (16-bit) | Full fine-tuning |
| --- | --- | --- | --- |
| Model gốc | Lượng tử hóa 4-bit + LoRA adapter | Độ chính xác đầy đủ (16-bit) + LoRA adapter | Train toàn bộ trọng số |
| VRAM (bộ nhớ card đồ họa) | Thấp nhất. Docs ghi mức giảm so với LoRA theo nhiều cách (xem hộp "Docs chưa thống nhất" bên dưới) | Trung bình. Docs ghi gấp 4× QLoRA | Cao nhất |
| Tốc độ | Chậm hơn LoRA một chút | Nhanh hơn QLoRA một chút | Docs chỉ ghi "compute-heavy", cần nhiều tài nguyên hơn hẳn |
| Chất lượng | Kém LoRA một chút. Docs ghi mức chênh khác nhau (xem hộp "Docs chưa thống nhất" bên dưới) | Chính xác hơn QLoRA một chút | Docs: LoRA làm đúng có thể ngang FFT |
| Ví dụ theo docs | Llama 70B vừa dưới 48GB VRAM với QLoRA trong Unsloth | — | — |
| Cờ trong code | `load_in_4bit = True` | `load_in_4bit = False` hoặc `load_in_16bit = True` | `full_finetuning = True` |
| Docs khuyên | **Bắt đầu từ đây** | Khi có môi trường 16-bit và cần độ chính xác tối đa | Thường không cần; thử LoRA hoặc QLoRA trước |

Ngoài ra còn chế độ 8-bit (`load_in_8bit = True`). Mỗi lần bạn chỉ được bật **một** phương pháp là `True`.

::: warning Docs chưa thống nhất: QLoRA tiết kiệm bao nhiêu VRAM và mất bao nhiêu độ chính xác
Các trang docs đưa ra những cách diễn đạt khác nhau cho hai câu hỏi này. Mục này liệt kê đủ, không chọn thay bạn.

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

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/new/studio/start

## Chọn model để fine-tune {#chon-model}

Chọn xong cách train, việc tiếp theo là chọn model gốc: model bạn chọn quyết định phần lớn kết quả và lượng VRAM cần dùng. Docs gợi ý đi theo các bước sau:

1. **Chọn theo use-case.** Train với ảnh thì dùng vision model, ví dụ Llama 3.2 Vision. Dataset code thì dùng model chuyên code, ví dụ Qwen Coder 2.5. 
2. **Đánh giá tài nguyên.** Dùng hướng dẫn VRAM của Unsloth để ước lượng. Dataset quyết định loại model và thời gian train.
3. **Ưu tiên model mới nhất.** Xem [Danh sách model hỗ trợ](/model-catalog).
4. **Chọn Instruct hay Base** theo lượng dữ liệu bạn có:

| Lượng dữ liệu | Khuyến nghị |
| --- | --- |
| Trên 1.000 dòng | Thường nên fine-tune **base** model |
| 300–1.000 dòng chất lượng cao | Base hoặc instruct đều được |
| Dưới 300 dòng | **Instruct** model thường tốt hơn (giữ được khả năng làm theo chỉ dẫn sẵn có) |

Hai loại này hợp với kiểu template khác nhau:

- **Instruct model** dùng chat template hội thoại (ChatML, ShareGPT) và cần ít dữ liệu hơn.
- **Base model** hợp với template dạng instruction (Alpaca, Vicuna).

Với người mới, docs khuyên bắt đầu từ **instruct model nhỏ** như Llama 3.1 (8B). Nếu có điều kiện, bạn fine-tune thử cả hai rồi so sánh.

::: warning Docs chưa thống nhất: chọn instruct hay base, và cần bao nhiêu dữ liệu
Ba trang docs đưa lời khuyên khác nhau về cùng một chuyện:

- "We recommend starting with **Instruct models**" (không kèm điều kiện về lượng dữ liệu) — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- Trên 1.000 dòng: "generally best to fine-tune the base model"; 300–1.000 dòng: base hoặc instruct; dưới 300 dòng: instruct — [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- Tối thiểu "at least 100 rows"; tối ưu "over 1,000 rows" (không gắn với việc chọn base hay instruct) — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
:::

**Đọc hậu tố tên model trên Hugging Face.** Phần đuôi tên model cho bạn biết nó đã được lượng tử hóa kiểu gì:

| Hậu tố | Ý nghĩa |
| --- | --- |
| `unsloth-bnb-4bit` | Unsloth Dynamic 4-bit quant. Tốn VRAM hơn BitsAndBytes 4-bit thường một chút nhưng chính xác hơn đáng kể |
| `bnb-4bit` (không có "unsloth") | BitsAndBytes 4-bit tiêu chuẩn |
| Không hậu tố | Bản gốc 16-bit hoặc 8-bit. Bản của Unsloth đôi khi có sửa lỗi chat template hoặc tokenizer, nên docs khuyên dùng bản của Unsloth |

Ví dụ tên model: `unsloth/llama-3.1-8b-unsloth-bnb-4bit`.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Đọc tiếp

- [Quy trình từng bước](/fine-tuning/quy-trinh) — đem cách train và model vừa chọn vào một lượt fine-tune đầy đủ.
- [Chọn hyperparameter](/fine-tuning/hyperparameter) — đặt rank, alpha, learning rate cho LoRA hoặc QLoRA.
- [Danh sách model hỗ trợ](/model-catalog) — tìm model mới nhất để làm model gốc.
