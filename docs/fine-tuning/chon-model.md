---
title: Chọn model để fine-tune
description: "Chọn model theo use-case, tài nguyên và lượng dữ liệu; base hay instruct; đọc hậu tố tên model trên Hugging Face."
---

# Chọn model để fine-tune

Model bạn chọn quyết định phần lớn kết quả và lượng VRAM cần dùng. Docs gợi ý đi theo các bước sau:

1. **Chọn theo use-case.** Train với ảnh thì dùng vision model, ví dụ Llama 3.2 Vision. Dataset code thì dùng model chuyên code, ví dụ Qwen Coder 2.5. Kiểm tra license và yêu cầu hệ thống.
2. **Đánh giá tài nguyên.** Dùng hướng dẫn VRAM của Unsloth để ước lượng. Dataset quyết định loại model và thời gian train.
3. **Ưu tiên model mới nhất.** Xem [Model catalog](/model-catalog).
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

::: tip Kiến thức nền
Base vs instruct là gì? Xem [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh). Cách đọc "8B" và ước tính VRAM: [Tham số và bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
:::

**Đọc hậu tố tên model trên Hugging Face.** Phần đuôi tên model cho bạn biết nó đã được lượng tử hóa kiểu gì:

| Hậu tố | Ý nghĩa |
| --- | --- |
| `unsloth-bnb-4bit` | Unsloth Dynamic 4-bit quant. Tốn VRAM hơn BitsAndBytes 4-bit thường một chút nhưng chính xác hơn đáng kể |
| `bnb-4bit` (không có "unsloth") | BitsAndBytes 4-bit tiêu chuẩn |
| Không hậu tố | Bản gốc 16-bit hoặc 8-bit. Bản của Unsloth đôi khi có sửa lỗi chat template hoặc tokenizer, nên docs khuyên dùng bản của Unsloth |

Ví dụ tên model: `unsloth/llama-3.1-8b-unsloth-bnb-4bit`.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
