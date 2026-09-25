---
title: Checklist chuẩn bị dữ liệu
description: "Danh sách rà lại dataset trước khi train, kèm các cạm bẫy thường gặp."
---

# Checklist chuẩn bị dữ liệu

Dùng danh sách này để rà lại dataset trước khi bắt đầu train.

1. Đã xác định **mục đích**, **kiểu đầu ra** và **nguồn dữ liệu**.
2. Chọn đúng định dạng theo kiểu train: raw text (CPT), Alpaca/instruction (SFT một lượt), ShareGPT/ChatML (SFT hội thoại), có ảnh (vision).
3. Đủ số lượng. Datasets Guide ghi tối thiểu 100 dòng, tốt nhất trên 1.000 dòng. What Model Should I Use? ghi dưới 300 dòng thì dùng instruct model. Các trang không thống nhất — xem hộp cảnh báo ở trang [Bao nhiêu dữ liệu là đủ](/du-lieu/so-luong).
4. Ưu tiên dữ liệu tuyển chọn dạng hỏi–đáp thay vì đổ tài liệu thô (trừ trường hợp code).
5. Đã làm sạch: bỏ mẫu lạc đề, mẫu kém chất lượng; cân bằng giữa các nhóm.
6. Dữ liệu bảng nhiều cột đã được gộp thành một prompt (`to_sharegpt`, đoạn tùy chọn trong `[[]]`).
7. Đã áp **đúng chat template** của model (`get_chat_template`). Đã quyết định có gọi `standardize_sharegpt` hay không. Docs chưa thống nhất ở điểm này: "Always call this!" hay chỉ gọi khi dataset là ShareGPT mà model cần ChatML — xem hộp cảnh báo ở mục [Dữ liệu dạng bảng nhiều cột](/du-lieu/dinh-dang#bang-nhieu-cot).
8. Token mới (nếu có) được thêm bằng `add_new_tokens` **trước** `get_peft_model`.
9. Đã tách tập eval (ví dụ khoảng 20% dữ liệu) để theo dõi eval loss.
10. Nhiều dataset: chuẩn hóa định dạng rồi gộp, train một lần thay vì fine-tune chồng nhiều lần.
11. Dữ liệu vision: ảnh cùng kích thước, khoảng 300–1000px.
12. Dữ liệu tổng hợp đã được kiểm tra chất lượng thủ công. Có sẵn ít nhất 10 ví dụ mẫu trước khi nhờ LLM sinh thêm.
13. Trong Studio: chọn đúng format (`auto`/`alpaca`/`chatml`/`sharegpt`), kiểm tra Column Mapping, chọn Eval split.

::: warning Cạm bẫy
- Dùng `chat_template` gốc của model có thể dính lỗi template. Docs khuyên dùng bản đã sửa của Unsloth.
- **[Nhận định]** Template lúc train và lúc inference phải khớp nhau. Train với một định dạng rồi chat bằng định dạng khác sẽ cho kết quả kém. Điều này suy ra từ hai điểm: docs nhấn mạnh chat template, và Unsloth tự nhúng template đã dùng khi train vào `Modelfile` khi export sang Ollama.
- `conversation_extension` đặt quá lớn làm train chậm.
- Với multi-image vision, `ds.map(...)` có thể vướng quy tắc chuẩn hóa và quy tắc Arrow khắt khe. Docs khuyên dùng list comprehension thay thế.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/basics/vision-fine-tuning, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/new/studio/start
