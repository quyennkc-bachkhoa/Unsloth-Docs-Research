---
title: Dữ liệu
description: "Chuẩn bị dataset cho fine-tuning với Unsloth: định dạng raw text, Alpaca, ShareGPT, ChatML, lượng dữ liệu, chat template, Data Recipes trong Studio, dữ liệu tổng hợp và checklist."
---

# Dữ liệu

Phần Dữ liệu gồm trang này và bảy trang con. Nếu mới bắt đầu, bạn đọc theo thứ tự từ trên xuống:

- **Dữ liệu** (trang này): dataset cho fine-tuning là gì, và ba câu hỏi nên trả lời trước khi định dạng dữ liệu.
  1. [Các định dạng dữ liệu](/du-lieu/dinh-dang): raw text, Alpaca, ShareGPT, ChatML, vision, bảng nhiều cột và dataset cho model reasoning. Mỗi định dạng có ví dụ chép từ docs, kèm ghi chú chỗ ví dụ chưa phải JSON hợp lệ.
  1. [Bao nhiêu dữ liệu là đủ](/du-lieu/so-luong): các mốc số dòng docs đưa ra, và lượng dữ liệu ảnh hưởng thế nào tới việc chọn base hay instruct model.
  1. [Chat template](/du-lieu/chat-template): áp template của Unsloth trong 4 bước, ánh xạ khóa ShareGPT, thêm token mới, chỉ train trên câu trả lời.
  1. [Nạp dữ liệu trong Studio](/du-lieu/studio): bước Dataset trong Unsloth Studio: nguồn dữ liệu, format và các tùy chọn split, slice, mapping cột.
  1. [Data Recipes](/du-lieu/data-recipes): biến tài liệu của bạn thành dataset bằng workflow dạng node trong Studio.
  1. [Dữ liệu tổng hợp](/du-lieu/synthetic): sinh dữ liệu train bằng LLM, prompt mẫu và việc cần làm sau khi sinh.
  1. [Checklist chuẩn bị dữ liệu](/du-lieu/checklist): danh sách rà lại dataset trước khi train, kèm các cạm bẫy thường gặp.

## Dataset cho fine-tuning là gì

Dataset là tập dữ liệu bạn dùng để train model. Với LLM, văn bản trong dataset phải ở định dạng mà tokenizer (bộ tách văn bản thành token) đọc được. Docs nhấn mạnh hai phần quan trọng nhất khi tạo dataset: **chat template** (khuôn định dạng hội thoại) và **tokenization**.

::: tip Kiến thức nền
Chưa rõ token, tokenizer là gì? Xem [Token và context](/kien-thuc-nen/token-va-context).
:::

Trước khi định dạng dữ liệu, docs khuyên bạn trả lời 3 câu hỏi:

| Câu hỏi | Ví dụ theo docs |
| --- | --- |
| **Mục đích** của dataset | Hội thoại (Q&A, học ngôn ngữ mới, chăm sóc khách hàng); tác vụ có cấu trúc (phân loại, tóm tắt, sinh nội dung); dữ liệu chuyên ngành (y tế, tài chính, kỹ thuật) |
| **Kiểu đầu ra** mong muốn | JSON, HTML, văn bản hay code; tiếng Tây Ban Nha, Anh hay Đức... |
| **Nguồn dữ liệu** | File CSV, PDF, website; Hugging Face và Wikipedia (Wikipedia đặc biệt hữu ích khi dạy model một ngôn ngữ); hoặc dữ liệu tổng hợp |

Một dataset thường gồm 2 cột: câu hỏi và câu trả lời. Chất lượng và số lượng dữ liệu quyết định phần lớn kết quả fine-tune.

Một số lưu ý từ docs:

- Chỉ "đổ" tài liệu thô vào thì hiệu quả không cao bằng dataset được tuyển chọn dạng cặp hỏi–đáp.
- Ngoại lệ là fine-tune cho code: đổ toàn bộ code vào vẫn có thể cải thiện đáng kể.
- Bạn có thể trộn dataset của mình với một dataset tổng quát trên Hugging Face (ví dụ ShareGPT) để model đa dạng hơn.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
