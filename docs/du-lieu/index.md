---
title: Dữ liệu
description: "Chuẩn bị dataset cho fine-tuning với Unsloth: định dạng raw text, Alpaca, ShareGPT, ChatML, lượng dữ liệu, chat template, Data Recipes trong Studio, dữ liệu tổng hợp và checklist."
---

# Dữ liệu

Dataset là tập ví dụ bạn đưa cho model học khi fine-tune, và chất lượng của nó quyết định phần lớn kết quả. Phần này giúp bạn chọn định dạng, áp chat template, biết cần bao nhiêu dòng và rà lại dataset trước khi train.

::: tip Tóm tắt
- **Dùng khi:** bạn sắp fine-tune và cần chuẩn bị, định dạng hoặc kiểm tra lại dataset.
- **Kết quả:** biết dataset cho fine-tuning là gì, ba câu hỏi cần trả lời trước khi định dạng, cần khoảng bao nhiêu dòng, và nên đọc trang con nào tiếp.
- **Nên biết trước:** [Token và context](/kien-thuc-nen/token-va-context) (token, tokenizer là gì).
:::

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [Định dạng dữ liệu](/du-lieu/dinh-dang) | Chọn định dạng theo kiểu train: raw text, Alpaca, ShareGPT, ChatML, vision, bảng nhiều cột, dataset cho model reasoning; ví dụ chép từ docs kèm ghi chú chỗ chưa phải JSON hợp lệ | Khi bắt đầu gom hoặc chuyển đổi dữ liệu |
| [Chat template](/du-lieu/chat-template) | Áp template của Unsloth trong 4 bước, ánh xạ khóa ShareGPT, thêm token mới, chỉ train trên câu trả lời | Khi dữ liệu đã có và cần ghép hội thoại đúng khuôn của model |
| [Làm dữ liệu trong Studio](/du-lieu/studio) | Nạp dataset ở bước Dataset của Unsloth Studio (nguồn, format, split, slice, mapping cột) và biến tài liệu thành dataset bằng Data Recipes | Khi bạn train bằng giao diện Studio thay vì viết code |
| [Sinh dữ liệu tổng hợp](/du-lieu/synthetic) | Sinh dữ liệu train bằng LLM, prompt mẫu và việc cần làm sau khi sinh | Khi dataset quá nhỏ hoặc chưa có dữ liệu |
| [Checklist chuẩn bị dữ liệu](/du-lieu/checklist) | Rà lại dataset trước khi train, kèm các lỗi thường gặp | Ngay trước khi bấm train |

## Dataset cho fine-tuning là gì

Dataset là tập dữ liệu bạn dùng để train model. Với LLM, văn bản trong dataset phải ở định dạng mà tokenizer (bộ tách văn bản thành token) đọc được. Docs nhấn mạnh hai phần quan trọng nhất khi tạo dataset: **chat template** (khuôn định dạng hội thoại) và **tokenization**.

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

## Cần bao nhiêu dữ liệu {#so-luong}

Lượng dữ liệu ảnh hưởng tới hai việc: kết quả fine-tune, và loại model bạn nên chọn. Các mốc docs đưa ra:

| Mốc | Theo docs |
| --- | --- |
| Tối thiểu | Ít nhất **100 dòng** để có kết quả hợp lý |
| Tốt nhất | Trên **1.000 dòng**; khi đó càng nhiều dữ liệu thường càng tốt |
| Dataset quá nhỏ | Bổ sung dữ liệu tổng hợp hoặc trộn thêm dataset từ Hugging Face |
| Sinh dữ liệu tổng hợp bằng LLM | Cần sẵn ít nhất **10 ví dụ** để model học cấu trúc |

Về chọn model:

- Trên 1.000 dòng: thường nên fine-tune base model.
- 300–1.000 dòng chất lượng cao: base hay instruct đều được.
- Dưới 300 dòng: instruct model thường tốt hơn.

Chi tiết ở trang [Chọn model để fine-tune](/fine-tuning/chon-cach-train#chon-model).

::: warning Docs chưa thống nhất: lượng dữ liệu và loại model
Các trang đưa ra những mốc và khuyến nghị khác nhau:
- Tối thiểu "at least 100 rows"; tốt nhất "over 1,000 rows" — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- Trên 1.000 dòng → base; 300–1.000 dòng → base hoặc instruct; "Less than 300 Rows" → instruct — [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- "We recommend starting with Instruct models" (không kèm điều kiện về lượng dữ liệu) — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- Khi sinh dữ liệu tổng hợp bằng LLM: cần sẵn "at least 10 examples" — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
:::

Docs nhấn mạnh rằng hiệu quả phụ thuộc rất lớn vào **chất lượng** dữ liệu. Vì vậy bạn cần làm sạch và chuẩn bị dữ liệu kỹ.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

## Nhiều dataset và fine-tune nhiều lần

Nếu có nhiều dataset, cách docs khuyên là gộp lại rồi train một lần.

### Gộp nhiều dataset

Chuẩn hóa định dạng rồi gộp thành một dataset, hoặc dùng [notebook Multiple Datasets](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing).

### Fine-tune một model nhiều lần

Làm được. Nhưng docs khuyên gộp mọi dataset và train một lần. Lý do: train chồng lên model đã fine-tune có thể làm thay đổi chất lượng và kiến thức đã học ở lần trước.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Đọc tiếp

- [Định dạng dữ liệu](/du-lieu/dinh-dang) — trang kế tiếp: chọn định dạng dataset theo kiểu train và xem ví dụ từ docs.
- [Chat template](/du-lieu/chat-template) — ghép dữ liệu đúng khuôn hội thoại của model trước khi train.
- [Sinh dữ liệu tổng hợp](/du-lieu/synthetic) — khi dataset chưa đủ các mốc số dòng ở trên.
- [Checklist chuẩn bị dữ liệu](/du-lieu/checklist) — rà lại toàn bộ trước khi train.
