---
title: Bao nhiêu dữ liệu là đủ
description: "Các mốc số dòng dữ liệu theo docs, và lượng dữ liệu ảnh hưởng thế nào đến việc chọn base hay instruct model."
---

# Bao nhiêu dữ liệu là đủ

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

Chi tiết ở trang [Chọn model để fine-tune](/fine-tuning/chon-model).

::: warning Docs chưa thống nhất: lượng dữ liệu và loại model
Các trang đưa ra những mốc và khuyến nghị khác nhau:
- Tối thiểu "at least 100 rows"; tốt nhất "over 1,000 rows" — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- Trên 1.000 dòng → base; 300–1.000 dòng → base hoặc instruct; "Less than 300 Rows" → instruct — [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- "We recommend starting with Instruct models" (không kèm điều kiện về lượng dữ liệu) — [Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use)
- Khi sinh dữ liệu tổng hợp bằng LLM: cần sẵn "at least 10 examples" — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
:::

Docs nhấn mạnh rằng hiệu quả phụ thuộc rất lớn vào **chất lượng** dữ liệu. Vì vậy bạn cần làm sạch và chuẩn bị dữ liệu kỹ.

**Nhiều dataset:** chuẩn hóa định dạng rồi gộp thành một dataset, hoặc dùng [notebook Multiple Datasets](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing).

**Fine-tune một model nhiều lần?** Làm được. Nhưng docs khuyên gộp mọi dataset và train một lần. Lý do: train chồng lên model đã fine-tune có thể làm thay đổi chất lượng và kiến thức đã học ở lần trước.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
