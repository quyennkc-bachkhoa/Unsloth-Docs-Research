---
title: Sinh dữ liệu tổng hợp
description: "Sinh dữ liệu train bằng LLM: các cách làm theo docs, prompt mẫu và việc cần làm sau khi sinh."
---

# Sinh dữ liệu tổng hợp

Dữ liệu tổng hợp là dữ liệu do một LLM sinh ra thay vì do người viết tay. Trang này giúp bạn chọn cách sinh, dùng prompt mẫu từ docs và biết phải làm gì với dữ liệu sau khi sinh.

::: tip Tóm tắt
- **Dùng khi:** dataset của bạn quá nhỏ, chưa có dữ liệu, hoặc dữ liệu chưa đúng định dạng.
- **Kết quả:** biết ba cách sinh dữ liệu theo docs (Data Recipes, notebook Synthetic Dataset, LLM local hoặc ChatGPT), có prompt mẫu cho từng tình huống và danh sách việc cần làm sau khi sinh.
- **Nên biết trước:** [Cần bao nhiêu dữ liệu](/du-lieu/#so-luong), [Làm dữ liệu trong Studio](/du-lieu/studio#data-recipes).
:::

## Dữ liệu tổng hợp dùng để làm gì

Docs nêu 3 mục tiêu khi dùng dữ liệu tổng hợp:

- Tạo dữ liệu hoàn toàn mới, từ đầu hoặc từ dataset sẵn có.
- Đa dạng hóa dataset để model không overfit và không quá hẹp.
- Tăng cường dữ liệu có sẵn, ví dụ tự động cấu trúc lại dataset theo đúng định dạng đã chọn.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Các cách làm theo docs

Có ba cách, từ dùng giao diện Studio tới tự chạy một LLM:

| Cách | Ghi chú |
| --- | --- |
| **Data Recipes** trong Unsloth Studio | Upload dữ liệu có hoặc không cấu trúc, tự chuyển thành dataset dùng được hoặc dataset tổng hợp (xem [Data Recipes](/du-lieu/studio#data-recipes)) |
| **Notebook Synthetic Dataset** | Tự parse tài liệu (PDF, video...), sinh cặp QA và tự làm sạch bằng model local như Llama 3.2: [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_\(3B\).ipynb) |
| **LLM local hoặc ChatGPT** | Ví dụ Llama 3.3 (70B) hay GPT 4.5; model lớn hơn cho chất lượng cao hơn. Có thể chạy qua vLLM, Ollama, llama.cpp, nhưng bạn phải tự thu thập đầu ra và tự viết prompt thêm |

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/new/studio/data-recipe

## Prompt mẫu

Docs đưa prompt mẫu cho từng tình huống:

- Sinh thêm hội thoại từ dataset có sẵn: "Using the dataset example I provided, follow the structure and generate conversations based on the examples."
- Chưa có dataset:

```
Create 10 examples of product reviews for Coca-Coca classified as either positive, negative, or neutral.
```

- Dataset chưa được định dạng:

```
Structure my dataset so it is in a QA ChatML format for fine-tuning. Then generate 5 synthetic data examples with the same topic and format.
```

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Việc cần làm sau khi sinh

Dữ liệu sinh ra chưa dùng ngay được; sau khi sinh xong, bạn cần:

- Kiểm tra chất lượng, loại bỏ hoặc sửa các câu trả lời lạc đề hay kém.
- Cân bằng dữ liệu để tránh overfit.
- Nếu muốn, đưa dataset đã làm sạch trở lại LLM để sinh tiếp với hướng dẫn tốt hơn.

**[Nhận định]** Khi dùng dịch vụ thương mại (ví dụ ChatGPT) để sinh dữ liệu train, bạn nên kiểm tra điều khoản sử dụng của nhà cung cấp. Docs không đề cập điểm này.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

## Đọc tiếp

- [Checklist chuẩn bị dữ liệu](/du-lieu/checklist) — trang kế tiếp: rà lại toàn bộ dataset, kể cả phần dữ liệu tổng hợp, trước khi train.
- [Làm dữ liệu trong Studio](/du-lieu/studio#data-recipes) — dựng quy trình sinh dữ liệu bằng Data Recipes.
- [Định dạng dữ liệu](/du-lieu/dinh-dang) — định dạng đích (ví dụ ChatML) mà bạn nhờ LLM sắp dữ liệu theo.
