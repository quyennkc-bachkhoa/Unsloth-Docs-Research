---
title: Dữ liệu tổng hợp
description: "Sinh dữ liệu train bằng LLM: các cách làm theo docs, prompt mẫu và việc cần làm sau khi sinh."
---

# Dữ liệu tổng hợp (synthetic data)

Dữ liệu tổng hợp là dữ liệu do một LLM sinh ra. Docs nêu 3 mục tiêu khi dùng nó:

- Tạo dữ liệu hoàn toàn mới, từ đầu hoặc từ dataset sẵn có.
- Đa dạng hóa dataset để model không overfit và không quá hẹp.
- Tăng cường dữ liệu có sẵn, ví dụ tự động cấu trúc lại dataset theo đúng định dạng đã chọn.

**Các cách làm theo docs:**

| Cách | Ghi chú |
| --- | --- |
| **Data Recipes** trong Unsloth Studio | Upload dữ liệu có hoặc không cấu trúc, tự chuyển thành dataset dùng được hoặc dataset tổng hợp (xem [Data Recipes](/du-lieu/data-recipes)) |
| **Notebook Synthetic Dataset** | Tự parse tài liệu (PDF, video...), sinh cặp QA và tự làm sạch bằng model local như Llama 3.2: [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_\(3B\).ipynb) |
| **LLM local hoặc ChatGPT** | Ví dụ Llama 3.3 (70B) hay GPT 4.5; model lớn hơn cho chất lượng cao hơn. Có thể chạy qua vLLM, Ollama, llama.cpp, nhưng bạn phải tự thu thập đầu ra và tự viết prompt thêm |

Prompt mẫu trong docs, theo từng tình huống:

- Sinh thêm hội thoại từ dataset có sẵn: "Using the dataset example I provided, follow the structure and generate conversations based on the examples."
- Chưa có dataset:

```
Create 10 examples of product reviews for Coca-Coca classified as either positive, negative, or neutral.
```

- Dataset chưa được định dạng:

```
Structure my dataset so it is in a QA ChatML format for fine-tuning. Then generate 5 synthetic data examples with the same topic and format.
```

Sau khi sinh xong, bạn cần:

- Kiểm tra chất lượng, loại bỏ hoặc sửa các câu trả lời lạc đề hay kém.
- Cân bằng dữ liệu để tránh overfit.
- Nếu muốn, đưa dataset đã làm sạch trở lại LLM để sinh tiếp với hướng dẫn tốt hơn.

**[Nhận định]** Khi dùng dịch vụ thương mại (ví dụ ChatGPT) để sinh dữ liệu train, bạn nên kiểm tra điều khoản sử dụng của nhà cung cấp. Docs không đề cập điểm này.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/new/studio/data-recipe
