---
title: Kiến thức nền LLM
description: Các khái niệm về cấu tạo, huấn luyện và suy luận của mô hình ngôn ngữ mà docs Unsloth dùng nhưng không giải thích.
---

# Kiến thức nền LLM

Route này giải thích những khái niệm mà docs Unsloth dùng nhưng không giải thích. Mức độ vừa đủ để bạn đọc hiểu docs và tự chọn tham số, không đi sâu vào toán ML.

Docs Unsloth viết cho người đã quen với LLM (Large Language Model, mô hình ngôn ngữ lớn). Vì vậy nhiều khái niệm xuất hiện mà không có lời giải thích, ví dụ `r`, `lora_alpha`, `UD-Q4_K_XL`, "A3B", KV cache hay `min_p`.

## Đọc route này khi nào

Bạn không cần đọc hết route. Tìm việc bạn sắp làm ở cột đầu, đọc nhóm trang tương ứng, rồi sang trang Unsloth ở cột cuối.

| Bạn sắp làm gì | Đọc trước | Trang Unsloth tương ứng |
| --- | --- | --- |
| Cài Unsloth, chọn model vừa với máy | Nhóm A và B | [Cài đặt & phần cứng](/cai-dat), [Model catalog](/model-catalog) |
| Chạy model, gọi API, chỉnh sampling | Nhóm D | [Inference & API](/inference) |
| Fine-tune bằng LoRA/QLoRA | Nhóm C (hai trang đầu) | [Fine-tuning](/fine-tuning/), [Dữ liệu](/du-lieu) |
| Dùng GRPO, DPO | Nhóm C (trang RL) | [Reinforcement Learning](/reinforcement-learning) |
| Export GGUF, NVFP4 | Nhóm B | [Export & deploy](/export-deploy) |

Thứ tự gợi ý nằm ở trang [Lộ trình học](/lo-trinh-hoc).

## Các nhóm

Các trang được chia thành bốn nhóm, đi từ cấu tạo model đến lúc dùng model:

- **A. Cấu tạo mô hình:** [Token & context](/kien-thuc-nen/token-va-context), [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh), [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer), [Dense & MoE](/kien-thuc-nen/dense-va-moe).
- **B. Số học & bộ nhớ:** [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) (có công cụ ước tính VRAM), [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
- **C. Huấn luyện:** [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora), [RL & preference](/kien-thuc-nen/rl-va-preference).
- **D. Suy luận:** [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).

## Bảng tra khái niệm

Gặp một khái niệm lạ trong docs Unsloth? Tra ở bảng này để biết trang nào giải thích nó, và trang Unsloth nào dùng tới nó.

| Khái niệm | Trang giải thích | Trang Unsloth dùng khái niệm |
| --- | --- | --- |
| Token, tokenizer, vocabulary | [Token & context](/kien-thuc-nen/token-va-context) | [Dữ liệu](/du-lieu), [Inference & API](/inference) |
| Context window, `max_seq_length` | [Token & context](/kien-thuc-nen/token-va-context) | [Fine-tuning](/fine-tuning/), [Model catalog](/model-catalog) |
| Base vs Instruct | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) | [Fine-tuning](/fine-tuning/) |
| Vision / multimodal | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) | [Model catalog](/model-catalog), [Fine-tuning](/fine-tuning/) |
| Embedding model | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) | [Ứng dụng RAG](/ung-dung-rag) |
| Attention, GQA, FFN, RoPE, RMSNorm | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) | [Fine-tuning](/fine-tuning/hyperparameter) (target modules) |
| `q_proj` … `down_proj` | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) | [Fine-tuning](/fine-tuning/) |
| Dense, MoE, "A3B", tham số kích hoạt | [Dense & MoE](/kien-thuc-nen/dense-va-moe) | [Model catalog](/model-catalog) |
| Số tham số, VRAM, RAM, offload | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) | [Cài đặt & phần cứng](/cai-dat), [Model catalog](/model-catalog) |
| Tên file `UD-Q4_K_XL`, `-bnb-4bit` | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) | [Model catalog](/model-catalog), [Export & deploy](/export-deploy) |
| BF16, FP8, NVFP4, MXFP4, INT4 | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) | [Export & deploy](/export-deploy), [Reinforcement Learning](/reinforcement-learning) |
| GGUF quant, Dynamic quants | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) | [Export & deploy](/export-deploy), [Model catalog](/model-catalog) |
| Loss, learning rate, epoch, batch | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) | [Fine-tuning](/fine-tuning/) |
| Optimizer AdamW 8-bit, scheduler, warmup | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) | [Fine-tuning](/fine-tuning/) |
| Overfitting, eval loss | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) | [Fine-tuning](/fine-tuning/) |
| LoRA `r`, `lora_alpha`, QLoRA, merge | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) | [Fine-tuning](/fine-tuning/), [Export & deploy](/export-deploy) |
| Gradient checkpointing | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) | [Fine-tuning](/fine-tuning/), [Reinforcement Learning](/reinforcement-learning) |
| SFT, reward, DPO, GRPO, reward hacking | [RL & preference](/kien-thuc-nen/rl-va-preference) | [Reinforcement Learning](/reinforcement-learning) |
| KV cache | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference), [Model catalog](/model-catalog) |
| Temperature, top-p, top-k, min-p | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference), [Model catalog](/model-catalog) |
| Chat template | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Dữ liệu](/du-lieu), [Export & deploy](/export-deploy) |
| Tool calling | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference) |

Danh sách đầy đủ các thuật ngữ nằm ở trang [Thuật ngữ](/thuat-ngu).

## Quy ước nhãn

Các trang trong route gắn nhãn để bạn biết mỗi thông tin đến từ đâu:

- **[Nguồn ngoài]:** nội dung lấy từ nguồn nằm ngoài docs Unsloth, có kèm URL. Nguồn ngoài gồm paper arXiv, tài liệu chính thức của Hugging Face, PyTorch, llama.cpp, NVIDIA và model card chính thức.
- **[Nhận định]:** ý kiến của người viết, không có trong nguồn.
- **[Ước tính]:** con số do người viết tự tính, dựa trên một công thức có nguồn.
- **Docs chưa thống nhất:** các trang docs Unsloth ghi giá trị khác nhau cho cùng một thứ. Website liệt kê đủ mọi giá trị và không tự chọn giá trị nào.
