---
title: Kiến thức nền LLM
description: Các khái niệm về cấu tạo, huấn luyện và suy luận của mô hình ngôn ngữ mà docs Unsloth dùng nhưng không giải thích.
---

# Kiến thức nền LLM

Docs Unsloth được viết cho người đã quen làm việc với LLM (Large Language Model, mô hình ngôn ngữ lớn). Vì vậy docs dùng nhiều thuật ngữ mà không dừng lại giải thích: `r`, `lora_alpha`, `UD-Q4_K_XL`, "A3B", KV cache, `min_p`…

Phần này giải thích những thuật ngữ đó. Mức độ vừa đủ để bạn đọc hiểu docs và tự chọn tham số cho máy của mình. Không có toán ML chuyên sâu.

## Bắt đầu từ đâu

Bạn không cần đọc hết từ đầu đến cuối. Hãy chọn theo việc bạn sắp làm:

| Bạn sắp… | Nên đọc trước | Rồi làm theo hướng dẫn ở |
| --- | --- | --- |
| Cài Unsloth và chọn model vừa với máy | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa), cùng các trang về cấu tạo mô hình | [Cài đặt & phần cứng](/cai-dat), [Model catalog](/model-catalog) |
| Chạy model, gọi API, chỉnh sampling | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference/) |
| Fine-tune bằng LoRA hoặc QLoRA | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) | [Fine-tuning](/fine-tuning/), [Dữ liệu](/du-lieu/) |
| Train bằng GRPO hoặc DPO | [RL & preference](/kien-thuc-nen/rl-va-preference) | [Reinforcement Learning](/reinforcement-learning/) |
| Export sang GGUF hoặc NVFP4 | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) | [Export & deploy](/export-deploy/) |

Nếu muốn học theo thứ tự từ đầu, xem [Lộ trình học](/lo-trinh-hoc).

## Nội dung phần này

Mười trang được xếp thành bốn nhóm. Thứ tự đi từ bên trong model, qua cách train, đến lúc dùng model.

### A. Cấu tạo mô hình

- [Token & context](/kien-thuc-nen/token-va-context): model đọc văn bản theo đơn vị nào, và một lần đọc được bao nhiêu.
- [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh): base hay instruct, text hay multimodal, cách đọc tên model trên Hugging Face.
- [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer): các khối bên trong một LLM, và chúng ứng với `target_modules` nào khi fine-tune.
- [Dense & MoE](/kien-thuc-nen/dense-va-moe): ký hiệu kiểu "30B-A3B" nghĩa là gì, vì sao MoE nhanh mà vẫn tốn bộ nhớ.

### B. Số học & bộ nhớ

- [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho): model cần bao nhiêu RAM, VRAM khi chạy và khi fine-tune. Trang có công cụ ước tính VRAM.
- [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa): BF16, FP8, 4-bit, các mức quant GGUF, và bạn đánh đổi gì khi nén model.

### C. Huấn luyện

- [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen): loss, learning rate, epoch, batch, overfitting, cách đọc biểu đồ loss.
- [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora): LoRA hoạt động thế nào, ý nghĩa từng tham số `r`, `lora_alpha`, `target_modules`.
- [RL & preference](/kien-thuc-nen/rl-va-preference): SFT khác RL ở đâu, DPO và GRPO khác nhau thế nào, khi nào dùng cách nào.

### D. Suy luận

- [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling): model sinh từng token ra sao, temperature và top-p làm gì, chat template, tool calling.

## Tra nhanh một thuật ngữ

Gặp một thuật ngữ lạ trong docs Unsloth? Tìm nó ở cột đầu để biết trang nào giải thích, và bạn sẽ gặp nó ở đâu trong phần Unsloth.

| Thuật ngữ | Giải thích ở | Gặp trong |
| --- | --- | --- |
| Token, tokenizer, vocabulary | [Token & context](/kien-thuc-nen/token-va-context) | [Dữ liệu](/du-lieu/), [Inference & API](/inference/) |
| Context window, `max_seq_length` | [Token & context](/kien-thuc-nen/token-va-context) | [Fine-tuning](/fine-tuning/), [Model catalog](/model-catalog) |
| Base vs Instruct | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) | [Fine-tuning](/fine-tuning/) |
| Vision / multimodal | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) | [Model catalog](/model-catalog), [Fine-tuning](/fine-tuning/) |
| Embedding model | [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) | [Ứng dụng RAG](/ung-dung-rag) |
| Attention, GQA, FFN, RoPE, RMSNorm | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) | [Fine-tuning](/fine-tuning/hyperparameter) (target modules) |
| `q_proj` … `down_proj` | [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) | [Fine-tuning](/fine-tuning/) |
| Dense, MoE, "A3B", tham số kích hoạt | [Dense & MoE](/kien-thuc-nen/dense-va-moe) | [Model catalog](/model-catalog) |
| Số tham số, VRAM, RAM, offload | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) | [Cài đặt & phần cứng](/cai-dat), [Model catalog](/model-catalog) |
| Tên file `UD-Q4_K_XL`, `-bnb-4bit` | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) | [Model catalog](/model-catalog), [Export & deploy](/export-deploy/) |
| BF16, FP8, NVFP4, MXFP4, INT4 | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) | [Export & deploy](/export-deploy/), [Reinforcement Learning](/reinforcement-learning/) |
| GGUF quant, Dynamic quants | [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) | [Unsloth Dynamic GGUF](/export-deploy/gguf#dynamic-gguf), [Model catalog](/model-catalog) |
| Loss, learning rate, epoch, batch | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) | [Fine-tuning](/fine-tuning/) |
| Optimizer AdamW 8-bit, scheduler, warmup | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) | [Fine-tuning](/fine-tuning/) |
| Overfitting, eval loss | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) | [Fine-tuning](/fine-tuning/) |
| LoRA `r`, `lora_alpha`, QLoRA, merge | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) | [Fine-tuning](/fine-tuning/), [Export & deploy](/export-deploy/) |
| Gradient checkpointing | [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) | [Fine-tuning](/fine-tuning/), [Reinforcement Learning](/reinforcement-learning/) |
| SFT, reward, DPO, GRPO, reward hacking | [RL & preference](/kien-thuc-nen/rl-va-preference) | [Reinforcement Learning](/reinforcement-learning/) |
| KV cache | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference/), [Model catalog](/model-catalog) |
| Temperature, top-p, top-k, min-p | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference/), [Model catalog](/model-catalog) |
| Chat template | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Chat template](/du-lieu/chat-template), [Export & deploy](/export-deploy/) |
| Tool calling | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) | [Inference & API](/inference/) |

Không thấy thuật ngữ bạn cần? Xem bảng đầy đủ ở trang [Thuật ngữ](/thuat-ngu).

## Các nhãn bạn sẽ gặp

Mỗi thông tin trong phần này đều cho biết nó đến từ đâu. Câu không có nhãn lấy từ docs Unsloth. Các trường hợp khác được đánh dấu như sau:

- **[Nguồn ngoài]:** lấy từ nguồn ngoài docs Unsloth, có kèm URL. Đó là paper arXiv, tài liệu chính thức của Hugging Face, PyTorch, llama.cpp, NVIDIA, hoặc model card chính thức của hãng.
- **[Nhận định]:** ý kiến của người viết, không có trong nguồn nào.
- **[Ước tính]:** con số người viết tự tính, dựa trên một công thức có nguồn.
- **Docs chưa thống nhất:** các trang docs Unsloth ghi khác nhau về cùng một thứ. Trang liệt kê đủ mọi giá trị kèm link, không tự chọn giá trị nào.
