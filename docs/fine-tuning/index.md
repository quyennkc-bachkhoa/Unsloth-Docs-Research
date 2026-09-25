---
title: Fine-tuning
description: Fine-tuning LLM với Unsloth — khi nào nên dùng, LoRA/QLoRA/full fine-tuning, quy trình data → train → đánh giá → export, hyperparameter khuyến nghị, benchmark và các hướng mở rộng.
---

# Fine-tuning

Phần này nói cách dạy thêm cho một model có sẵn bằng dữ liệu của riêng bạn với Unsloth: chọn cách train và model, đi hết quy trình, chọn hyperparameter, đánh giá kết quả và tránh các lỗi hay gặp.

::: tip Tóm tắt
- **Dùng khi:** bạn muốn model biết thêm kiến thức chuyên ngành, trả lời theo giọng riêng, hoặc làm tốt hơn một tác vụ cụ thể.
- **Kết quả:** hiểu fine-tuning là gì, có những cách nào, khi nào nên dùng thay cho RAG, và biết nên đọc trang con nào tiếp theo.
- **Nên biết trước:** loss, learning rate, epoch, batch, overfitting ở [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen).
:::

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [Chọn cách train và model](/fine-tuning/chon-cach-train) | So sánh LoRA, QLoRA và full fine-tuning về VRAM, tốc độ, chất lượng, kèm cờ bật trong code; chọn base hay instruct theo lượng dữ liệu (mốc khoảng 300 và 1.000 dòng) và đọc hậu tố tên model | Trước khi bắt đầu train; người mới nên bắt đầu từ QLoRA và một instruct model nhỏ |
| [Quy trình từng bước](/fine-tuning/quy-trinh) | Đi hết một lượt fine-tune qua 6 bước, đặt cách làm trong Studio (không cần code) cạnh code Python (Core), kèm code mẫu | Khi fine-tune lần đầu |
| [Chọn hyperparameter](/fine-tuning/hyperparameter) | Bảng giá trị nên dùng cho learning rate, epoch, rank, alpha, batch…, kèm cột "đặt sai thì sao" và bảng so sánh mặc định giữa Studio, tutorial và hướng dẫn hyperparameter | Khi cấu hình một lần train |
| [Đánh giá và overfitting](/fine-tuning/danh-gia) | Đọc training loss và eval loss, tách tập test, early stopping, xử lý overfitting hoặc underfitting | Trong và sau khi train, để biết model học thật hay chỉ học thuộc |
| [Notebook chạy sẵn](/fine-tuning/notebooks) | Notebook Colab dựng sẵn cho SFT cơ bản, hội thoại, vision, continued pretraining, QAT và sinh dữ liệu | Muốn thử lần đầu nhanh nhất |
| [Hiệu năng và benchmark](/fine-tuning/benchmark) | Các con số Unsloth công bố về tốc độ (khoảng 2×), mức giảm VRAM và độ dài context so với Hugging Face + FA2, kèm điều kiện đo | Khi cần ước lượng tốc độ, VRAM hoặc so sánh |
| [Kỹ thuật nâng cao](/fine-tuning/mo-rong) | Train nhiều GPU, fine-tune model nhận ảnh, continued pretraining, train tiếp từ checkpoint, QAT | Sau khi đã fine-tune cơ bản xong |
| [Lỗi thường gặp](/fine-tuning/loi-thuong-gap) | Các lỗi hay gặp và cách tránh: nhảy thẳng vào full fine-tuning, train quá nhiều epoch, tăng batch size rồi OOM, fine-tune chồng nhiều lần | Trước khi train, và khi kết quả không như ý |

## Fine-tuning là gì, khi nào nên dùng

Fine-tuning là dạy thêm cho một model có sẵn bằng dữ liệu của riêng bạn. Bạn cần nó khi muốn model biết thêm kiến thức chuyên ngành, trả lời theo giọng riêng, hoặc làm tốt hơn một tác vụ cụ thể.

Nói chính xác hơn, fine-tuning (tinh chỉnh) là huấn luyện tiếp một model đã được pre-train (huấn luyện trước), ví dụ Llama-3.1-8B, trên dữ liệu chuyên biệt của bạn. Docs gọi chung việc này là fine-tuning, training hoặc post-training. Docs nêu 3 mục tiêu:

- **Cập nhật kiến thức**: đưa vào thông tin chuyên ngành mà model gốc chưa có.
- **Tùy biến hành vi**: chỉnh giọng điệu, tính cách, phong cách trả lời, ví dụ theo giọng thương hiệu.
- **Tối ưu cho tác vụ**: tăng độ chính xác trên một tác vụ cụ thể.

### Có những cách fine-tune nào?

Docs Unsloth kể tên bốn cách "dạy thêm" cho model. Điểm khác nhau chính là **model học từ tín hiệu gì**:

| Cách | Model học từ gì | Phương pháp docs nêu | Đọc thêm |
| --- | --- | --- | --- |
| **SFT** (Supervised Fine-Tuning — tinh chỉnh có giám sát). Docs gọi đây là cách chuẩn. | Cặp đầu vào và đầu ra mẫu. Model học để trả lời giống đầu ra mẫu. | SFT | Các trang con của phần này |
| **Preference optimization** (tối ưu theo sở thích) | Câu trả lời được ưa thích so với câu trả lời bị chê, thay vì một đáp án mẫu duy nhất. | DPO, ORPO | [RL & preference](/kien-thuc-nen/rl-va-preference) |
| **Distillation** (chưng cất) | Kiến thức của một model lớn hơn, được nén vào model nhỏ hơn. | — | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) |
| **RL** (Reinforcement Learning — học tăng cường) | Điểm thưởng hoặc điểm phạt. Model tự thử trả lời, được chấm điểm, rồi học theo hướng được điểm cao. | GRPO, GSPO | [Reinforcement Learning](/reinforcement-learning/) |

Nếu mới bắt đầu, bạn chỉ cần SFT. Docs RL của Unsloth cũng ghi: với đa số trường hợp, SFT là đủ. Các trang con của phần Fine-tuning đều nói về SFT.

Docs đưa ra ba ví dụ ứng dụng:

- Phân tích cảm xúc tin tài chính: đoán một tiêu đề tác động tích cực hay tiêu cực tới công ty.
- Chatbot chăm sóc khách hàng, học từ các hội thoại cũ.
- Trợ lý pháp lý: phân tích hợp đồng, án lệ, tuân thủ.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

## Fine-tuning hay RAG?

Đây là câu hỏi người mới hay gặp. RAG (Retrieval-Augmented Generation) là cách truy xuất tài liệu rồi đưa vào prompt. RAG mạnh khi dữ liệu thay đổi liên tục. Fine-tuning thì đưa kiến thức và hành vi vào thẳng trọng số model. Quan điểm của docs:

| Câu hỏi | Trả lời theo docs |
| --- | --- |
| Fine-tuning có thêm kiến thức mới không? | Có. Nếu dataset chứa thông tin mới, model học được. |
| RAG luôn tốt hơn fine-tuning? | Không nhất thiết. Model tinh chỉnh đúng cách thường ngang hoặc hơn RAG ở tác vụ chuyên biệt. Các nhận định "RAG luôn tốt hơn" thường đến từ cấu hình LoRA sai hoặc train chưa đủ. |
| Fine-tuning có đắt không? | Không nhất thiết. Full fine-tuning và pretraining tốn kém nhưng thường không cần. LoRA hoặc QLoRA chạy được trên notebook Colab, Kaggle miễn phí hoặc máy cá nhân. |
| Nên chọn một trong hai? | Docs khuyên **kết hợp** cả hai: fine-tuning cho chuyên môn và định dạng, RAG cho dữ liệu thay đổi nhanh. Model tinh chỉnh còn là "phương án dự phòng" khi retrieval trả sai. |

Docs liệt kê thêm vài ưu điểm của fine-tuning:

- Không phụ thuộc hệ thống truy xuất lúc inference.
- Trả lời nhanh hơn vì bỏ bước retrieval.
- Kiểm soát chặt giọng điệu.

Docs cũng tuyên bố "fine-tuning can replicate all of RAG's capabilities, but not vice versa".

**[Nhận định]** Câu "fine-tuning làm được mọi thứ RAG làm" là quan điểm của Unsloth. Với dữ liệu thay đổi hằng ngày, bạn sẽ phải retrain liên tục, nên RAG vẫn là lựa chọn thực tế hơn. Chính docs cũng khuyên kết hợp. Xem thêm [Ứng dụng RAG](/ung-dung-rag).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

## Đọc tiếp

- [Chọn cách train và model](/fine-tuning/chon-cach-train) — bước đầu tiên: chọn LoRA, QLoRA hay full fine-tuning và chọn model gốc.
- [Notebook chạy sẵn](/fine-tuning/notebooks) — cách nhanh nhất để thử fine-tune lần đầu mà không phải tự viết code.
- [Ứng dụng RAG](/ung-dung-rag) — khi dữ liệu thay đổi liên tục và bạn muốn kết hợp RAG với model đã fine-tune.
