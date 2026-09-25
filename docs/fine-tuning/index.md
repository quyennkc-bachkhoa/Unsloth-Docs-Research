---
title: Fine-tuning
description: Fine-tuning LLM với Unsloth — khi nào nên dùng, LoRA/QLoRA/full fine-tuning, quy trình data → train → đánh giá → export, hyperparameter khuyến nghị, benchmark và các hướng mở rộng.
---

# Fine-tuning

Phần Fine-tuning gồm trang này và chín trang con. Nếu mới bắt đầu, bạn đọc theo thứ tự từ trên xuống:

- **Fine-tuning** (trang này): fine-tuning là gì, các cách fine-tune, khi nào nên dùng thay cho RAG.
  1. [LoRA vs QLoRA vs full fine-tuning](/fine-tuning/lora-qlora-full): ba cách train khác nhau ở phần model được cập nhật, nên tốn VRAM rất khác nhau. Trang này so sánh VRAM, tốc độ và chất lượng, kèm cờ bật tương ứng trong code. Người mới nên bắt đầu từ QLoRA.
  1. [Chọn model để fine-tune](/fine-tuning/chon-model): chọn base hay instruct tùy lượng dữ liệu bạn có (mốc khoảng 300 và 1.000 dòng), và cách đọc hậu tố trong tên model trên Hugging Face. Người mới nên bắt đầu từ một instruct model nhỏ.
  1. [Quy trình data → train → đánh giá → export](/fine-tuning/quy-trinh): đi hết một lượt fine-tune qua 6 bước, đặt cách làm trong Studio (không cần code) cạnh cách làm bằng code Python (Core), kèm đoạn code mẫu cho từng bước.
  1. [Hyperparameter khuyến nghị](/fine-tuning/hyperparameter): bảng giá trị nên dùng cho learning rate, epoch, rank, alpha, batch…, kèm cột "đặt sai thì sao". Docs khuyên giữ mặc định, nhưng mặc định ở Studio, tutorial và hướng dẫn hyperparameter lại khác nhau, nên trang có thêm bảng so sánh.
  1. [Đánh giá và tránh overfitting](/fine-tuning/danh-gia): kiểm tra xem model học được thật hay chỉ học thuộc dữ liệu. Trang này có cách đọc training loss và eval loss, tách tập test, early stopping, và cách xử lý khi overfitting hoặc underfitting.
  1. [Notebooks](/fine-tuning/notebooks): notebook Colab dựng sẵn, bấm chạy là fine-tune được ngay. Có cho SFT cơ bản, hội thoại, vision, continued pretraining, QAT và sinh dữ liệu. Đây là cách nhanh nhất để thử lần đầu.
  1. [Hiệu năng và benchmark](/fine-tuning/benchmark): các con số Unsloth công bố về tốc độ (khoảng 2×), mức giảm VRAM và độ dài context so với Hugging Face + FA2. Trang ghi kèm điều kiện đo để bạn biết con số áp dụng được đến đâu.
  1. [Mở rộng](/fine-tuning/mo-rong): các bước tiếp theo sau khi đã fine-tune cơ bản: train trên nhiều GPU, fine-tune model nhận ảnh, dạy model ngôn ngữ hoặc lĩnh vực mới (continued pretraining), train tiếp từ checkpoint, và QAT để model giữ chất lượng khi lượng tử hóa.
  1. [Lỗi thường gặp](/fine-tuning/loi-thuong-gap): danh sách lỗi hay gặp và cách tránh, như nhảy thẳng vào full fine-tuning, train quá nhiều epoch, tăng batch size rồi bị OOM, hay fine-tune chồng nhiều lần lên cùng một model.

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

**Fine-tuning hay RAG?** Đây là câu hỏi người mới hay gặp. RAG (Retrieval-Augmented Generation) là cách truy xuất tài liệu rồi đưa vào prompt. RAG mạnh khi dữ liệu thay đổi liên tục. Fine-tuning thì đưa kiến thức và hành vi vào thẳng trọng số model. Quan điểm của docs:

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

::: tip Kiến thức nền
Chưa rõ loss, learning rate, epoch, batch, overfitting là gì? Xem [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
