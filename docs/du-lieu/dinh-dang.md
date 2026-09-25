---
title: Định dạng dữ liệu
description: "Các định dạng dataset theo kiểu train: raw text, Alpaca, ShareGPT, ChatML, vision, bảng CSV/Excel và dataset cho model reasoning."
---

# Định dạng dữ liệu

Mỗi kiểu train cần dữ liệu được sắp theo một khuôn riêng, ví dụ văn bản thô, cặp chỉ dẫn và đáp án, hay hội thoại nhiều lượt. Trang này giúp bạn chọn đúng định dạng và xem ví dụ chép từ docs, kèm ghi chú những chỗ ví dụ chưa phải JSON hợp lệ.

::: tip Tóm tắt
- **Dùng khi:** bạn đã biết mình định train kiểu gì (CPT, SFT, RL, vision) và cần sắp dữ liệu cho đúng khuôn.
- **Kết quả:** biết định dạng nào hợp với kiểu train nào, có ví dụ raw text, Alpaca, ShareGPT, ChatML, vision, bảng CSV/Excel và dataset cho model reasoning.
- **Nên biết trước:** [Dữ liệu](/du-lieu/), [Chọn cách train và model](/fine-tuning/chon-cach-train).
:::

## Chọn định dạng theo kiểu train

Định dạng dữ liệu phụ thuộc vào kiểu train bạn định làm. Bảng dưới ghép từng loại dữ liệu với kiểu train tương ứng:

| Định dạng | Mô tả | Kiểu train |
| --- | --- | --- |
| Raw Corpus | Văn bản thô từ website, sách, bài báo | Continued Pretraining (CPT — huấn luyện tiếp trên văn bản thô) |
| Instruct | Chỉ dẫn cho model + ví dụ đầu ra mong muốn | SFT (Supervised Fine-Tuning — tinh chỉnh có giám sát) |
| Conversation | Hội thoại nhiều lượt giữa user và trợ lý AI | SFT |
| RLHF | Hội thoại mà câu trả lời của trợ lý được chấm hạng bởi script, model khác hoặc người | RL (Reinforcement Learning — học tăng cường) |

Mỗi loại lại có nhiều biến thể định dạng. Các ví dụ trong các mục dưới đây chép nguyên văn từ docs.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Raw text (cho continued pretraining)

Raw text không cần cấu trúc. Bạn giữ mạch văn tự nhiên để model học từ văn bản liên tục:

```json
  "text": "Pasta carbonara is a traditional Roman pasta dish. The sauce is made by mixing raw eggs with grated Pecorino Romano cheese and black pepper. The hot pasta is then tossed with crispy guanciale (cured pork cheek) and the egg mixture, creating a creamy sauce from the residual heat. Despite popular belief, authentic carbonara never contains cream or garlic. The dish likely originated in Rome in the mid-20th century, though its exact origins are debated..."
```

::: warning Lỗi thường gặp
Ví dụ trên không phải JSON hợp lệ: ví dụ raw text trong [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) chỉ là một cặp khóa–giá trị, thiếu dấu `{}` bao ngoài. Nếu bạn copy làm file dữ liệu, mỗi bản ghi cần là một object JSON đầy đủ có trường `"text"`.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Instruction (Alpaca)

Dùng định dạng này khi bạn muốn model trả lời một lượt theo một chỉ dẫn cụ thể:

```json
"Instruction": "Task we want the model to perform."

"Input": "Optional, but useful, it will essentially be the user's query."

"Output": "The expected result of the task and the output of the model."
```

Alpaca dataset gốc có 52.000 cặp chỉ dẫn và đầu ra do GPT-4 sinh ra. Dataset có 3 cột `instruction`, `input`, `output`. Khi train, mỗi dòng được ghép thành một prompt lớn (supervised instruction finetuning). Bản GPT-4: https://huggingface.co/datasets/vicgalle/alpaca-gpt4. Phần lớn notebook của Unsloth dùng Alpaca dataset.

::: warning Docs chưa thống nhất: tên cột Alpaca, và ví dụ không phải JSON hợp lệ
Các trang viết tên cột Alpaca theo hai kiểu:
- Viết hoa: `"Instruction"`, `"Input"`, `"Output"` — [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- Viết thường: cột `instruction` / `input` / `output` (format `alpaca`) — [Studio](https://unsloth.ai/docs/new/studio/start); cột "instruction", "input", "output" — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)

Ví dụ Alpaca ở trên cũng chỉ là các cặp khóa–giá trị rời. Nó thiếu `{}` và thiếu dấu phẩy giữa các trường, nên không parse được nếu bạn copy nguyên văn thành file JSON.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama

## Conversation — ShareGPT

ShareGPT dùng cho hội thoại nhiều lượt. Mỗi lượt có khóa `"from"`/`"value"`, và các lượt luân phiên giữa `human` và `gpt`:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Can you help me make pasta carbonara?"
    },
    {
      "from": "gpt",
      "value": "Would you like the traditional Roman recipe, or a simpler version?"
    },
    {
      "from": "human",
      "value": "The traditional version please"
    },
    {
      "from": "gpt",
      "value": "The authentic Roman carbonara uses just a few ingredients: pasta, guanciale, eggs, Pecorino Romano, and black pepper. Would you like the detailed recipe?"
    }
  ]
}
```

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Conversation — ChatML (OpenAI)

ChatML là định dạng mặc định của Hugging Face. Theo docs, đây có lẽ là định dạng phổ biến nhất. Mỗi lượt có khóa `"role"`/`"content"`, và các lượt luân phiên giữa `user` và `assistant`:

```
{
  "messages": [
    {
      "role": "user",
      "content": "What is 1+1?"
    },
    {
      "role": "assistant",
      "content": "It's 2!"
    },
  ]
}
```

::: warning Lỗi thường gặp
Ví dụ ChatML trên không phải JSON hợp lệ: ví dụ chép nguyên văn từ [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) có một dấu phẩy thừa sau phần tử cuối của mảng `"messages"` (`},` trước `]`). JSON chuẩn không cho phép dấu phẩy này. Nếu bạn copy làm file dữ liệu, hãy bỏ dấu phẩy đó đi.
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Vision (ảnh + chữ)

Dữ liệu vision giống cặp hỏi–đáp, chỉ khác là phần input có thêm ảnh. Mọi tác vụ vision phải có dạng:

```python
[
{ "role": "user",
  "content": [{"type": "text",  "text": instruction}, {"type": "image", "image": image} ]
},
{ "role": "assistant",
  "content": [{"type": "text",  "text": answer} ]
},
]
```

Ví dụ trong docs dùng bản rút gọn của ROCO radiography dataset: 1978 dòng, gồm các cột `image`, `image_id`, `caption`, `cui`. Mỗi mẫu được chuyển sang dạng `messages` bằng hàm `convert_to_conversation`. Xem thêm mục Vision ở trang [Kỹ thuật nâng cao](/fine-tuning/mo-rong).

::: warning Docs chưa thống nhất: thứ tự text/ảnh trong lượt user
Thứ tự phần chữ và phần ảnh khác nhau giữa dữ liệu train và ví dụ inference:
- Dữ liệu train: `{"type": "text", ...}` đứng trước `{"type": "image", ...}` — khuôn định dạng và hàm `convert_to_conversation` trong [Vision Fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning) và [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- Ví dụ inference ngay sau đó trên cùng hai trang: `{"type": "image"}` đứng trước `{"type": "text", ...}` — [Vision Fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning), [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)

Docs không nói thứ tự này có ảnh hưởng hay không: cần kiểm tra lại với chat template của model bạn dùng.
:::

**Nguồn:** https://unsloth.ai/docs/basics/vision-fine-tuning, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Dữ liệu dạng bảng nhiều cột (CSV/Excel) {#bang-nhieu-cot}

Trợ lý kiểu ChatGPT chỉ nhận **một** prompt. Vì vậy dataset nhiều cột (ví dụ Titanic: tuổi, hạng vé, giá vé...) phải được gộp thành một prompt. Unsloth có hàm `to_sharegpt` làm việc này. Cách khai báo:

- Tên cột đặt trong ngoặc nhọn `{}`, đúng tên cột trong file CSV hoặc Excel.
- Đoạn tùy chọn đặt trong `[[]]`. Nếu cột trống, cả đoạn bị bỏ qua. Cách này hữu ích khi thiếu dữ liệu.
- Cột đích (đầu ra) khai báo ở `output_column_name`. Với Alpaca, cột này là `output`.

Ví dụ: với dòng thiếu giá vé, prompt sẽ không có câu "Their fare is EMPTY", vì cả đoạn đó bị lược bỏ.

Tham số `conversation_extension` chọn ngẫu nhiên N dòng và ghép chúng thành một hội thoại nhiều lượt. Ví dụ đặt 3 thì 3 dòng được gộp lại. Đặt giá trị lớn thì train chậm hơn, nhưng chatbot có thể tốt hơn.

Tutorial dặn gọi `standardize_sharegpt` sau bước này (xem hộp cảnh báo bên dưới). Notebook CSV/Excel: https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing

::: warning Docs chưa thống nhất: khi nào gọi `standardize_sharegpt`
Hai chỗ trong docs nói khác nhau về việc có luôn phải gọi hàm này:
- "We then use the `standardize_sharegpt` function ... Always call this!" (sau `to_sharegpt` / `conversation_extension`) — [Tutorial Llama-3 + Ollama](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama), [Chat Templates](https://unsloth.ai/docs/basics/chat-templates), [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- "Only use the standardize_sharegpt method if your target dataset is formatted in the sharegpt format, but your model expect a ChatML format instead." — mục Q&A của [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
:::

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama, https://unsloth.ai/docs/basics/chat-templates, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Dataset cho model reasoning

Cách chuẩn bị dữ liệu phụ thuộc vào việc model đã biết suy luận (reasoning) hay chưa:

- Model **đã** có reasoning (ví dụ DeepSeek-R1-Distill-Llama-8B): bạn vẫn dùng cặp câu hỏi–trả lời. Nhưng câu trả lời phải chứa quá trình suy luận (chain-of-thought) và các bước dẫn tới đáp án.
- Model **chưa** có reasoning và bạn muốn dạy nó suy luận: dùng dataset thường, câu trả lời không kèm suy luận, rồi train bằng RL/GRPO — xem [Train bằng GRPO](/reinforcement-learning/grpo).

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide

## Đọc tiếp

- [Chat template](/du-lieu/chat-template) — trang kế tiếp: ghép dữ liệu đã định dạng vào đúng khuôn hội thoại của model.
- [Làm dữ liệu trong Studio](/du-lieu/studio) — chọn format và mapping cột bằng giao diện thay vì code.
- [Train bằng GRPO](/reinforcement-learning/grpo) — khi bạn muốn dạy model chưa biết suy luận bằng RL.
